from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np
import torch
from omegaconf import OmegaConf


class AgentRayTrainerDataloaderTest(unittest.TestCase):
    def test_create_dataloader_uses_config_driven_tool_dataset_signature(self):
        from training.verl import agent_ray_trainer as trainer_module

        calls = []

        class FakeDataset:
            def __init__(
                self,
                *,
                data_files,
                tokenizer,
                config,
                processor=None,
                max_samples=-1,
                tool_env=None,
                use_custom_tool_format_func=False,
            ):
                calls.append(
                    {
                        "data_files": data_files,
                        "tokenizer": tokenizer,
                        "config": config,
                        "processor": processor,
                        "max_samples": max_samples,
                        "tool_env": tool_env,
                        "use_custom_tool_format_func": use_custom_tool_format_func,
                    }
                )

            def __len__(self):
                return 3

        class FakeLoader:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def __len__(self):
                if self.kwargs["batch_size"] == len(self.kwargs["dataset"]):
                    return 1
                return 2

        trainer = trainer_module.RayAgentTrainer.__new__(trainer_module.RayAgentTrainer)
        trainer.tokenizer = object()
        trainer.processor = object()
        trainer.env = object()
        trainer.val_env = object()
        trainer.config = OmegaConf.create(
            {
                "data": {
                    "train_files": "./datasets/grpo/train.parquet",
                    "val_files": "./datasets/grpo/validation.parquet",
                    "train_batch_size": 4,
                    "shuffle": False,
                    "return_raw_chat": False,
                    "use_custom_tool_format_func": True,
                },
                "trainer": {"total_epochs": 3, "total_training_steps": None},
                "actor_rollout_ref": {"actor": {"optim": {"total_training_steps": None}}},
                "critic": {"optim": {"total_training_steps": None}},
            }
        )

        with mock.patch.object(trainer_module, "ToolRLDataset", FakeDataset), mock.patch.object(
            trainer_module, "StatefulDataLoader", side_effect=lambda **kwargs: FakeLoader(**kwargs)
        ):
            trainer._create_dataloader()

        self.assertEqual(2, len(calls))
        self.assertEqual("./datasets/grpo/train.parquet", calls[0]["data_files"])
        self.assertIs(trainer.config.data, calls[0]["config"])
        self.assertIs(trainer.env, calls[0]["tool_env"])
        self.assertTrue(calls[0]["use_custom_tool_format_func"])
        self.assertEqual("./datasets/grpo/validation.parquet", calls[1]["data_files"])
        self.assertIs(trainer.val_env, calls[1]["tool_env"])

    def test_compute_advantage_uses_verl_071_grpo_signature(self):
        from training.verl import agent_ray_trainer as trainer_module

        data = trainer_module.DataProto.from_dict(
            tensors={
                "token_level_rewards": torch.ones((2, 3), dtype=torch.float32),
                "responses": torch.ones((2, 3), dtype=torch.long),
                "attention_mask": torch.ones((2, 6), dtype=torch.long),
            },
            non_tensors={"uid": np.array(["a", "a"], dtype=object)},
        )

        with mock.patch.object(
            trainer_module.core_algos,
            "compute_grpo_outcome_advantage",
            return_value=(
                torch.ones((2, 3), dtype=torch.float32),
                torch.ones((2, 3), dtype=torch.float32),
            ),
        ) as mocked:
            trainer_module.compute_advantage(
                data,
                trainer_module.AdvantageEstimator.GRPO,
            )

        response_mask = mocked.call_args.kwargs.get("response_mask")
        if response_mask is None:
            response_mask = mocked.call_args.args[1]
        self.assertEqual(response_mask.shape, (2, 3))
        self.assertNotIn("eos_mask", mocked.call_args.kwargs)

    def test_compute_advantage_prefers_explicit_response_mask(self):
        from training.verl import agent_ray_trainer as trainer_module

        explicit_response_mask = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.long)
        data = trainer_module.DataProto.from_dict(
            tensors={
                "token_level_rewards": torch.ones((2, 3), dtype=torch.float32),
                "responses": torch.ones((2, 3), dtype=torch.long),
                "response_mask": explicit_response_mask,
                "attention_mask": torch.ones((2, 6), dtype=torch.long),
            },
            non_tensors={"uid": np.array(["a", "a"], dtype=object)},
        )

        with mock.patch.object(
            trainer_module.core_algos,
            "compute_grpo_outcome_advantage",
            return_value=(
                torch.ones((2, 3), dtype=torch.float32),
                torch.ones((2, 3), dtype=torch.float32),
            ),
        ) as mocked:
            trainer_module.compute_advantage(
                data,
                trainer_module.AdvantageEstimator.GRPO,
            )

        response_mask = mocked.call_args.kwargs["response_mask"]
        self.assertTrue(torch.equal(response_mask, explicit_response_mask))

    def test_init_workers_uses_actor_as_ref_for_lora(self):
        from training.verl import agent_ray_trainer as trainer_module

        trainer = trainer_module.RayAgentTrainer.__new__(trainer_module.RayAgentTrainer)
        trainer.hybrid_engine = True
        trainer.role_worker_mapping = {
            trainer_module.Role.ActorRollout: "actor_cls",
            trainer_module.Role.RefPolicy: "ref_cls",
        }
        trainer.config = OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "hybrid_engine": True,
                    "model": {"lora_rank": 16},
                    "rollout": {"agent": {}, "checkpoint_engine": {}},
                },
                "trainer": {},
            }
        )
        trainer.use_reference_policy = True
        trainer.ref_in_actor = True
        trainer.use_critic = False
        trainer.use_rm = False
        trainer.resource_pool_manager = mock.Mock()
        trainer.resource_pool_manager.resource_pool_dict = {"global_pool": "pool"}
        trainer.resource_pool_manager.get_resource_pool.return_value = "pool"
        trainer.ray_worker_group_cls = mock.Mock()
        trainer.wg_dicts = []

        fake_actor_wg = mock.Mock()
        fake_actor_wg.init_model = mock.Mock()
        fake_actor_wg.compute_ref_log_prob = mock.Mock()
        fake_spawned = {"actor_rollout": fake_actor_wg}

        fake_wg_dict = mock.Mock()
        fake_wg_dict.spawn.return_value = fake_spawned
        trainer.ray_worker_group_cls.return_value = fake_wg_dict

        fake_async_rollout_manager = mock.Mock()
        fake_async_rollout_manager.rollout_replicas = []
        fake_checkpoint_manager = mock.Mock()

        with mock.patch.object(trainer_module, "RayClassWithInitArgs", side_effect=lambda *args, **kwargs: {"args": args, "kwargs": kwargs}), mock.patch.object(
            trainer_module, "create_colocated_worker_cls", return_value="worker_dict_cls"
        ), mock.patch.object(
            trainer_module, "omega_conf_to_dataclass", return_value=mock.Mock(checkpoint_engine={})
        ), mock.patch.dict(
            "sys.modules",
            {"verl.experimental.agent_loop": mock.Mock(AgentLoopManager=mock.Mock(create=mock.Mock(return_value=fake_async_rollout_manager)))},
        ), mock.patch.object(
            trainer_module, "CheckpointEngineManager", return_value=fake_checkpoint_manager
        ):
            trainer.init_workers()

        self.assertNotIn("ref", trainer.resource_pool_to_cls["pool"])
        self.assertIs(trainer.ref_policy_wg, fake_actor_wg)


if __name__ == "__main__":
    unittest.main()
