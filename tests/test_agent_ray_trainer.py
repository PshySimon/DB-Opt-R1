from types import SimpleNamespace
import unittest
from unittest import mock
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


if __name__ == "__main__":
    unittest.main()
