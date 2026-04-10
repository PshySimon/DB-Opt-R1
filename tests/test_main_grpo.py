import sys
import types
import unittest
from unittest import mock
import importlib.util
import io
import json
import tempfile
from contextlib import redirect_stdout

import numpy as np
import torch
from verl import DataProto
from omegaconf import OmegaConf

from training.verl import main_grpo


class MainGrpoWorkerSelectionTest(unittest.TestCase):
    def test_build_worker_components_uses_async_worker_for_fsdp(self):
        config = OmegaConf.create(
            {
                "actor_rollout_ref": {"actor": {"strategy": "fsdp"}},
                "critic": {"strategy": "fsdp"},
            }
        )

        fake_async_worker = type("AsyncActorRolloutRefWorker", (), {})
        fake_critic_worker = type("CriticWorker", (), {})
        fake_worker_group = type("RayWorkerGroup", (), {})

        fake_fsdp_module = types.SimpleNamespace(
            AsyncActorRolloutRefWorker=fake_async_worker,
            CriticWorker=fake_critic_worker,
        )
        fake_ray_module = types.SimpleNamespace(RayWorkerGroup=fake_worker_group)

        with mock.patch.dict(
            sys.modules,
            {
                "verl.workers.fsdp_workers": fake_fsdp_module,
                "verl.single_controller.ray": fake_ray_module,
            },
        ), mock.patch.object(main_grpo.ray, "remote", side_effect=lambda cls: cls):
            role_worker_mapping, ray_worker_group_cls = main_grpo._build_worker_components(config)

        self.assertIs(fake_async_worker, role_worker_mapping[main_grpo.Role.ActorRollout])
        self.assertIs(fake_async_worker, role_worker_mapping[main_grpo.Role.RefPolicy])
        self.assertIs(fake_critic_worker, role_worker_mapping[main_grpo.Role.Critic])
        self.assertIs(fake_worker_group, ray_worker_group_cls)

    def test_db_reward_manager_returns_nonzero_answer_score_for_valid_set_knob_trajectory(self):
        solution = (
            "<|im_start|>assistant\n"
            "<think>set knob</think>\n"
            '<tool_call>{"name":"set_knob","arguments":{"knobs":"{\\"shared_buffers\\": \\"8GB\\", \\"work_mem\\": \\"64MB\\"}"}}</tool_call>\n'
            "<|im_end|>"
        )

        class FakeTokenizer:
            pad_token_id = 0

            def decode(self, ids, skip_special_tokens=False):
                if isinstance(ids, list) and ids == [self.pad_token_id]:
                    return "<pad>"
                return solution

        class FakeCostModel:
            def predict(self, knobs, hardware):
                return 120.0 if knobs.get("shared_buffers") == "8GB" else 100.0

        batch = DataProto.from_dict(
            tensors={
                "prompts": torch.tensor([[1, 2]], dtype=torch.long),
                "responses": torch.tensor([[3, 4, 5]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long),
            },
            non_tensors={
                "reward_model": np.array(
                    [{"ground_truth": {"hardware": {"total_memory_gb": 80.0}}}],
                    dtype=object,
                ),
                "data_source": np.array(["test"], dtype=object),
            },
        )

        reward_tensor, answer_scores, format_scores = main_grpo.DBRewardManager(
            tokenizer=FakeTokenizer(),
            cost_model=FakeCostModel(),
        )(batch)

        self.assertGreater(answer_scores[0], 0.0)
        self.assertGreater(format_scores[0], 0.0)
        self.assertGreater(reward_tensor[0, -1].item(), format_scores[0])

    def test_db_reward_manager_debug_output_includes_extracted_knobs_and_component_scores(self):
        solution = (
            "<|im_start|>assistant\n"
            "<think>set knob</think>\n"
            '<tool_call>{"name":"set_knob","arguments":{"knobs":"{\\"shared_buffers\\": \\"8GB\\"}"}}</tool_call>\n'
            "<|im_end|>"
        )

        class FakeTokenizer:
            pad_token_id = 0

            def decode(self, ids, skip_special_tokens=False):
                if isinstance(ids, list) and ids == [self.pad_token_id]:
                    return "<pad>"
                return solution

        class FakeCostModel:
            def predict(self, knobs, hardware):
                return 120.0 if knobs.get("shared_buffers") == "8GB" else 100.0

        batch = DataProto.from_dict(
            tensors={
                "prompts": torch.tensor([[1, 2]], dtype=torch.long),
                "responses": torch.tensor([[3, 4, 5]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long),
            },
            non_tensors={
                "reward_model": np.array(
                    [{"ground_truth": {"hardware": {"total_memory_gb": 80.0}}}],
                    dtype=object,
                ),
                "data_source": np.array(["debug"], dtype=object),
            },
        )

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            main_grpo.DBRewardManager(
                tokenizer=FakeTokenizer(),
                num_examine=1,
                cost_model=FakeCostModel(),
            )(batch)

        output = stdout.getvalue()
        self.assertIn("[extracted_knobs]", output)
        self.assertIn("shared_buffers", output)
        self.assertIn("[answer_score]", output)
        self.assertIn("[format_score]", output)
        self.assertIn("[response_only]", output)
        self.assertIn("[tool_calls]", output)

    def test_db_reward_manager_writes_debug_rollout_jsonl(self):
        solution = (
            "<|im_start|>assistant\n"
            "<think>set knob</think>\n"
            '<tool_call>{"name":"set_knob","arguments":{"knobs":"{\\"shared_buffers\\": \\"8GB\\"}"}}</tool_call>\n'
            "<|im_end|>"
        )

        class FakeTokenizer:
            pad_token_id = 0

            def decode(self, ids, skip_special_tokens=False):
                if isinstance(ids, list) and ids == [self.pad_token_id]:
                    return "<pad>"
                return solution

        class FakeCostModel:
            def predict(self, knobs, hardware):
                return 120.0 if knobs.get("shared_buffers") == "8GB" else 100.0

        batch = DataProto.from_dict(
            tensors={
                "prompts": torch.tensor([[1, 2]], dtype=torch.long),
                "responses": torch.tensor([[3, 4, 5]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long),
            },
            non_tensors={
                "reward_model": np.array(
                    [{"ground_truth": {"hardware": {"total_memory_gb": 80.0}}}],
                    dtype=object,
                ),
                "data_source": np.array(["debug"], dtype=object),
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            main_grpo.DBRewardManager(
                tokenizer=FakeTokenizer(),
                num_examine=1,
                cost_model=FakeCostModel(),
                debug_rollout_dir=tmpdir,
                rollout_split="train",
                experiment_name="exp-a",
            )(batch)

            with open(f"{tmpdir}/exp-a/train.jsonl", "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["split"], "train")
        self.assertIn("shared_buffers", rows[0]["extracted_knobs"])
        self.assertGreater(rows[0]["answer_score"], 0.0)

    def test_verify_grpo_reward_path_module_finds_positive_candidate(self):
        spec = importlib.util.spec_from_file_location(
            "verify_grpo_reward_path",
            "scripts/verify_grpo_reward_path.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        class FakeCostModel:
            def predict(self, knobs, hardware):
                return 120.0 if knobs.get("shared_buffers") == "8GB" else 100.0

        ground_truth = {"hardware": {"total_memory_gb": 80.0, "cpu_count": 16}}
        knobs, score = module.find_positive_knob_config(
            ground_truth=ground_truth,
            cost_model=FakeCostModel(),
            knob_space_path="configs/knob_space.yaml",
            max_random_candidates=0,
        )

        self.assertEqual(knobs["shared_buffers"], "8GB")
        self.assertGreater(score, 0.0)

    def test_verify_grpo_reward_path_builds_batch_with_nonempty_prompt_mask(self):
        spec = importlib.util.spec_from_file_location(
            "verify_grpo_reward_path",
            "scripts/verify_grpo_reward_path.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        class FakeTokenizer:
            pad_token_id = 0

            def apply_chat_template(self, prompt, add_generation_prompt=True, tokenize=True):
                return [11, 12, 13]

            def encode(self, text, add_special_tokens=False):
                return [21, 22]

        batch = module.build_reward_batch(
            tokenizer=FakeTokenizer(),
            prompt_messages=[{"role": "user", "content": "hello"}],
            solution="<tool_call></tool_call>",
            ground_truth={"hardware": {"total_memory_gb": 80.0}},
        )

        self.assertTrue(torch.equal(batch.batch["attention_mask"], torch.tensor([[1, 1, 1, 1, 1]])))
        self.assertTrue(torch.equal(batch.batch["prompts"], torch.tensor([[11, 12, 13]])))


if __name__ == "__main__":
    unittest.main()
