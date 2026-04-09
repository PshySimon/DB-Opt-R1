import sys
import types
import unittest
from unittest import mock
import subprocess
from pathlib import Path

import numpy as np
import torch
from verl import DataProto
from omegaconf import OmegaConf

from training.verl import main_grpo

ROOT = Path(__file__).resolve().parents[1]


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

    def test_verify_grpo_reward_path_script_reports_positive_answer_score(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "verify_grpo_reward_path.py")],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("db_reward_manager_answer_score=", result.stdout)
        self.assertIn("status=PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
