from pathlib import Path
import unittest

import torch

from training.llm_agent.generation import ToolGenerationConfig, ToolGenerationManager
from verl import DataProto


ROOT = Path(__file__).resolve().parents[1]


class _FakeTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"

    def __call__(self, texts, add_special_tokens=False, return_tensors=None, padding=None):
        max_len = max(len(t) for t in texts) if texts else 0
        rows = []
        for text in texts:
            row = [1] * len(text)
            row += [0] * (max_len - len(row))
            rows.append(row)
        return {"input_ids": torch.tensor(rows, dtype=torch.long)}

    def batch_decode(self, responses, skip_special_tokens=True):
        return ["<tool_call>{}</tool_call>".format("{}")] * responses.shape[0]


class _FakeSequenceGenerator:
    def __init__(self):
        self.calls = 0
        self.last_non_tensor_batch = None

    def generate_sequences(self, batch):
        self.calls += 1
        self.last_non_tensor_batch = batch.non_tensor_batch
        return DataProto.from_dict(
            tensors={
                "responses": torch.ones((batch.batch["input_ids"].shape[0], 2), dtype=torch.long),
            },
            meta_info={},
        )


class _FakeEnv:
    def __init__(self):
        self.steps_taken = 0

    def step(self, action_text):
        return "", 0.0, False, {}


class AsyncRolloutIntegrationTest(unittest.TestCase):
    def test_tool_generation_manager_uses_injected_sequence_generator(self):
        tokenizer = _FakeTokenizer()
        sequence_generator = _FakeSequenceGenerator()
        manager = ToolGenerationManager(
            tokenizer=tokenizer,
            sequence_generator=sequence_generator,
            config=ToolGenerationConfig(
                max_turns=1,
                max_start_length=8,
                max_prompt_length=8,
                max_response_length=8,
                max_tool_response_length=8,
                num_gpus=1,
            ),
        )
        batch = DataProto.from_dict(
            tensors={
                "input_ids": torch.ones((2, 4), dtype=torch.long),
                "attention_mask": torch.ones((2, 4), dtype=torch.long),
                "position_ids": torch.arange(4).repeat(2, 1),
            }
        )

        output = manager._generate_with_gpu_padding(batch)

        self.assertEqual(sequence_generator.calls, 1)
        self.assertEqual(output.batch["responses"].shape, (2, 2))

    def test_agent_ray_trainer_uses_async_rollout_manager(self):
        content = (ROOT / "training" / "verl" / "agent_ray_trainer.py").read_text()
        self.assertIn("self.async_rollout_manager = AgentLoopManager.create(", content)
        self.assertIn("generation_manager = ToolGenerationManager(", content)
        self.assertIn("sequence_generator=self.async_rollout_manager", content)
        self.assertIn("self.checkpoint_manager.update_weights(self.global_steps)", content)
        self.assertIn("self.checkpoint_manager.sleep_replicas()", content)
        self.assertNotIn("actor_rollout_wg=self.actor_rollout_wg", content)

    def test_tool_generation_manager_preserves_raw_prompt_for_async_rollout(self):
        tokenizer = _FakeTokenizer()
        sequence_generator = _FakeSequenceGenerator()
        manager = ToolGenerationManager(
            tokenizer=tokenizer,
            sequence_generator=sequence_generator,
            config=ToolGenerationConfig(
                max_turns=1,
                max_start_length=8,
                max_prompt_length=8,
                max_response_length=8,
                max_tool_response_length=8,
                num_gpus=1,
            ),
        )
        batch = DataProto.from_dict(
            tensors={
                "input_ids": torch.ones((2, 4), dtype=torch.long),
                "attention_mask": torch.ones((2, 4), dtype=torch.long),
                "position_ids": torch.arange(4).repeat(2, 1),
            },
            non_tensors={
                "raw_prompt": [["hello"], ["world"]],
                "raw_prompt_ids": [[1, 2], [3, 4]],
            },
        )

        with unittest.mock.patch.object(manager, "_execute_tool_calls", return_value=["", ""]):
            manager.run_llm_loop(
                gen_batch=batch,
                envs=[_FakeEnv(), _FakeEnv()],
                initial_input_ids=batch.batch["input_ids"],
            )

        self.assertIsNotNone(sequence_generator.last_non_tensor_batch)
        self.assertIn("raw_prompt", sequence_generator.last_non_tensor_batch)


if __name__ == "__main__":
    unittest.main()
