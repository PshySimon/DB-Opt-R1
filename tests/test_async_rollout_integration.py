from pathlib import Path
import unittest

import torch

from core.tool.tool_base import Tool
from training.llm_agent.generation import ToolGenerationConfig, ToolGenerationManager
from training.tool.tool_env import ToolEnv
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
        self.termination_reason = None
        self.last_valid_tool_call = None
        self.last_valid_config = None

    def step(self, action_text):
        return "", 0.0, False, {}


class _FinishTool(Tool):
    def __init__(self):
        super().__init__(
            name="finish_tuning",
            description="finish",
            parameters={"type": "object", "properties": {}, "required": []},
        )

    def execute(self, args):
        return "done"


class _DoneEnv(ToolEnv):
    def __init__(self):
        super().__init__(tools=[_FinishTool()], max_turns=3)
        self._config_snapshot = None

    def get_current_config_snapshot(self):
        return self._config_snapshot


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

        with unittest.mock.patch.object(manager, "_execute_tool_calls", return_value=(["", ""], [False, False])):
            manager.run_llm_loop(
                gen_batch=batch,
                envs=[_FakeEnv(), _FakeEnv()],
                initial_input_ids=batch.batch["input_ids"],
            )

        self.assertIsNotNone(sequence_generator.last_non_tensor_batch)
        self.assertIn("raw_prompt", sequence_generator.last_non_tensor_batch)

    def test_tool_generation_manager_returns_response_mask_for_agent_loop(self):
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
        )

        with unittest.mock.patch.object(
            manager,
            "_postprocess_responses",
            return_value=(
                torch.tensor([[11, 12], [21, 22]], dtype=torch.long),
                ["call-a", "call-b"],
                torch.tensor([False, False]),
            ),
        ), unittest.mock.patch.object(
            manager,
            "_execute_tool_calls",
            return_value=(["tool-a", "tool-b"], [False, False]),
        ), unittest.mock.patch.object(
            manager,
            "_process_tool_responses",
            return_value=torch.tensor([[31, 32, 33], [41, 42, 43]], dtype=torch.long),
        ):
            output = manager.run_llm_loop(
                gen_batch=batch,
                envs=[_FakeEnv(), _FakeEnv()],
                initial_input_ids=batch.batch["input_ids"],
            )

        self.assertIn("response_mask", output.batch.keys())
        self.assertEqual(
            output.batch["response_mask"].tolist(),
            [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
        )

    def test_tool_generation_manager_stops_when_env_marks_done(self):
        tokenizer = _FakeTokenizer()
        sequence_generator = _FakeSequenceGenerator()
        manager = ToolGenerationManager(
            tokenizer=tokenizer,
            sequence_generator=sequence_generator,
            config=ToolGenerationConfig(
                max_turns=3,
                max_start_length=8,
                max_prompt_length=8,
                max_response_length=8,
                max_tool_response_length=8,
                num_gpus=1,
            ),
        )
        batch = DataProto.from_dict(
            tensors={
                "input_ids": torch.ones((1, 4), dtype=torch.long),
                "attention_mask": torch.ones((1, 4), dtype=torch.long),
                "position_ids": torch.arange(4).repeat(1, 1),
            },
        )
        env = _DoneEnv()

        with unittest.mock.patch.object(
            manager,
            "_postprocess_responses",
            return_value=(
                torch.tensor([[11, 12]], dtype=torch.long),
                ['<tool_call>{"name":"finish_tuning","arguments":{}}</tool_call>'],
                torch.tensor([True]),
            ),
        ):
            output = manager.run_llm_loop(
                gen_batch=batch,
                envs=[env],
                initial_input_ids=batch.batch["input_ids"],
            )

        self.assertEqual(1, env.steps_taken)
        self.assertIn("termination_reason", output.non_tensor_batch)
        self.assertEqual("finish_tuning", output.non_tensor_batch["termination_reason"][0])

    def test_tool_generation_manager_returns_last_valid_state_metadata(self):
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
                "input_ids": torch.ones((1, 4), dtype=torch.long),
                "attention_mask": torch.ones((1, 4), dtype=torch.long),
                "position_ids": torch.arange(4).repeat(1, 1),
            },
        )
        env = _DoneEnv()
        env.last_valid_tool_call = {"tool": "finish_tuning", "args": {}}
        env._config_snapshot = {"shared_buffers": "8GB"}

        with unittest.mock.patch.object(
            manager,
            "_postprocess_responses",
            return_value=(
                torch.tensor([[11, 12]], dtype=torch.long),
                ['<tool_call>{"name":"finish_tuning","arguments":{}}</tool_call>'],
                torch.tensor([True]),
            ),
        ):
            output = manager.run_llm_loop(
                gen_batch=batch,
                envs=[env],
                initial_input_ids=batch.batch["input_ids"],
            )

        self.assertEqual(
            {"tool": "finish_tuning", "args": {}},
            output.non_tensor_batch["last_valid_tool_call"][0],
        )
        self.assertEqual(
            {"shared_buffers": "8GB"},
            output.non_tensor_batch["last_valid_config"][0],
        )


if __name__ == "__main__":
    unittest.main()
