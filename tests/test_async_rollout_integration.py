from pathlib import Path
import io
import unittest
from contextlib import redirect_stdout

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


class _CapturingTokenizer(_FakeTokenizer):
    def __init__(self):
        self.tokenized_texts = []

    def __call__(self, texts, add_special_tokens=False, return_tensors=None, padding=None):
        self.tokenized_texts.extend(texts)
        return super().__call__(texts, add_special_tokens, return_tensors, padding)


class _BudgetTokenizer(_FakeTokenizer):
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        rendered = "\n".join(f"{message['role']}:{message['content']}" for message in messages)
        if add_generation_prompt:
            rendered += "\nassistant:"
        if tokenize:
            return [1] * len(rendered)
        return rendered


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

    def test_tool_generation_manager_logs_rollout_timing_profile(self):
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

        with unittest.mock.patch.object(
            manager,
            "_postprocess_responses",
            return_value=(
                torch.tensor([[11, 12]], dtype=torch.long),
                ["call-a"],
                torch.tensor([False]),
            ),
        ), unittest.mock.patch.object(
            manager,
            "_execute_tool_calls",
            return_value=(["tool-a"], [False]),
        ), unittest.mock.patch.object(
            manager,
            "_process_tool_responses",
            return_value=torch.tensor([[31, 32]], dtype=torch.long),
        ):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                manager.run_llm_loop(
                    gen_batch=batch,
                    envs=[_FakeEnv()],
                    initial_input_ids=batch.batch["input_ids"],
                )

        logs = stdout.getvalue()
        self.assertIn("rollout_turn_profile", logs)
        self.assertIn("postprocess_s=", logs)
        self.assertIn("tool_tokenize_s=", logs)
        self.assertIn("other_s=", logs)
        self.assertIn("rollout_profile_total", logs)

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

    def test_tool_generation_manager_strips_think_from_next_turn_history(self):
        tokenizer = _CapturingTokenizer()
        manager = ToolGenerationManager(
            tokenizer=tokenizer,
            sequence_generator=_FakeSequenceGenerator(),
            config=ToolGenerationConfig(
                max_turns=1,
                max_start_length=8,
                max_prompt_length=8,
                max_response_length=8,
                max_tool_response_length=64,
                num_gpus=1,
                strip_think_history=True,
            ),
        )

        manager._responses_for_next_turn(
            [
                '<think>observe hardware</think>\n<tool_call>{"name":"get_hardware_info","arguments":{}}</tool_call><eos>'
            ]
        )

        self.assertEqual(
            ['<tool_call>{"name":"get_hardware_info","arguments":{}}</tool_call><eos>'],
            tokenizer.tokenized_texts,
        )

    def test_tool_generation_manager_updates_raw_prompt_for_async_next_turn(self):
        tokenizer = _FakeTokenizer()
        manager = ToolGenerationManager(
            tokenizer=tokenizer,
            sequence_generator=_FakeSequenceGenerator(),
            config=ToolGenerationConfig(
                max_turns=1,
                max_start_length=8,
                max_prompt_length=8,
                max_response_length=8,
                max_tool_response_length=64,
                num_gpus=1,
                strip_think_history=True,
            ),
        )
        batch = DataProto.from_dict(
            tensors={
                "input_ids": torch.ones((1, 4), dtype=torch.long),
                "attention_mask": torch.ones((1, 4), dtype=torch.long),
                "position_ids": torch.arange(4).repeat(1, 1),
            },
            non_tensors={
                "raw_prompt": [[
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "slow writes"},
                ]],
            },
        )

        manager._update_raw_prompts_for_next_turn(
            batch,
            ['<think>observe</think>\n<tool_call>{"name":"get_hardware_info","arguments":{}}</tool_call><eos>'],
            ['{"cpu_count": 8}'],
            torch.tensor([True]),
        )

        messages = batch.non_tensor_batch["raw_prompt"][0]
        self.assertEqual(
            {"role": "assistant", "content": '<tool_call>{"name":"get_hardware_info","arguments":{}}</tool_call>'},
            messages[-2],
        )
        self.assertEqual(
            {"role": "user", "content": '<tool_response>\n{"cpu_count": 8}\n</tool_response>'},
            messages[-1],
        )

    def test_tool_generation_manager_limits_raw_prompt_history_like_sft(self):
        tokenizer = _FakeTokenizer()
        manager = ToolGenerationManager(
            tokenizer=tokenizer,
            sequence_generator=_FakeSequenceGenerator(),
            config=ToolGenerationConfig(
                max_turns=1,
                max_start_length=8,
                max_prompt_length=8,
                max_response_length=8,
                max_tool_response_length=64,
                num_gpus=1,
                raw_prompt_history_turns=4,
                strip_think_history=True,
            ),
        )
        history = [{"role": "system", "content": "system"}]
        for i in range(5):
            history.extend(
                [
                    {"role": "user", "content": f"u{i}"},
                    {"role": "assistant", "content": f"a{i}"},
                ]
            )
        history.append({"role": "user", "content": "current"})
        batch = DataProto.from_dict(
            tensors={
                "input_ids": torch.ones((1, 4), dtype=torch.long),
                "attention_mask": torch.ones((1, 4), dtype=torch.long),
                "position_ids": torch.arange(4).repeat(1, 1),
            },
            non_tensors={"raw_prompt": [history]},
        )

        manager._update_raw_prompts_for_next_turn(
            batch,
            ['<tool_call>{"name":"get_current_config","arguments":{}}</tool_call><eos>'],
            ['{"shared_buffers": "2GB"}'],
            torch.tensor([True]),
        )

        messages = batch.non_tensor_batch["raw_prompt"][0]
        self.assertEqual("system", messages[0]["content"])
        self.assertEqual(
            ["u2", "a2", "u3", "a3", "u4", "a4", "current"],
            [message["content"] for message in messages[1:8]],
        )
        self.assertEqual(
            '<tool_call>{"name":"get_current_config","arguments":{}}</tool_call>',
            messages[-2]["content"],
        )
        self.assertEqual('<tool_response>\n{"shared_buffers": "2GB"}\n</tool_response>', messages[-1]["content"])

    def test_tool_generation_manager_trims_raw_prompt_history_to_token_budget(self):
        tokenizer = _BudgetTokenizer()
        manager = ToolGenerationManager(
            tokenizer=tokenizer,
            sequence_generator=_FakeSequenceGenerator(),
            config=ToolGenerationConfig(
                max_turns=1,
                max_start_length=90,
                max_prompt_length=90,
                max_response_length=8,
                max_tool_response_length=8,
                num_gpus=1,
                raw_prompt_history_turns=4,
                strip_think_history=True,
            ),
        )
        messages = [{"role": "system", "content": "system"}]
        for i in range(4):
            messages.extend(
                [
                    {"role": "user", "content": f"old-user-{i}" * 4},
                    {"role": "assistant", "content": f"old-assistant-{i}" * 4},
                ]
            )
        messages.append({"role": "user", "content": "current"})

        trimmed = manager._trim_raw_prompt_history(messages)

        rendered_tokens = tokenizer.apply_chat_template(trimmed, add_generation_prompt=True, tokenize=True)
        self.assertLessEqual(len(rendered_tokens), 90)
        self.assertEqual("system", trimmed[0]["content"])
        self.assertEqual("current", trimmed[-1]["content"])
        self.assertNotIn("old-user-0" * 4, [message["content"] for message in trimmed])

    def test_tool_generation_manager_truncates_raw_tool_response_history(self):
        tokenizer = _FakeTokenizer()
        manager = ToolGenerationManager(
            tokenizer=tokenizer,
            sequence_generator=_FakeSequenceGenerator(),
            config=ToolGenerationConfig(
                max_turns=1,
                max_start_length=64,
                max_prompt_length=64,
                max_response_length=8,
                max_tool_response_length=8,
                num_gpus=1,
                strip_think_history=True,
            ),
        )

        content = manager._tool_response_message_content("0123456789abcdef")

        self.assertEqual("<tool_response>\n01234567\n</tool_response>", content)


if __name__ == "__main__":
    unittest.main()
