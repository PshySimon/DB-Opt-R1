import unittest
import sys
import types
from types import SimpleNamespace


trl_module = types.ModuleType("trl")
trl_module.GRPOConfig = type("GRPOConfig", (), {})
trl_module.GRPOTrainer = type("GRPOTrainer", (), {})
trl_module.SFTConfig = type("SFTConfig", (), {})
trl_module.SFTTrainer = type("SFTTrainer", (), {})
sys.modules.setdefault("trl", trl_module)

trl_trainer_module = types.ModuleType("trl.trainer")
sys.modules.setdefault("trl.trainer", trl_trainer_module)

trl_trainer_utils_module = types.ModuleType("trl.trainer.utils")
trl_trainer_utils_module.pad = lambda *args, **kwargs: None
trl_trainer_utils_module.selective_log_softmax = lambda *args, **kwargs: None
sys.modules.setdefault("trl.trainer.utils", trl_trainer_utils_module)

trl_data_utils_module = types.ModuleType("trl.data_utils")
trl_data_utils_module.maybe_apply_chat_template = lambda *args, **kwargs: None
trl_data_utils_module.is_conversational = lambda *args, **kwargs: False
sys.modules.setdefault("trl.data_utils", trl_data_utils_module)

from training.trl import grpo


class FakeCompletionsAPI:
    def __init__(self, responses):
        self._responses = list(responses)

    def create(self, **kwargs):
        if not self._responses:
            raise AssertionError("No fake response configured")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeOpenAIClient:
    def __init__(self, responses):
        self.completions = FakeCompletionsAPI(responses)


class GRPOServerModeTests(unittest.TestCase):
    def test_parser_uses_vllm_server_args(self):
        parser = grpo.build_parser()
        args = parser.parse_args([
            "--model_path", "model",
            "--train_data", "train.jsonl",
            "--scenario_files", "a.json",
        ])

        self.assertEqual(args.vllm_server_host, "127.0.0.1")
        self.assertEqual(args.vllm_server_port, 8000)
        self.assertEqual(args.vllm_model_name, "qwen3-4b-sft")
        self.assertEqual(args.attn_impl, "sdpa")
        self.assertEqual(args.rollout_log_interval, 1)

    def test_server_backend_returns_batch_texts_in_index_order(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(index=1, text="second"),
                SimpleNamespace(index=0, text="first"),
            ]
        )
        backend = grpo.VLLMServerBackend(
            model_name="qwen3-4b-sft",
            base_url="http://127.0.0.1:8000/v1",
            client=FakeOpenAIClient([response]),
        )

        outputs = backend.generate_texts(
            ["prompt-1", "prompt-2"],
            temperature=1.0,
            top_p=1.0,
            max_tokens=32,
            stop=["</tool_call>"],
        )

        self.assertEqual(outputs, ["first", "second"])

    def test_format_rollout_turn_log_includes_lengths_and_elapsed(self):
        line = grpo.format_rollout_turn_log(
            rollout_id=3,
            turn_idx=2,
            active_count=4,
            batch_size=4,
            prompt_token_lengths=[128, 256, 512, 1024],
            elapsed_s=3.14159,
        )

        self.assertIn("rollout#3", line)
        self.assertIn("turn 3", line)
        self.assertIn("active=4/4", line)
        self.assertIn("prompt_tokens(avg=480", line)
        self.assertIn("max=1024", line)
        self.assertIn("elapsed=3.14s", line)


if __name__ == "__main__":
    unittest.main()
