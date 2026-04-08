import sys
import types
import unittest


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

from training.trl import grpo, sft


class TrainingModeTests(unittest.TestCase):
    def test_grpo_defaults_to_lora(self):
        parser = grpo.build_parser()
        args = parser.parse_args([
            "--model_path", "model",
            "--train_data", "train.jsonl",
            "--scenario_files", "a.json",
        ])

        self.assertTrue(args.use_lora)
        self.assertFalse(args.full_finetune)

    def test_grpo_can_switch_to_full_finetune(self):
        parser = grpo.build_parser()
        args = parser.parse_args([
            "--model_path", "model",
            "--train_data", "train.jsonl",
            "--scenario_files", "a.json",
            "--full_finetune",
        ])

        self.assertFalse(args.use_lora)
        self.assertTrue(args.full_finetune)

    def test_sft_defaults_to_lora(self):
        parser = sft.build_parser()
        args = parser.parse_args([
            "--data_files", "train.jsonl",
        ])

        self.assertTrue(args.use_lora)
        self.assertFalse(args.full_finetune)

    def test_sft_can_switch_to_full_finetune(self):
        parser = sft.build_parser()
        args = parser.parse_args([
            "--data_files", "train.jsonl",
            "--full_finetune",
        ])

        self.assertFalse(args.use_lora)
        self.assertTrue(args.full_finetune)


if __name__ == "__main__":
    unittest.main()
