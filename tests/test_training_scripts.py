from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TrainingScriptDefaultsTest(unittest.TestCase):
    def test_verl_sft_lora_uses_multiturn_dataset_fields(self):
        content = (ROOT / "scripts" / "train_sft_verl_lora.sh").read_text()
        self.assertIn('DATA_DIR="${DATA_DIR:-./datasets/sft_cleaned}"', content)
        self.assertIn("model.path=$BASE_MODEL", content)
        self.assertIn("model.lora_rank=$LORA_RANK", content)
        self.assertIn("model.lora_alpha=$LORA_ALPHA", content)
        self.assertIn("model.target_modules=$TARGET_MODULES", content)
        self.assertNotIn("prompt_key", content)
        self.assertNotIn("response_key", content)
        self.assertNotIn("prompt_dict_keys", content)
        self.assertNotIn("response_dict_keys", content)
        self.assertNotIn("partial_pretrain", content)
        self.assertNotIn("torch_dtype", content)
        self.assertNotIn("attn_implementation", content)

    def test_verl_sft_full_uses_multiturn_dataset_fields(self):
        content = (ROOT / "scripts" / "train_sft_verl_full.sh").read_text()
        self.assertIn('DATA_DIR="${DATA_DIR:-./datasets/sft_cleaned}"', content)
        self.assertIn("model.path=$BASE_MODEL", content)
        self.assertNotIn("prompt_key", content)
        self.assertNotIn("response_key", content)
        self.assertNotIn("prompt_dict_keys", content)
        self.assertNotIn("response_dict_keys", content)
        self.assertNotIn("partial_pretrain", content)
        self.assertNotIn("torch_dtype", content)
        self.assertNotIn("attn_implementation", content)


if __name__ == "__main__":
    unittest.main()
