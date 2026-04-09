from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TrainingScriptDefaultsTest(unittest.TestCase):
    def test_verl_sft_lora_uses_multiturn_dataset_fields(self):
        content = (ROOT / "scripts" / "train_sft_verl_lora.sh").read_text()
        self.assertIn('DATA_DIR="${DATA_DIR:-./datasets/sft_cleaned}"', content)
        self.assertIn('ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"', content)
        self.assertIn("model.path=$BASE_MODEL", content)
        self.assertNotIn("model.override_config.attn_implementation", content)
        self.assertIn("data.ignore_input_ids_mismatch=True", content)
        self.assertIn("model.lora_rank=$LORA_RANK", content)
        self.assertIn("model.lora_alpha=$LORA_ALPHA", content)
        self.assertIn("model.target_modules=$TARGET_MODULES", content)
        self.assertNotIn("prompt_key", content)
        self.assertNotIn("response_key", content)
        self.assertNotIn("prompt_dict_keys", content)
        self.assertNotIn("response_dict_keys", content)
        self.assertNotIn("partial_pretrain", content)
        self.assertNotIn("torch_dtype", content)

    def test_verl_sft_full_uses_multiturn_dataset_fields(self):
        content = (ROOT / "scripts" / "train_sft_verl_full.sh").read_text()
        self.assertIn('DATA_DIR="${DATA_DIR:-./datasets/sft_cleaned}"', content)
        self.assertIn('ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"', content)
        self.assertIn("model.path=$BASE_MODEL", content)
        self.assertNotIn("model.override_config.attn_implementation", content)
        self.assertIn("data.ignore_input_ids_mismatch=True", content)
        self.assertNotIn("prompt_key", content)
        self.assertNotIn("response_key", content)
        self.assertNotIn("prompt_dict_keys", content)
        self.assertNotIn("response_dict_keys", content)
        self.assertNotIn("partial_pretrain", content)
        self.assertNotIn("torch_dtype", content)

    def test_requirements_verl_includes_flash_attn(self):
        content = (ROOT / "requirements-verl.txt").read_text()
        self.assertIn("verl==0.7.1", content)
        self.assertIn("torch==2.6.0+cu124", content)
        self.assertIn("vllm==0.8.5.post1", content)
        self.assertIn("flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE", content)
        self.assertIn("https://flashinfer.ai/whl/cu124/torch2.6/", content)
        self.assertIn("flashinfer-python==0.2.5", content)

    def test_verl_grpo_scripts_use_configurable_attention_impl(self):
        for name in ["train_grpo_verl_lora.sh", "train_grpo_verl_full.sh"]:
            content = (ROOT / "scripts" / name).read_text()
            self.assertIn('ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"', content)
            self.assertNotIn("actor_rollout_ref.model.override_config.attn_implementation", content)
            self.assertNotIn("actor_rollout_ref.model.torch_dtype", content)


if __name__ == "__main__":
    unittest.main()
