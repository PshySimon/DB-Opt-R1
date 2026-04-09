from pathlib import Path
import unittest
from hydra.utils import instantiate
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]


class TrainingScriptDefaultsTest(unittest.TestCase):
    def test_verl_sft_lora_uses_multiturn_dataset_fields(self):
        content = (ROOT / "scripts" / "train_sft_verl_lora.sh").read_text()
        self.assertIn('DATA_DIR="${DATA_DIR:-./datasets/sft_cleaned}"', content)
        self.assertIn('VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"', content)
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
        self.assertIn('VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"', content)
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
        self.assertIn("torch==2.8.0+cu128", content)
        self.assertIn("vllm==0.11.0", content)
        self.assertIn("transformers>=4.55.2,<5", content)
        self.assertIn("xformers==0.0.32.post1", content)
        self.assertIn("flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE", content)
        self.assertNotIn("flashinfer-python", content)

    def test_verl_grpo_scripts_use_configurable_attention_impl(self):
        lora_content = (ROOT / "scripts" / "train_grpo_verl_lora.sh").read_text()
        self.assertIn('VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"', lora_content)
        self.assertIn('ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"', lora_content)
        self.assertNotIn("actor_rollout_ref.model.override_config.attn_implementation", lora_content)
        self.assertNotIn("actor_rollout_ref.model.torch_dtype", lora_content)
        self.assertIn("++actor_rollout_ref.model.lora_rank=$LORA_RANK", lora_content)
        self.assertIn("++actor_rollout_ref.model.lora_alpha=$LORA_ALPHA", lora_content)
        self.assertIn("++actor_rollout_ref.model.target_modules=$TARGET_MODULES", lora_content)
        self.assertIn("actor_rollout_ref.rollout.n=$N_REPEAT", lora_content)
        self.assertNotIn("actor_rollout_ref.rollout.n_repeat", lora_content)

        full_content = (ROOT / "scripts" / "train_grpo_verl_full.sh").read_text()
        self.assertIn('VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"', full_content)
        self.assertIn('ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"', full_content)
        self.assertNotIn("actor_rollout_ref.model.override_config.attn_implementation", full_content)
        self.assertNotIn("actor_rollout_ref.model.torch_dtype", full_content)

    def test_verl_grpo_main_sets_vllm_v1_runtime_env(self):
        content = (ROOT / "training" / "verl" / "main_grpo.py").read_text()
        self.assertIn("'VLLM_USE_V1': '1'", content)

    def test_grpo_trainer_config_has_verl_071_targets(self):
        content = (ROOT / "configs" / "grpo_trainer.yaml").read_text()
        self.assertIn("return_raw_chat: True", content)
        self.assertIn("_target_: verl.workers.config.HFModelConfig", content)
        self.assertIn("_target_: verl.workers.config.FSDPActorConfig", content)
        self.assertIn("_target_: verl.workers.config.FSDPCriticConfig", content)
        self.assertIn("_target_: verl.workers.config.FSDPCriticModelCfg", content)
        self.assertIn("_target_: verl.workers.config.FSDPEngineConfig", content)
        self.assertIn("_target_: verl.workers.config.FSDPOptimizerConfig", content)
        self.assertIn("_target_: verl.workers.config.RolloutConfig", content)
        self.assertIn("rollout_n: ${oc.select:actor_rollout_ref.rollout.n,1}", content)
        self.assertIn("weight_decay: 0.01", content)
        self.assertIn("optimizer: AdamW", content)
        self.assertIn("forward_only: True", content)
        self.assertIn("entropy_from_logits_with_chunking: False", content)
        self.assertIn("entropy_checkpointing: False", content)
        self.assertNotIn("use_fire_sampling", content)
        self.assertIn("checkpoint:", content)
        self.assertIn("_target_: verl.trainer.config.CheckpointConfig", content)
        self.assertIn("_target_: verl.utils.profiler.ProfilerConfig", content)
        self.assertIn("_target_: verl.workers.config.RouterReplayConfig", content)
        self.assertIn("data_parallel_size: 1", content)
        self.assertIn("expert_parallel_size: 1", content)
        self.assertIn("pipeline_model_parallel_size: 1", content)
        self.assertIn("logprobs_mode: null", content)
        self.assertIn("_target_: verl.workers.config.PrometheusConfig", content)
        self.assertIn("_target_: verl.workers.config.AgentLoopConfig", content)
        self.assertIn("_target_: verl.workers.config.CustomAsyncServerConfig", content)
        self.assertIn("_target_: verl.workers.config.TraceConfig", content)
        self.assertIn("_target_: verl.workers.config.MultiTurnConfig", content)
        self.assertIn("_target_: verl.workers.config.ServerConfig", content)
        self.assertIn("_target_: verl.workers.config.CheckpointEngineConfig", content)
        self.assertIn("calculate_log_probs: True", content)

    def test_verl_grpo_lora_script_accepts_extra_hydra_overrides(self):
        content = (ROOT / "scripts" / "train_grpo_verl_lora.sh").read_text()
        self.assertIn('"$@"', content)

    def test_agent_ray_trainer_forwards_raw_prompt_to_async_rollout(self):
        content = (ROOT / "training" / "verl" / "agent_ray_trainer.py").read_text()
        self.assertIn("non_tensor_batch_keys=['raw_prompt_ids', 'raw_prompt', 'multi_modal_data', 'multi_modal_inputs']", content)
        self.assertIn("non_tensor_batch_keys=['raw_prompt_ids', 'raw_prompt']", content)

    def test_grpo_rollout_config_instantiates_with_verl_071_schema(self):
        cfg = OmegaConf.load(ROOT / "configs" / "grpo_trainer.yaml")
        rollout = instantiate(cfg.actor_rollout_ref.rollout, _convert_="partial")
        self.assertEqual(rollout.name, "vllm")
        self.assertEqual(rollout.data_parallel_size, 1)
        self.assertTrue(rollout.calculate_log_probs)
        self.assertEqual(cfg.actor_rollout_ref.actor.checkpoint._target_, "verl.trainer.config.CheckpointConfig")
        self.assertEqual(cfg.actor_rollout_ref.ref.profiler._target_, "verl.utils.profiler.ProfilerConfig")


if __name__ == "__main__":
    unittest.main()
