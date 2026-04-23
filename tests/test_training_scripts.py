from pathlib import Path
import subprocess
import unittest
from hydra.utils import instantiate
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]


class TrainingScriptDefaultsTest(unittest.TestCase):
    def test_configure_accelerator_visible_devices_remaps_physical_ids(self):
        script = """
source scripts/_train_common.sh
configure_accelerator_visible_devices "4,5,6,7" "4" "4,5,6,7" "4,5,6,7" ""
printf '%s\\n' "$CUDA_VISIBLE_DEVICES" "$HIP_VISIBLE_DEVICES" "$ROCR_VISIBLE_DEVICES"
"""
        result = subprocess.run(
            ["bash", "-lc", script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        lines = result.stdout.strip().splitlines()
        self.assertEqual(lines, ["0,1,2,3", "0,1,2,3", "4,5,6,7"])

    def test_trl_sft_scripts_support_multigpu_and_write_config_json(self):
        lora_content = (ROOT / "scripts" / "train_sft_trl_lora.sh").read_text()
        self.assertIn('N_GPUS="${N_GPUS:-1}"', lora_content)
        self.assertIn('CUDA_DEVICES="${CUDA_DEVICES:-0}"', lora_content)
        self.assertIn('MAX_LENGTH="${MAX_LENGTH:-8192}"', lora_content)
        self.assertIn('SFT_TRAIN_RATIO="${SFT_TRAIN_RATIO:-0.95}"', lora_content)
        self.assertIn('configure_accelerator_visible_devices', lora_content)
        self.assertIn('TRAIN_CONFIG_JSON="${TRAIN_CONFIG_JSON:-$OUTPUT_DIR/train_config.json}"', lora_content)
        self.assertIn('TORCHRUN_PORT="${TORCHRUN_PORT:-${MASTER_PORT:-}}"', lora_content)
        self.assertIn('TORCHRUN_RUN_ID="${TORCHRUN_RUN_ID:-}"', lora_content)
        self.assertIn('TORCHRUN_PORT="$(infer_torchrun_port "$TORCHRUN_PORT")"', lora_content)
        self.assertIn('TORCHRUN_RUN_ID="$(infer_torchrun_run_id "$TORCHRUN_RUN_ID")"', lora_content)
        self.assertIn("--rdzv-endpoint=\"$MASTER_ADDR:$TORCHRUN_PORT\"", lora_content)
        self.assertIn("--rdzv-id=\"$TORCHRUN_RUN_ID\"", lora_content)
        self.assertIn('--save_config_path "$TRAIN_CONFIG_JSON"', lora_content)
        self.assertIn('--train_ratio $SFT_TRAIN_RATIO', lora_content)

        full_content = (ROOT / "scripts" / "train_sft_trl_full.sh").read_text()
        self.assertIn('N_GPUS="${N_GPUS:-1}"', full_content)
        self.assertIn('CUDA_DEVICES="${CUDA_DEVICES:-0}"', full_content)
        self.assertIn('MAX_LENGTH="${MAX_LENGTH:-8192}"', full_content)
        self.assertIn('SFT_TRAIN_RATIO="${SFT_TRAIN_RATIO:-0.95}"', full_content)
        self.assertIn('configure_accelerator_visible_devices', full_content)
        self.assertIn('TRAIN_CONFIG_JSON="${TRAIN_CONFIG_JSON:-$OUTPUT_DIR/train_config.json}"', full_content)
        self.assertIn('TORCHRUN_PORT="${TORCHRUN_PORT:-${MASTER_PORT:-}}"', full_content)
        self.assertIn('TORCHRUN_RUN_ID="${TORCHRUN_RUN_ID:-}"', full_content)
        self.assertIn('TORCHRUN_PORT="$(infer_torchrun_port "$TORCHRUN_PORT")"', full_content)
        self.assertIn('TORCHRUN_RUN_ID="$(infer_torchrun_run_id "$TORCHRUN_RUN_ID")"', full_content)
        self.assertIn("--rdzv-endpoint=\"$MASTER_ADDR:$TORCHRUN_PORT\"", full_content)
        self.assertIn("--rdzv-id=\"$TORCHRUN_RUN_ID\"", full_content)
        self.assertIn('--save_config_path "$TRAIN_CONFIG_JSON"', full_content)
        self.assertIn('--train_ratio $SFT_TRAIN_RATIO', full_content)

        common_content = (ROOT / "scripts" / "_train_common.sh").read_text()
        self.assertIn("infer_torchrun_port()", common_content)
        self.assertIn("infer_torchrun_run_id()", common_content)
        self.assertIn("infer_logical_visible_devices()", common_content)
        self.assertIn("configure_accelerator_visible_devices()", common_content)

        trl_entry = (ROOT / "training" / "trl" / "sft.py").read_text()
        self.assertIn('parser.add_argument("--save_config_path"', trl_entry)
        self.assertIn('parser.add_argument("--train_ratio"', trl_entry)
        self.assertIn('"assistant_only_loss": True', trl_entry)
        self.assertIn('eval_dataset=eval_dataset', trl_entry)
        self.assertIn("save_training_config(", trl_entry)

    def test_verl_training_scripts_write_config_json(self):
        sft_lora = (ROOT / "scripts" / "train_sft_verl_lora.sh").read_text()
        self.assertIn('configure_accelerator_visible_devices', sft_lora)
        self.assertIn('TRAIN_CONFIG_JSON="${TRAIN_CONFIG_JSON:-$SFT_OUTPUT_DIR/train_config.json}"', sft_lora)
        self.assertIn("write_train_config_json", sft_lora)

        sft_full = (ROOT / "scripts" / "train_sft_verl_full.sh").read_text()
        self.assertIn('configure_accelerator_visible_devices', sft_full)
        self.assertIn('TRAIN_CONFIG_JSON="${TRAIN_CONFIG_JSON:-$SFT_OUTPUT_DIR/train_config.json}"', sft_full)
        self.assertIn("write_train_config_json", sft_full)

        grpo_lora = (ROOT / "scripts" / "train_grpo_verl_lora.sh").read_text()
        self.assertIn('configure_accelerator_visible_devices', grpo_lora)
        self.assertIn('TRAIN_CONFIG_JSON="${TRAIN_CONFIG_JSON:-$OUTPUT_DIR/train_config.json}"', grpo_lora)
        self.assertIn("write_train_config_json", grpo_lora)
        self.assertIn("trainer.default_local_dir=$OUTPUT_DIR", grpo_lora)

        grpo_full = (ROOT / "scripts" / "train_grpo_verl_full.sh").read_text()
        self.assertIn('configure_accelerator_visible_devices', grpo_full)
        self.assertIn('TRAIN_CONFIG_JSON="${TRAIN_CONFIG_JSON:-$OUTPUT_DIR/train_config.json}"', grpo_full)
        self.assertIn("write_train_config_json", grpo_full)
        self.assertIn("trainer.default_local_dir=$OUTPUT_DIR", grpo_full)

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
        self.assertIn("tokenizers>=0.21.1", content)
        self.assertIn("pyarrow", content)
        self.assertIn("xformers==0.0.32.post1", content)
        self.assertIn("flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE", content)
        self.assertNotIn("flashinfer-python", content)

    def test_requirements_trl_pins_supported_trl_range(self):
        content = (ROOT / "requirements-trl.txt").read_text()
        self.assertIn("trl==1.2.0", content)
        self.assertIn("transformers==5.5.4", content)

    def test_verl_grpo_scripts_use_configurable_attention_impl(self):
        lora_content = (ROOT / "scripts" / "train_grpo_verl_lora.sh").read_text()
        self.assertIn('VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"', lora_content)
        self.assertIn('COST_MODEL_PATH="${COST_MODEL_PATH:-./cost_model/checkpoints/v9_lgbm}"', lora_content)
        self.assertIn('ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"', lora_content)
        self.assertIn('MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"', lora_content)
        self.assertIn('GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"', lora_content)
        self.assertIn('SCENARIO_FILES="${SCENARIO_FILES:-./data_pipeline/data/scenarios/collected/collected_server1.json,./data_pipeline/data/scenarios/collected/collected_server2.json,./data_pipeline/data/scenarios/collected/collected_server3.json}"', lora_content)
        self.assertIn('SCENARIO_SOURCE_FILTER="${SCENARIO_SOURCE_FILTER:-llm_generated}"', lora_content)
        self.assertNotIn("actor_rollout_ref.model.override_config.attn_implementation", lora_content)
        self.assertNotIn("actor_rollout_ref.model.torch_dtype", lora_content)
        self.assertIn("++actor_rollout_ref.model.lora_rank=$LORA_RANK", lora_content)
        self.assertIn("++actor_rollout_ref.model.lora_alpha=$LORA_ALPHA", lora_content)
        self.assertIn("++actor_rollout_ref.model.target_modules=$TARGET_MODULES", lora_content)
        self.assertIn('echo "Cost Model:  $COST_MODEL_PATH"', lora_content)
        self.assertIn("actor_rollout_ref.rollout.n=$N_REPEAT", lora_content)
        self.assertIn("+scenario_dir=[$SCENARIO_FILES]", lora_content)
        self.assertIn("+scenario_source_filter=$SCENARIO_SOURCE_FILTER", lora_content)
        self.assertIn('DEBUG_ROLLOUT_DIR="${DEBUG_ROLLOUT_DIR:-debug/rollout}"', lora_content)
        self.assertIn('EARLY_STOPPING_ENABLED="${EARLY_STOPPING_ENABLED:-False}"', lora_content)
        self.assertIn("debug_rollout_dir=$DEBUG_ROLLOUT_DIR", lora_content)
        self.assertIn("trainer.early_stopping.enabled=$EARLY_STOPPING_ENABLED", lora_content)
        self.assertNotIn("actor_rollout_ref.rollout.n_repeat", lora_content)

        full_content = (ROOT / "scripts" / "train_grpo_verl_full.sh").read_text()
        self.assertIn('VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"', full_content)
        self.assertIn('COST_MODEL_PATH="${COST_MODEL_PATH:-./cost_model/checkpoints/v9_lgbm}"', full_content)
        self.assertIn('ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"', full_content)
        self.assertIn('MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"', full_content)
        self.assertIn('GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"', full_content)
        self.assertIn('SCENARIO_FILES="${SCENARIO_FILES:-./data_pipeline/data/scenarios/collected/collected_server1.json,./data_pipeline/data/scenarios/collected/collected_server2.json,./data_pipeline/data/scenarios/collected/collected_server3.json}"', full_content)
        self.assertIn('SCENARIO_SOURCE_FILTER="${SCENARIO_SOURCE_FILTER:-llm_generated}"', full_content)
        self.assertNotIn("actor_rollout_ref.model.override_config.attn_implementation", full_content)
        self.assertNotIn("actor_rollout_ref.model.torch_dtype", full_content)
        self.assertIn("actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_LOG_PROB_MICRO_BATCH_SIZE", full_content)
        self.assertIn("actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE", full_content)
        self.assertIn('echo "Cost Model:  $COST_MODEL_PATH"', full_content)
        self.assertIn("actor_rollout_ref.rollout.n=$N_REPEAT", full_content)
        self.assertIn("+scenario_dir=[$SCENARIO_FILES]", full_content)
        self.assertIn("+scenario_source_filter=$SCENARIO_SOURCE_FILTER", full_content)
        self.assertIn('DEBUG_ROLLOUT_DIR="${DEBUG_ROLLOUT_DIR:-debug/rollout}"', full_content)
        self.assertIn('EARLY_STOPPING_ENABLED="${EARLY_STOPPING_ENABLED:-False}"', full_content)
        self.assertIn("debug_rollout_dir=$DEBUG_ROLLOUT_DIR", full_content)
        self.assertIn("trainer.early_stopping.enabled=$EARLY_STOPPING_ENABLED", full_content)
        self.assertNotIn("actor_rollout_ref.rollout.n_repeat", full_content)

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
        self.assertIn("early_stopping:", content)

    def test_verl_grpo_lora_script_accepts_extra_hydra_overrides(self):
        content = (ROOT / "scripts" / "train_grpo_verl_lora.sh").read_text()
        self.assertIn('"$@"', content)

    def test_verl_grpo_full_script_accepts_extra_hydra_overrides(self):
        content = (ROOT / "scripts" / "train_grpo_verl_full.sh").read_text()
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
