from pathlib import Path
import json
import subprocess
import tempfile
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
        self.assertIn('MAX_STEPS="${MAX_STEPS:--1}"', full_content)
        self.assertIn('LOGGING_STEPS="${LOGGING_STEPS:-5}"', full_content)
        self.assertIn('SAVE_STEPS="${SAVE_STEPS:-50}"', full_content)
        self.assertIn('SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"', full_content)
        self.assertIn('SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"', full_content)
        self.assertIn('EVAL_STEPS="${EVAL_STEPS:-50}"', full_content)
        self.assertIn('EVAL_STRATEGY="${EVAL_STRATEGY:-steps}"', full_content)
        self.assertIn('DISABLE_EVAL="${DISABLE_EVAL:-false}"', full_content)
        self.assertIn('configure_accelerator_visible_devices', full_content)
        self.assertIn('TRAIN_CONFIG_JSON="${TRAIN_CONFIG_JSON:-$OUTPUT_DIR/train_config.json}"', full_content)
        self.assertIn('DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"', full_content)
        self.assertIn('FSDP="${FSDP:-}"', full_content)
        self.assertIn('FSDP_CONFIG="${FSDP_CONFIG:-}"', full_content)
        self.assertIn('TOKENIZED_DATASET_DIR="${TOKENIZED_DATASET_DIR:-}"', full_content)
        self.assertIn('RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"', full_content)
        self.assertIn('TORCHRUN_PORT="${TORCHRUN_PORT:-${MASTER_PORT:-}}"', full_content)
        self.assertIn('TORCHRUN_RUN_ID="${TORCHRUN_RUN_ID:-}"', full_content)
        self.assertIn('TORCHRUN_PORT="$(infer_torchrun_port "$TORCHRUN_PORT")"', full_content)
        self.assertIn('TORCHRUN_RUN_ID="$(infer_torchrun_run_id "$TORCHRUN_RUN_ID")"', full_content)
        self.assertIn("--rdzv-endpoint=\"$MASTER_ADDR:$TORCHRUN_PORT\"", full_content)
        self.assertIn("--rdzv-id=\"$TORCHRUN_RUN_ID\"", full_content)
        self.assertIn('--save_config_path "$TRAIN_CONFIG_JSON"', full_content)
        self.assertIn('--train_ratio $SFT_TRAIN_RATIO', full_content)
        self.assertIn('--max_steps $MAX_STEPS', full_content)
        self.assertIn('--logging_steps $LOGGING_STEPS', full_content)
        self.assertIn('--save_steps $SAVE_STEPS', full_content)
        self.assertIn('--save_total_limit $SAVE_TOTAL_LIMIT', full_content)
        self.assertIn('--save_strategy "$SAVE_STRATEGY"', full_content)
        self.assertIn('--eval_steps $EVAL_STEPS', full_content)
        self.assertIn('--eval_strategy "$EVAL_STRATEGY"', full_content)
        self.assertIn('cmd+=(--disable_eval)', full_content)
        self.assertIn('--deepspeed "$DEEPSPEED_CONFIG"', full_content)
        self.assertIn('--fsdp "$FSDP"', full_content)
        self.assertIn('--fsdp_config "$FSDP_CONFIG"', full_content)
        self.assertIn('--tokenized_dataset_dir "$TOKENIZED_DATASET_DIR"', full_content)
        self.assertIn('--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT"', full_content)

        common_content = (ROOT / "scripts" / "_train_common.sh").read_text()
        self.assertIn("infer_torchrun_port()", common_content)
        self.assertIn("infer_torchrun_run_id()", common_content)
        self.assertIn("infer_logical_visible_devices()", common_content)
        self.assertIn("configure_accelerator_visible_devices()", common_content)

        trl_entry = (ROOT / "training" / "trl" / "sft.py").read_text()
        self.assertIn('parser.add_argument("--save_config_path"', trl_entry)
        self.assertIn('parser.add_argument("--train_ratio"', trl_entry)
        self.assertIn('"assistant_only_loss": not bool(getattr(args, "tokenized_dataset_dir", None))', trl_entry)
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
        self.assertIn("peft==0.18.1", content)

    def test_requirements_llama_factory_pins_separate_stack(self):
        content = (ROOT / "requirements-llama-factory.txt").read_text()
        self.assertIn("llamafactory[deepspeed,metrics]==0.9.3", content)
        self.assertIn("transformers>=4.51.0,<=4.52.4", content)
        self.assertIn("datasets>=2.16.0,<=3.6.0", content)

    def test_sm120_vllm_flashinfer_setup_script_pins_cuda13_stack(self):
        content = (ROOT / "scripts" / "setup_sm120_vllm_flashinfer_env.sh").read_text()
        self.assertIn('ENV_PREFIX="${ENV_PREFIX:-/root/autodl-tmp/conda_envs/dbopt-vllm-flashinfer-cu130}"', content)
        self.assertIn('PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/DB-Opt-R1}"', content)
        self.assertIn('BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/Qwen3-8B}"', content)
        self.assertIn('FLASHINFER_VERSION="${FLASHINFER_VERSION:-0.6.9}"', content)
        self.assertIn('FLASHINFER_CUDA="${FLASHINFER_CUDA:-cu130}"', content)
        self.assertIn('VLLM_VERSION="${VLLM_VERSION:-0.20.0}"', content)
        self.assertIn('VLLM_INDEX_URL="${VLLM_INDEX_URL:-https://wheels.vllm.ai/cu130}"', content)
        self.assertIn('"vllm==$VLLM_VERSION"', content)
        self.assertIn('--index-url "$VLLM_INDEX_URL"', content)
        self.assertIn('"flashinfer-python==$FLASHINFER_VERSION"', content)
        self.assertIn('"flashinfer-cubin==$FLASHINFER_VERSION"', content)
        self.assertIn('"flashinfer-jit-cache==${FLASHINFER_VERSION}+${FLASHINFER_CUDA}"', content)
        self.assertIn('export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.0f}"', content)
        self.assertIn('export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"', content)
        self.assertIn('AttentionBackendEnum.FLASHINFER', content)
        self.assertIn("Using AttentionBackendEnum.FLASHINFER backend", content)
        self.assertNotIn("uv pip", content)
        self.assertNotIn("VLLM_ATTENTION_BACKEND", content)

    def test_sm120_vllm_flashinfer_setup_script_is_safe_by_default(self):
        content = (ROOT / "scripts" / "setup_sm120_vllm_flashinfer_env.sh").read_text()
        self.assertIn('RECREATE="${RECREATE:-false}"', content)
        self.assertIn('if [ -d "$ENV_PREFIX" ] && [ "$RECREATE" = "true" ]; then', content)
        self.assertIn('conda env remove -p "$ENV_PREFIX" -y', content)
        self.assertIn('RUN_SMOKE="${RUN_SMOKE:-false}"', content)
        self.assertIn('SMOKE_ONLY="${SMOKE_ONLY:-false}"', content)
        self.assertIn('SMOKE_ONLY=true requires an existing env', content)
        self.assertIn('SMOKE_ONLY=true, skipping package installation', content)
        self.assertIn('cat > "$SMOKE_SCRIPT" <<PY', content)
        self.assertIn('python "$SMOKE_SCRIPT"', content)

    def test_sm120_llamafactory_setup_script_pins_training_stack(self):
        content = (ROOT / "scripts" / "setup_sm120_llamafactory_env.sh").read_text()
        self.assertIn('ENV_PREFIX="${ENV_PREFIX:-/root/autodl-tmp/conda_envs/dbopt-lf-sm120}"', content)
        self.assertIn('PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/DB-Opt-R1}"', content)
        self.assertIn('BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/Qwen3-8B}"', content)
        self.assertIn('PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"', content)
        self.assertIn('TORCH_VERSION="${TORCH_VERSION:-2.8.0}"', content)
        self.assertIn('TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-4.52.4}"', content)
        self.assertIn('DATASETS_VERSION="${DATASETS_VERSION:-3.6.0}"', content)
        self.assertIn('ACCELERATE_VERSION="${ACCELERATE_VERSION:-1.7.0}"', content)
        self.assertIn('DEEPSPEED_VERSION="${DEEPSPEED_VERSION:-0.16.9}"', content)
        self.assertIn('LLAMAFACTORY_VERSION="${LLAMAFACTORY_VERSION:-0.9.3}"', content)
        self.assertIn('FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"', content)
        self.assertIn('flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl', content)
        self.assertIn('"torch==$TORCH_VERSION" torchvision torchaudio', content)
        self.assertIn('"llamafactory[deepspeed,metrics]==$LLAMAFACTORY_VERSION"', content)
        self.assertIn('python -m pip install --no-cache-dir "$FLASH_ATTN_WHEEL_URL"', content)
        self.assertIn('Using FlashAttention-2 for faster training and inference', content)
        self.assertIn('Fine-tuning method: LoRA', content)
        self.assertNotIn("uv pip", content)
        self.assertNotIn("flashinfer-python", content)
        self.assertNotIn("vllm==", content)

    def test_sm120_llamafactory_setup_script_is_safe_by_default(self):
        content = (ROOT / "scripts" / "setup_sm120_llamafactory_env.sh").read_text()
        self.assertIn('RECREATE="${RECREATE:-false}"', content)
        self.assertIn('RUN_SMOKE="${RUN_SMOKE:-false}"', content)
        self.assertIn('SMOKE_ONLY="${SMOKE_ONLY:-false}"', content)
        self.assertIn('SKIP_GPU_CHECK="${SKIP_GPU_CHECK:-false}"', content)
        self.assertIn('conda env remove -p "$ENV_PREFIX" -y', content)
        self.assertIn('SMOKE_ONLY=true requires an existing env', content)
        self.assertIn('SMOKE_ONLY=true, skipping package installation', content)
        self.assertIn('DRY_RUN=true', content)
        self.assertIn('llamafactory-cli train "$SMOKE_OUTPUT_DIR/llamafactory_train.yaml"', content)

    def test_llamafactory_sft_full_script_uses_v3_step_data(self):
        content = (ROOT / "scripts" / "train_sft_llamafactory_full.sh").read_text()
        self.assertIn('DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data_pipeline/data/train/v3/full_v3_b_step_no_think_history_3k}"', content)
        self.assertIn('OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/model_save/experiments/v3/sft/llamafactory/full/full_v3_b_step_no_think_history_3k}"', content)
        self.assertIn('TEMPLATE="${TEMPLATE:-qwen3}"', content)
        self.assertIn('MASK_HISTORY="${MASK_HISTORY:-true}"', content)
        self.assertIn('ENABLE_THINKING="${ENABLE_THINKING:-true}"', content)
        self.assertIn('FLASH_ATTN="${FLASH_ATTN:-fa2}"', content)
        self.assertIn('LR="${LR:-0.000001}"', content)
        self.assertIn('DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$PROJECT_ROOT/configs/deepspeed_zero2_bf16.json}"', content)
        self.assertIn('DATASET_FILE="$LF_DATASET_DIR/${DATASET_NAME}.jsonl"', content)
        self.assertIn('DATASET_INFO="$LF_DATASET_DIR/dataset_info.json"', content)
        self.assertIn('"formatting": "alpaca"', content)
        self.assertIn('"history": "history"', content)
        self.assertIn("mask_history: $(yaml_bool \"$MASK_HISTORY\")", content)
        self.assertIn("enable_thinking: $(yaml_bool \"$ENABLE_THINKING\")", content)
        self.assertIn("flash_attn: $FLASH_ATTN", content)
        self.assertIn('llamafactory-cli train "$TRAIN_YAML"', content)
        self.assertIn('N_GPUS="${N_GPUS:-1}"', content)
        self.assertIn('configure_accelerator_visible_devices', content)
        self.assertNotIn('--nproc_per_node="$N_GPUS"', content)
        self.assertNotIn('exec torchrun', content)
        self.assertIn('DRY_RUN="${DRY_RUN:-false}"', content)
        self.assertIn('DRY_RUN=true，仅生成 LLaMA-Factory 数据和配置，不启动训练。', content)

    def test_llamafactory_sft_full_dry_run_writes_eval_dataset_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_dir = tmp_path / "train"
            val_dir = tmp_path / "val"
            output_dir = tmp_path / "out"
            train_dir.mkdir()
            val_dir.mkdir()
            sample = {"instruction": "inst", "input": "", "output": "out", "system": "", "history": []}
            (train_dir / "train.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
            (val_dir / "validation.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")

            subprocess.run(
                [
                    "bash",
                    "-lc",
                    (
                        "DRY_RUN=true "
                        f"PROJECT_ROOT='{ROOT}' "
                        f"DATA_DIR='{train_dir}' "
                        f"VAL_DATA_DIR='{val_dir}' "
                        "DATASET_NAME=db_train "
                        "VAL_DATASET_NAME=db_val "
                        f"OUTPUT_DIR='{output_dir}' "
                        "RESUME_FROM_CHECKPOINT=/tmp/checkpoint-50 "
                        "EVAL_STEPS=25 "
                        "LOAD_BEST_MODEL_AT_END=true "
                        "scripts/train_sft_llamafactory_full.sh"
                    ),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            )

            yaml_text = (output_dir / "llamafactory_train.yaml").read_text(encoding="utf-8")
            dataset_info = json.loads(
                (output_dir / "llamafactory_dataset" / "dataset_info.json").read_text(encoding="utf-8")
            )
            self.assertIn("eval_dataset: db_val", yaml_text)
            self.assertIn("eval_strategy: steps", yaml_text)
            self.assertIn("eval_steps: 25", yaml_text)
            self.assertIn("load_best_model_at_end: true", yaml_text)
            self.assertIn("metric_for_best_model: eval_loss", yaml_text)
            self.assertIn("greater_is_better: false", yaml_text)
            self.assertIn("resume_from_checkpoint: /tmp/checkpoint-50", yaml_text)
            self.assertEqual("db_train.jsonl", dataset_info["db_train"]["file_name"])
            self.assertEqual("db_val.jsonl", dataset_info["db_val"]["file_name"])

    def test_llamafactory_sft_full_dry_run_writes_val_size_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_dir = tmp_path / "train"
            output_dir = tmp_path / "out"
            train_dir.mkdir()
            sample = {"instruction": "inst", "input": "", "output": "out", "system": "", "history": []}
            (train_dir / "train.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")

            subprocess.run(
                [
                    "bash",
                    "-lc",
                    (
                        "DRY_RUN=true "
                        f"PROJECT_ROOT='{ROOT}' "
                        f"DATA_DIR='{train_dir}' "
                        f"OUTPUT_DIR='{output_dir}' "
                        "VAL_SIZE=0.05 "
                        "scripts/train_sft_llamafactory_full.sh"
                    ),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            )

            yaml_text = (output_dir / "llamafactory_train.yaml").read_text(encoding="utf-8")
            self.assertIn("val_size: 0.05", yaml_text)
            self.assertIn("eval_strategy: steps", yaml_text)
            self.assertIn("load_best_model_at_end: true", yaml_text)
            self.assertNotIn("eval_dataset:", yaml_text)

    def test_deepspeed_zero2_bf16_config_is_checked_in(self):
        payload = OmegaConf.load(ROOT / "configs" / "deepspeed_zero2_bf16.json")
        self.assertEqual(2, payload.zero_optimization.stage)
        self.assertTrue(payload.bf16.enabled)
        self.assertFalse(payload.fp16.enabled)
        self.assertEqual(500000000, payload.zero_optimization.allgather_bucket_size)
        self.assertFalse(hasattr(payload, "scheduler"))

    def test_deepspeed_zero3_bf16_llamafactory_config_is_checked_in(self):
        payload = OmegaConf.load(ROOT / "configs" / "deepspeed_zero3_bf16_llamafactory.json")
        self.assertEqual(3, payload.zero_optimization.stage)
        self.assertTrue(payload.bf16.enabled)
        self.assertFalse(payload.fp16.enabled)
        self.assertEqual(500000000, payload.zero_optimization.reduce_bucket_size)
        self.assertEqual(500000000, payload.zero_optimization.stage3_prefetch_bucket_size)
        self.assertTrue(payload.zero_optimization.stage3_gather_16bit_weights_on_model_save)
        self.assertFalse(hasattr(payload, "scheduler"))

    def test_verl_grpo_scripts_use_configurable_attention_impl(self):
        lora_content = (ROOT / "scripts" / "train_grpo_verl_lora.sh").read_text()
        self.assertIn('VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"', lora_content)
        self.assertIn('COST_MODEL_PATH="${COST_MODEL_PATH:-./cost_model/checkpoints/v10_lgbm}"', lora_content)
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
        self.assertIn('COST_MODEL_PATH="${COST_MODEL_PATH:-./cost_model/checkpoints/v10_lgbm}"', full_content)
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

    def test_grpo_trl_scripts_default_to_v10_cost_model(self):
        lora_content = (ROOT / "scripts" / "train_grpo_trl_lora.sh").read_text()
        self.assertIn('COST_MODEL="${COST_MODEL:-./cost_model/checkpoints/v10_lgbm}"', lora_content)

        full_content = (ROOT / "scripts" / "train_grpo_trl_full.sh").read_text()
        self.assertIn('COST_MODEL="${COST_MODEL:-./cost_model/checkpoints/v10_lgbm}"', full_content)

        bench_content = (ROOT / "scripts" / "bench_grpo_trl_lora_16.sh").read_text()
        self.assertIn('COST_MODEL="${COST_MODEL:-./cost_model/checkpoints/v10_lgbm}"', bench_content)

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
