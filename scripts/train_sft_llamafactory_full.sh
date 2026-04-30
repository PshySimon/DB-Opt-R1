#!/bin/bash
# LLaMA-Factory SFT training (full fine-tuning).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_train_common.sh"

BASE_MODEL="${BASE_MODEL:-/root/workspace/models/Qwen3-8B}"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data_pipeline/data/train/v3/full_v3_b_step_no_think_history_3k}"
DATASET_NAME="${DATASET_NAME:-db_opt_v3}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/model_save/experiments/v3/sft/llamafactory/full/full_v3_b_step_no_think_history_3k}"
LF_DATASET_DIR="${LF_DATASET_DIR:-$OUTPUT_DIR/llamafactory_dataset}"
TRAIN_YAML="${TRAIN_YAML:-$OUTPUT_DIR/llamafactory_train.yaml}"
TRAIN_CONFIG_JSON="${TRAIN_CONFIG_JSON:-$OUTPUT_DIR/train_config.json}"

EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:--1}"
LR="${LR:-0.000001}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-8}"
TEMPLATE="${TEMPLATE:-qwen3}"
MASK_HISTORY="${MASK_HISTORY:-true}"
ENABLE_THINKING="${ENABLE_THINKING:-true}"
PACKING="${PACKING:-false}"
FLASH_ATTN="${FLASH_ATTN:-fa2}"
BF16="${BF16:-true}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$PROJECT_ROOT/configs/deepspeed_zero2_bf16.json}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
VAL_DATA_DIR="${VAL_DATA_DIR:-}"
VAL_DATASET_NAME="${VAL_DATASET_NAME:-${DATASET_NAME}_val}"
VAL_SIZE="${VAL_SIZE:-}"
EVAL_STRATEGY="${EVAL_STRATEGY:-steps}"
EVAL_STEPS="${EVAL_STEPS:-50}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-$PER_DEVICE_BATCH_SIZE}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
LOAD_BEST_MODEL_AT_END="${LOAD_BEST_MODEL_AT_END:-true}"
METRIC_FOR_BEST_MODEL="${METRIC_FOR_BEST_MODEL:-eval_loss}"
GREATER_IS_BETTER="${GREATER_IS_BETTER:-false}"
REPORT_TO="${REPORT_TO:-none}"
PLOT_LOSS="${PLOT_LOSS:-true}"
OVERWRITE_CACHE="${OVERWRITE_CACHE:-true}"
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-false}"
DRY_RUN="${DRY_RUN:-false}"

CUDA_DEVICES="${CUDA_DEVICES:-0}"
N_GPUS="${N_GPUS:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
TORCHRUN_PORT="${TORCHRUN_PORT:-${MASTER_PORT:-}}"
TORCHRUN_RUN_ID="${TORCHRUN_RUN_ID:-}"
REQUESTED_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
REQUESTED_HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-}"
REQUESTED_ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-}"

mkdir -p "$OUTPUT_DIR" "$LF_DATASET_DIR"
if [ -n "$VAL_DATA_DIR" ] && [ -n "$VAL_SIZE" ]; then
    echo "VAL_DATA_DIR 和 VAL_SIZE 不能同时设置：请使用独立验证集，或从训练集切分验证集。" >&2
    exit 2
fi

N_GPUS="$(infer_n_gpus "$CUDA_DEVICES" "$N_GPUS")"
configure_accelerator_visible_devices \
    "$CUDA_DEVICES" \
    "$N_GPUS" \
    "$REQUESTED_CUDA_VISIBLE_DEVICES" \
    "$REQUESTED_HIP_VISIBLE_DEVICES" \
    "$REQUESTED_ROCR_VISIBLE_DEVICES"
TORCHRUN_PORT="$(infer_torchrun_port "$TORCHRUN_PORT")"
TORCHRUN_RUN_ID="$(infer_torchrun_run_id "$TORCHRUN_RUN_ID")"
export MASTER_ADDR TORCHRUN_PORT TORCHRUN_RUN_ID
export MASTER_PORT="$TORCHRUN_PORT"

DATASET_FILE="$LF_DATASET_DIR/${DATASET_NAME}.jsonl"
VAL_DATASET_FILE=""
if [ -n "$VAL_DATA_DIR" ]; then
    VAL_DATASET_FILE="$LF_DATASET_DIR/${VAL_DATASET_NAME}.jsonl"
fi
DATASET_INFO="$LF_DATASET_DIR/dataset_info.json"

python - "$DATA_DIR" "$DATASET_FILE" <<'PY'
import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
shards = sorted(src.glob("train-*-of-*.jsonl"))
if not shards:
    single = src / "train.jsonl"
    if single.exists():
        shards = [single]
if not shards:
    raise SystemExit(f"No train JSONL shards found under {src}")

dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("wb") as out:
    for shard in shards:
        with shard.open("rb") as f:
            shutil.copyfileobj(f, out)
PY

if [ -n "$VAL_DATA_DIR" ]; then
    python - "$VAL_DATA_DIR" "$VAL_DATASET_FILE" <<'PY'
import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
patterns = [
    "validation-*-of-*.jsonl",
    "val-*-of-*.jsonl",
    "eval-*-of-*.jsonl",
    "train-*-of-*.jsonl",
    "validation.jsonl",
    "val.jsonl",
    "eval.jsonl",
    "train.jsonl",
    "*.jsonl",
]
shards = []
for pattern in patterns:
    shards = sorted(src.glob(pattern))
    if shards:
        break
if not shards:
    raise SystemExit(f"No validation JSONL shards found under {src}")

dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("wb") as out:
    for shard in shards:
        with shard.open("rb") as f:
            shutil.copyfileobj(f, out)
PY
fi

env DATASET_NAME="$DATASET_NAME" DATASET_FILE="$DATASET_FILE" \
    VAL_DATASET_NAME="$VAL_DATASET_NAME" VAL_DATASET_FILE="$VAL_DATASET_FILE" \
    python - "$DATASET_INFO" <<'PY'
import json
import os
import sys
from pathlib import Path

output = Path(sys.argv[1])
columns = {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "system": "system",
    "history": "history",
}

payload = {
    os.environ["DATASET_NAME"]: {
        "file_name": Path(os.environ["DATASET_FILE"]).name,
        "formatting": "alpaca",
        "columns": columns,
    }
}
val_file = os.environ.get("VAL_DATASET_FILE")
if val_file:
    payload[os.environ["VAL_DATASET_NAME"]] = {
        "file_name": Path(val_file).name,
        "formatting": "alpaca",
        "columns": columns,
    }
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

write_train_config_json "$TRAIN_CONFIG_JSON" \
    BASE_MODEL PROJECT_ROOT DATA_DIR DATASET_NAME OUTPUT_DIR LF_DATASET_DIR DATASET_FILE \
    DATASET_INFO TRAIN_YAML EPOCHS MAX_STEPS LR MAX_LENGTH PER_DEVICE_BATCH_SIZE \
    GRAD_ACCUM WARMUP_RATIO LR_SCHEDULER_TYPE WEIGHT_DECAY LOGGING_STEPS SAVE_STEPS \
    SAVE_TOTAL_LIMIT PREPROCESSING_NUM_WORKERS TEMPLATE MASK_HISTORY ENABLE_THINKING \
    PACKING FLASH_ATTN BF16 GRADIENT_CHECKPOINTING DEEPSPEED_CONFIG RESUME_FROM_CHECKPOINT \
    VAL_DATA_DIR VAL_DATASET_NAME VAL_DATASET_FILE VAL_SIZE EVAL_STRATEGY EVAL_STEPS \
    PER_DEVICE_EVAL_BATCH_SIZE SAVE_STRATEGY LOAD_BEST_MODEL_AT_END METRIC_FOR_BEST_MODEL \
    GREATER_IS_BETTER REPORT_TO PLOT_LOSS OVERWRITE_CACHE OVERWRITE_OUTPUT_DIR DRY_RUN \
    CUDA_DEVICES CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES N_GPUS \
    MASTER_ADDR MASTER_PORT TORCHRUN_PORT TORCHRUN_RUN_ID TRAIN_CONFIG_JSON

yaml_bool() {
    case "$1" in
        true|True|TRUE|1|yes|YES) echo "true" ;;
        *) echo "false" ;;
    esac
}

{
    echo "stage: sft"
    echo "do_train: true"
    echo "model_name_or_path: $BASE_MODEL"
    echo "dataset: $DATASET_NAME"
    echo "dataset_dir: $LF_DATASET_DIR"
    echo "template: $TEMPLATE"
    echo "finetuning_type: full"
    echo "output_dir: $OUTPUT_DIR"
    echo "overwrite_cache: $(yaml_bool "$OVERWRITE_CACHE")"
    echo "overwrite_output_dir: $(yaml_bool "$OVERWRITE_OUTPUT_DIR")"
    echo "cutoff_len: $MAX_LENGTH"
    echo "preprocessing_num_workers: $PREPROCESSING_NUM_WORKERS"
    echo "mask_history: $(yaml_bool "$MASK_HISTORY")"
    echo "enable_thinking: $(yaml_bool "$ENABLE_THINKING")"
    echo "packing: $(yaml_bool "$PACKING")"
    echo "per_device_train_batch_size: $PER_DEVICE_BATCH_SIZE"
    echo "gradient_accumulation_steps: $GRAD_ACCUM"
    echo "learning_rate: $LR"
    echo "num_train_epochs: $EPOCHS"
    echo "max_steps: $MAX_STEPS"
    echo "lr_scheduler_type: $LR_SCHEDULER_TYPE"
    echo "warmup_ratio: $WARMUP_RATIO"
    echo "weight_decay: $WEIGHT_DECAY"
    echo "bf16: $(yaml_bool "$BF16")"
    echo "gradient_checkpointing: $(yaml_bool "$GRADIENT_CHECKPOINTING")"
    echo "flash_attn: $FLASH_ATTN"
    echo "logging_steps: $LOGGING_STEPS"
    echo "save_strategy: $SAVE_STRATEGY"
    echo "save_steps: $SAVE_STEPS"
    echo "save_total_limit: $SAVE_TOTAL_LIMIT"
    if [ -n "$VAL_DATA_DIR" ] || [ -n "$VAL_SIZE" ]; then
        echo "eval_strategy: $EVAL_STRATEGY"
        echo "eval_steps: $EVAL_STEPS"
        echo "per_device_eval_batch_size: $PER_DEVICE_EVAL_BATCH_SIZE"
        if [ -n "$VAL_DATA_DIR" ]; then
            echo "eval_dataset: $VAL_DATASET_NAME"
        fi
        if [ -n "$VAL_SIZE" ]; then
            echo "val_size: $VAL_SIZE"
        fi
        echo "load_best_model_at_end: $(yaml_bool "$LOAD_BEST_MODEL_AT_END")"
        echo "metric_for_best_model: $METRIC_FOR_BEST_MODEL"
        echo "greater_is_better: $(yaml_bool "$GREATER_IS_BETTER")"
    fi
    echo "report_to: $REPORT_TO"
    echo "plot_loss: $(yaml_bool "$PLOT_LOSS")"
    echo "ddp_timeout: 180000000"
    if [ -n "$DEEPSPEED_CONFIG" ]; then
        echo "deepspeed: $DEEPSPEED_CONFIG"
    fi
    if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
        echo "resume_from_checkpoint: $RESUME_FROM_CHECKPOINT"
    fi
} > "$TRAIN_YAML"

echo "============================================================"
echo "SFT 训练配置 (LLaMA-Factory, 全量)"
echo "============================================================"
echo "基座模型:     $BASE_MODEL"
echo "源数据目录:   $DATA_DIR"
echo "LF 数据目录:  $LF_DATASET_DIR"
echo "Dataset:      $DATASET_NAME"
echo "输出目录:     $OUTPUT_DIR"
echo "max_length:   $MAX_LENGTH"
echo "lr:           $LR"
echo "epochs:       $EPOCHS"
echo "max_steps:    $MAX_STEPS"
echo "batch/gacc:   $PER_DEVICE_BATCH_SIZE / $GRAD_ACCUM"
echo "GPU 数量:     $N_GPUS"
echo "template:     $TEMPLATE"
echo "mask_history: $MASK_HISTORY"
echo "thinking:     $ENABLE_THINKING"
echo "flash_attn:   $FLASH_ATTN"
if [ -n "$DEEPSPEED_CONFIG" ]; then
    echo "DeepSpeed:    $DEEPSPEED_CONFIG"
fi
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    echo "Resume:       $RESUME_FROM_CHECKPOINT"
fi
if [ -n "$VAL_DATA_DIR" ]; then
    echo "Val 数据目录: $VAL_DATA_DIR"
    echo "Val Dataset:  $VAL_DATASET_NAME"
elif [ -n "$VAL_SIZE" ]; then
    echo "Val split:    $VAL_SIZE"
fi
echo "YAML:         $TRAIN_YAML"
echo "配置:         $TRAIN_CONFIG_JSON"
echo "Rdzv:         $MASTER_ADDR:$TORCHRUN_PORT ($TORCHRUN_RUN_ID)"
echo "============================================================"

if [ "$DRY_RUN" = "true" ]; then
    echo "DRY_RUN=true，仅生成 LLaMA-Factory 数据和配置，不启动训练。"
    exit 0
fi

exec llamafactory-cli train "$TRAIN_YAML"
