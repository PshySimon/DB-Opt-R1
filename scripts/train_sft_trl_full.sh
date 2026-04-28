#!/bin/bash
# trl SFT 训练（全量）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_train_common.sh"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
DATA_FILES="${DATA_FILES:-datasets/sft/cold_start.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./model_save/sft_full/}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
SFT_TRAIN_RATIO="${SFT_TRAIN_RATIO:-0.95}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
FLASH_ATTN="${FLASH_ATTN:-false}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
N_GPUS="${N_GPUS:-1}"
TRAIN_CONFIG_JSON="${TRAIN_CONFIG_JSON:-$OUTPUT_DIR/train_config.json}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"
FSDP="${FSDP:-}"
FSDP_CONFIG="${FSDP_CONFIG:-}"
TOKENIZED_DATASET_DIR="${TOKENIZED_DATASET_DIR:-}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
TORCHRUN_PORT="${TORCHRUN_PORT:-${MASTER_PORT:-}}"
TORCHRUN_RUN_ID="${TORCHRUN_RUN_ID:-}"
REQUESTED_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
REQUESTED_HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-}"
REQUESTED_ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-}"

mkdir -p "$OUTPUT_DIR"
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

write_train_config_json "$TRAIN_CONFIG_JSON" \
    BASE_MODEL DATA_FILES OUTPUT_DIR EPOCHS LR BATCH_SIZE GRAD_ACCUM \
    MAX_LENGTH SFT_TRAIN_RATIO GRADIENT_CHECKPOINTING FLASH_ATTN \
    DEEPSPEED_CONFIG FSDP FSDP_CONFIG TOKENIZED_DATASET_DIR \
    CUDA_DEVICES CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES \
    N_GPUS MASTER_ADDR MASTER_PORT TORCHRUN_PORT TORCHRUN_RUN_ID TRAIN_CONFIG_JSON

echo "============================================"
echo "  SFT 训练 (trl, 全量)"
echo "============================================"
echo "模型:     $BASE_MODEL"
echo "数据:     $DATA_FILES"
echo "输出:     $OUTPUT_DIR"
echo "Epochs:   $EPOCHS"
echo "GPU 数量: $N_GPUS"
if [ -n "$DEEPSPEED_CONFIG" ]; then
    echo "DeepSpeed: $DEEPSPEED_CONFIG"
fi
if [ -n "$FSDP" ]; then
    echo "FSDP:      $FSDP"
fi
if [ -n "$TOKENIZED_DATASET_DIR" ]; then
    echo "Tokenized: $TOKENIZED_DATASET_DIR"
fi
echo "Rdzv:     $MASTER_ADDR:$TORCHRUN_PORT ($TORCHRUN_RUN_ID)"
echo "配置:     $TRAIN_CONFIG_JSON"
echo "============================================"

cmd=(
    python -m training.trl.sft
    --model_path "$BASE_MODEL"
    --data_files $DATA_FILES
    --output_dir "$OUTPUT_DIR"
    --save_config_path "$TRAIN_CONFIG_JSON"
    --num_epochs $EPOCHS
    --lr $LR
    --batch_size $BATCH_SIZE
    --grad_accum $GRAD_ACCUM
    --max_length $MAX_LENGTH
    --train_ratio $SFT_TRAIN_RATIO
    --full_finetune
)

if [ "$GRADIENT_CHECKPOINTING" = "false" ]; then
    cmd+=(--no_gradient_checkpointing)
fi
if [ "$FLASH_ATTN" = "true" ]; then
    cmd+=(--flash_attn)
fi
if [ -n "$DEEPSPEED_CONFIG" ]; then
    cmd+=(--deepspeed "$DEEPSPEED_CONFIG")
fi
if [ -n "$FSDP" ]; then
    cmd+=(--fsdp "$FSDP")
fi
if [ -n "$FSDP_CONFIG" ]; then
    cmd+=(--fsdp_config "$FSDP_CONFIG")
fi
if [ -n "$TOKENIZED_DATASET_DIR" ]; then
    cmd+=(--tokenized_dataset_dir "$TOKENIZED_DATASET_DIR")
fi

if [ "$N_GPUS" -gt 1 ]; then
    exec torchrun \
        --nnodes=1 \
        --nproc_per_node=$N_GPUS \
        --rdzv-backend=c10d \
        --rdzv-endpoint="$MASTER_ADDR:$TORCHRUN_PORT" \
        --rdzv-id="$TORCHRUN_RUN_ID" \
        "${cmd[@]:1}"
else
    exec "${cmd[@]}"
fi
