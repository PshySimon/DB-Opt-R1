#!/bin/bash
# trl SFT 训练（全量）
set -e

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
DATA_FILES="${DATA_FILES:-datasets/sft/cold_start.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./model_save/sft_full/}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
FLASH_ATTN="${FLASH_ATTN:-false}"

echo "============================================"
echo "  SFT 训练 (trl, 全量)"
echo "============================================"
echo "模型:     $BASE_MODEL"
echo "数据:     $DATA_FILES"
echo "输出:     $OUTPUT_DIR"
echo "Epochs:   $EPOCHS"
echo "============================================"

python -m training.trl.sft \
    --model_path "$BASE_MODEL" \
    --data_files $DATA_FILES \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --max_length $MAX_LENGTH \
    --full_finetune \
    $( [ "$GRADIENT_CHECKPOINTING" = "false" ] && echo "--no_gradient_checkpointing" ) \
    $( [ "$FLASH_ATTN" = "true" ] && echo "--flash_attn" )
