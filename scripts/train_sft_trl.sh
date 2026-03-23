#!/bin/bash
# trl SFT 训练
set -e

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
DATA_FILES="${DATA_FILES:-datasets/sft/cold_start.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./model_save/sft/}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LORA_RANK="${LORA_RANK:-64}"
MAX_LENGTH="${MAX_LENGTH:-4096}"

echo "============================================"
echo "  SFT 训练 (trl)"
echo "============================================"
echo "模型:     $BASE_MODEL"
echo "数据:     $DATA_FILES"
echo "输出:     $OUTPUT_DIR"
echo "Epochs:   $EPOCHS"
echo "LoRA r:   $LORA_RANK"
echo "============================================"

python -m training.trl.sft \
    --model_path "$BASE_MODEL" \
    --data_files $DATA_FILES \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --lora_rank $LORA_RANK \
    --max_length $MAX_LENGTH
