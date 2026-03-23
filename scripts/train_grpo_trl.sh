#!/bin/bash
# trl GRPO 训练
set -e

MODEL_PATH="${MODEL_PATH:-./model_save/sft/}"
SCENARIO_DIR="${SCENARIO_DIR:-./datasets/data/scenarios/}"
COST_MODEL="${COST_MODEL:-./cost_model/checkpoints/v1}"
OUTPUT_DIR="${OUTPUT_DIR:-./model_save/grpo/}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-1e-6}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LORA_RANK="${LORA_RANK:-64}"

echo "============================================"
echo "  GRPO 训练 (trl)"
echo "============================================"
echo "模型:       $MODEL_PATH"
echo "场景数据:   $SCENARIO_DIR"
echo "Cost Model: $COST_MODEL"
echo "输出:       $OUTPUT_DIR"
echo "============================================"

python -m training.trl.grpo \
    --model_path "$MODEL_PATH" \
    --scenario_dir "$SCENARIO_DIR" \
    --cost_model "$COST_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --lora_rank $LORA_RANK
