#!/bin/bash
# trl GRPO 多轮工具调用训练
set -e

MODEL_PATH="${MODEL_PATH:-./model_save/sft/}"
TRAIN_DATA="${TRAIN_DATA:-./data_pipeline/data/train/sft_trajectories_v2.jsonl}"
SCENARIO_FILES="${SCENARIO_FILES:-data_pipeline/data/scenarios/collected/collected_server1.json data_pipeline/data/scenarios/collected/collected_server2.json data_pipeline/data/scenarios/collected/collected_server3.json}"
COST_MODEL="${COST_MODEL:-./cost_model/checkpoints/v9_lgbm}"
OUTPUT_DIR="${OUTPUT_DIR:-./model_save/grpo/}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-5e-7}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
NUM_GEN="${NUM_GEN:-4}"
LORA_RANK="${LORA_RANK:-64}"
MAX_TURNS="${MAX_TURNS:-10}"

echo "============================================"
echo "  GRPO 多轮工具调用训练 (trl)"
echo "============================================"
echo "模型:         $MODEL_PATH"
echo "训练数据:     $TRAIN_DATA"
echo "场景数据:     $SCENARIO_FILES"
echo "Cost Model:   $COST_MODEL"
echo "输出:         $OUTPUT_DIR"
echo "Batch:        ${BATCH_SIZE} x ${GRAD_ACCUM} = $((BATCH_SIZE * GRAD_ACCUM))"
echo "Generations:  $NUM_GEN"
echo "Max Turns:    $MAX_TURNS"
echo "============================================"

PYTHONPATH=. python -m training.trl.grpo \
    --model_path "$MODEL_PATH" \
    --train_data "$TRAIN_DATA" \
    --scenario_files $SCENARIO_FILES \
    --cost_model "$COST_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --num_generations $NUM_GEN \
    --lora_rank $LORA_RANK \
    --max_turns $MAX_TURNS \
    --max_completion_length 4096 \
    --max_prompt_length 1024 \
    --bf16
