#!/bin/bash
# trl GRPO 多轮工具调用训练（vLLM 加速版）
set -e

MODEL_PATH="${MODEL_PATH:-/root/private_data/DB-Opt-R1/model_save/sft_qwen3_4b_cleaned_merged}"
TRAIN_DATA="${TRAIN_DATA:-./data_pipeline/data/train/sft_trajectories.jsonl}"
SCENARIO_FILES="${SCENARIO_FILES:-data_pipeline/data/scenarios/collected/collected_server1.json data_pipeline/data/scenarios/collected/collected_server2.json data_pipeline/data/scenarios/collected/collected_server3.json}"
COST_MODEL="${COST_MODEL:-./cost_model/checkpoints/v9_lgbm}"
OUTPUT_DIR="${OUTPUT_DIR:-./model_save/grpo/}"
VLLM_URL="${VLLM_URL:-http://localhost:8000/v1}"
VLLM_MODEL="${VLLM_MODEL:-qwen3-4b-sft}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-5e-7}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
NUM_GEN="${NUM_GEN:-4}"
LORA_RANK="${LORA_RANK:-64}"
MAX_TURNS="${MAX_TURNS:-10}"
ROLLOUT_CONCURRENCY="${ROLLOUT_CONCURRENCY:-8}"

echo "============================================"
echo "  GRPO 训练 (vLLM 加速)"
echo "============================================"
echo "HF 模型:      $MODEL_PATH"
echo "vLLM:         $VLLM_URL ($VLLM_MODEL)"
echo "训练数据:     $TRAIN_DATA"
echo "输出:         $OUTPUT_DIR"
echo "Batch:        ${BATCH_SIZE} x ${NUM_GEN} x ${GRAD_ACCUM}"
echo "Rollout 并发: $ROLLOUT_CONCURRENCY"
echo "============================================"

PYTHONPATH=. python -m training.trl.grpo \
    --model_path "$MODEL_PATH" \
    --train_data "$TRAIN_DATA" \
    --scenario_files $SCENARIO_FILES \
    --cost_model "$COST_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --vllm_base_url "$VLLM_URL" \
    --vllm_model "$VLLM_MODEL" \
    --num_epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --num_generations $NUM_GEN \
    --lora_rank $LORA_RANK \
    --max_turns $MAX_TURNS \
    --max_completion_length 4096 \
    --max_prompt_length 1024 \
    --rollout_concurrency $ROLLOUT_CONCURRENCY \
    --bf16
