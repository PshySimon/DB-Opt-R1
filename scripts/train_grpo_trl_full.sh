#!/bin/bash
# trl GRPO 多轮工具调用训练（外置 vLLM server, 全量）
set -e

MODEL_PATH="${MODEL_PATH:-/root/private_data/DB-Opt-R1/model_save/sft_qwen3_4b_cleaned_merged}"
TRAIN_DATA="${TRAIN_DATA:-./data_pipeline/data/train/sft_trajectories.jsonl}"
SCENARIO_FILES="${SCENARIO_FILES:-data_pipeline/data/scenarios/collected/collected_server1.json data_pipeline/data/scenarios/collected/collected_server2.json data_pipeline/data/scenarios/collected/collected_server3.json}"
COST_MODEL="${COST_MODEL:-./cost_model/checkpoints/v9_lgbm}"
OUTPUT_DIR="${OUTPUT_DIR:-./model_save/grpo_full/}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-5e-7}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
NUM_GEN="${NUM_GEN:-4}"
MAX_TURNS="${MAX_TURNS:-10}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
VLLM_SERVER_HOST="${VLLM_SERVER_HOST:-127.0.0.1}"
VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-8000}"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-qwen3-4b-sft}"
VLLM_TIMEOUT="${VLLM_TIMEOUT:-300}"
VLLM_MAX_TOKENS="${VLLM_MAX_TOKENS:-1024}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-}"

echo "============================================"
echo "  GRPO 训练 (trl, 全量)"
echo "============================================"
echo "模型:         $MODEL_PATH"
echo "训练数据:     $TRAIN_DATA"
echo "输出:         $OUTPUT_DIR"
echo "Batch:        ${BATCH_SIZE} x ${NUM_GEN} x ${GRAD_ACCUM}"
echo "Attention:    ${ATTN_IMPL}"
echo "vLLM 服务:    http://${VLLM_SERVER_HOST}:${VLLM_SERVER_PORT}/v1"
echo "vLLM 模型:    ${VLLM_MODEL_NAME}"
echo "Rollout 并发: ${ROLLOUT_BATCH_SIZE:-TRL默认}"
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
    --max_turns $MAX_TURNS \
    --max_completion_length 4096 \
    --max_prompt_length 1024 \
    --full_finetune \
    --attn_impl $ATTN_IMPL \
    --vllm_server_host $VLLM_SERVER_HOST \
    --vllm_server_port $VLLM_SERVER_PORT \
    --vllm_model_name $VLLM_MODEL_NAME \
    --vllm_timeout $VLLM_TIMEOUT \
    --vllm_max_tokens $VLLM_MAX_TOKENS \
    $( [ -n "$ROLLOUT_BATCH_SIZE" ] && echo "--rollout_batch_size $ROLLOUT_BATCH_SIZE" ) \
    --bf16
