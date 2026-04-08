#!/bin/bash
# 基准脚本：抽取 16 条样本，跑 1 个 GRPO LoRA step，测整体耗时
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/root/private_data/DB-Opt-R1/model_save/sft_qwen3_4b_cleaned_merged}"
TRAIN_DATA="${TRAIN_DATA:-./data_pipeline/data/train/sft_trajectories.jsonl}"
SCENARIO_FILES="${SCENARIO_FILES:-data_pipeline/data/scenarios/collected/collected_server1.json data_pipeline/data/scenarios/collected/collected_server2.json data_pipeline/data/scenarios/collected/collected_server3.json}"
COST_MODEL="${COST_MODEL:-./cost_model/checkpoints/v9_lgbm}"
OUTPUT_DIR="${OUTPUT_DIR:-./model_save/grpo_bench_16/}"

NUM_SAMPLES="${NUM_SAMPLES:-16}"
EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-1}"
LR="${LR:-5e-7}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
NUM_GEN="${NUM_GEN:-4}"
LORA_RANK="${LORA_RANK:-64}"
MAX_TURNS="${MAX_TURNS:-10}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-4096}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
VLLM_SERVER_HOST="${VLLM_SERVER_HOST:-127.0.0.1}"
VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-8000}"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-qwen3-4b-sft}"
VLLM_TIMEOUT="${VLLM_TIMEOUT:-300}"
VLLM_MAX_TOKENS="${VLLM_MAX_TOKENS:-1024}"
ROLLOUT_LOG_INTERVAL="${ROLLOUT_LOG_INTERVAL:-1}"

TMP_TRAIN_DATA="$(mktemp /tmp/grpo-bench-16.XXXXXX.jsonl)"
trap 'rm -f "$TMP_TRAIN_DATA"' EXIT

python - "$TRAIN_DATA" "$TMP_TRAIN_DATA" "$NUM_SAMPLES" <<'PY'
import json
import sys

src_path, dst_path, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
seen = set()
written = 0

with open(src_path, "r", encoding="utf-8") as src, open(dst_path, "w", encoding="utf-8") as dst:
    for line in src:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        question = row.get("question", "")
        if not question or question in seen:
            continue
        seen.add(question)
        dst.write(json.dumps(row, ensure_ascii=False) + "\n")
        written += 1
        if written >= limit:
            break

if written < limit:
    raise SystemExit(f"only found {written} unique questions, expected at least {limit}")
PY

echo "============================================"
echo "  GRPO 16 样本基准 (trl, LoRA)"
echo "============================================"
echo "模型:         $MODEL_PATH"
echo "原始数据:     $TRAIN_DATA"
echo "基准数据:     $TMP_TRAIN_DATA"
echo "样本数:       $NUM_SAMPLES"
echo "Batch:        ${BATCH_SIZE} x ${NUM_GEN} x ${GRAD_ACCUM}"
echo "Turns:        $MAX_TURNS"
echo "Lengths:      prompt=${MAX_PROMPT_LENGTH}, completion=${MAX_COMPLETION_LENGTH}, vllm=${VLLM_MAX_TOKENS}"
echo "Attention:    ${ATTN_IMPL}"
echo "vLLM 服务:    http://${VLLM_SERVER_HOST}:${VLLM_SERVER_PORT}/v1"
echo "============================================"

START_TS="$(date +%s)"

PYTHONPATH=. python -m training.trl.grpo \
    --model_path "$MODEL_PATH" \
    --train_data "$TMP_TRAIN_DATA" \
    --scenario_files $SCENARIO_FILES \
    --cost_model "$COST_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs $EPOCHS \
    --max_steps $MAX_STEPS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --num_generations $NUM_GEN \
    --lora_rank $LORA_RANK \
    --max_turns $MAX_TURNS \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    --max_prompt_length $MAX_PROMPT_LENGTH \
    --use_lora \
    --attn_impl $ATTN_IMPL \
    --vllm_server_host $VLLM_SERVER_HOST \
    --vllm_server_port $VLLM_SERVER_PORT \
    --vllm_model_name $VLLM_MODEL_NAME \
    --vllm_timeout $VLLM_TIMEOUT \
    --vllm_max_tokens $VLLM_MAX_TOKENS \
    --rollout_log_interval $ROLLOUT_LOG_INTERVAL \
    --bf16

END_TS="$(date +%s)"
ELAPSED="$((END_TS - START_TS))"

echo "============================================"
echo "  基准完成"
echo "============================================"
echo "总耗时: ${ELAPSED}s"
echo "约合:   $((ELAPSED / 60))m $((ELAPSED % 60))s"
echo "============================================"
