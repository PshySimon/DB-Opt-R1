#!/bin/bash
# vLLM 部署 Qwen3-4B SFT 模型
set -e

MODEL_PATH="${MODEL_PATH:-/root/private_data/DB-Opt-R1/model_save/sft_qwen3_4b_cleaned_merged}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_UTIL="${GPU_UTIL:-0.90}"
TP="${TP:-1}"

echo "============================================"
echo "  vLLM 部署 Qwen3-4B"
echo "============================================"
echo "模型:         $MODEL_PATH"
echo "端口:         $PORT"
echo "上下文长度:   $MAX_MODEL_LEN"
echo "GPU 利用率:   $GPU_UTIL"
echo "TP:           $TP"
echo "============================================"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name qwen3-4b-sft \
    --port $PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_UTIL \
    --tensor-parallel-size $TP \
    --trust-remote-code \
    --dtype bfloat16 \
    --disable-log-requests
