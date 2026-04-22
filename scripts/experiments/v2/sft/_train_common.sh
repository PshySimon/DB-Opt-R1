#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

run_train_experiment() {
    local exp_id="$1"
    local manifest_file="$2"
    local manifest_path="$V2_MANIFEST_DIR/$manifest_file"
    local train_jsonl
    local run_name

    train_jsonl="$(prepare_experiment_train_jsonl "$exp_id" "$manifest_path")"
    run_name="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"

    export DATA_FILES="${DATA_FILES:-$train_jsonl}"
    export OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/model_save/experiments/v2/sft/$exp_id/$run_name}"
    export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
    export EPOCHS="${EPOCHS:-3}"
    export LR="${LR:-1e-5}"
    export BATCH_SIZE="${BATCH_SIZE:-2}"
    export GRAD_ACCUM="${GRAD_ACCUM:-4}"
    export LORA_RANK="${LORA_RANK:-64}"
    export MAX_LENGTH="${MAX_LENGTH:-4096}"
    export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
    export FLASH_ATTN="${FLASH_ATTN:-false}"

    echo "============================================"
    echo "  v2 SFT 实验训练"
    echo "============================================"
    echo "实验:     $exp_id"
    echo "manifest: $manifest_path"
    echo "数据:     $DATA_FILES"
    echo "输出:     $OUTPUT_DIR"
    echo "============================================"

    exec bash "$REPO_ROOT/scripts/train_sft_trl_lora.sh"
}
