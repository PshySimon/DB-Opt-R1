#!/bin/bash
set -euo pipefail

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$COMMON_DIR/../_common.sh"

run_trl_sft_train_experiment() {
    local exp_id="$1"
    local manifest_file="$2"
    local train_mode="$3"
    shift 3

    local manifest_path="$V2_MANIFEST_DIR/$manifest_file"
    local train_jsonl
    local model_path
    local run_name
    local output_dir
    local train_script

    train_jsonl="$(prepare_experiment_train_jsonl "$exp_id" "$manifest_path")"
    model_path="${MODEL_PATH:-${1:-Qwen/Qwen2.5-3B-Instruct}}"
    if [ $# -gt 0 ]; then
        shift
    fi
    if [ $# -gt 0 ]; then
        echo "错误: SFT TRL 包装脚本只支持可选的模型路径参数，额外参数请通过环境变量传递。"
        exit 1
    fi

    run_name="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
    output_dir="${OUTPUT_DIR:-$REPO_ROOT/model_save/experiments/v2/sft/trl/$train_mode/$exp_id/$run_name}"
    train_script="$REPO_ROOT/scripts/train_sft_trl_${train_mode}.sh"

    export BASE_MODEL="$model_path"
    export DATA_FILES="${DATA_FILES:-$train_jsonl}"
    export OUTPUT_DIR="$output_dir"
    export EPOCHS="${EPOCHS:-3}"
    export LR="${LR:-1e-5}"
    export BATCH_SIZE="${BATCH_SIZE:-2}"
    export GRAD_ACCUM="${GRAD_ACCUM:-4}"
    export MAX_LENGTH="${MAX_LENGTH:-8192}"
    export SFT_TRAIN_RATIO="${SFT_TRAIN_RATIO:-0.95}"
    export N_GPUS="${N_GPUS:-1}"
    export CUDA_DEVICES="${CUDA_DEVICES:-0}"
    export TRAIN_CONFIG_JSON="${TRAIN_CONFIG_JSON:-$output_dir/train_config.json}"
    export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
    export FLASH_ATTN="${FLASH_ATTN:-false}"
    if [ "$train_mode" = "lora" ]; then
        export LORA_RANK="${LORA_RANK:-64}"
    fi

    echo "============================================"
    echo "  v2 SFT 实验训练 (trl/$train_mode)"
    echo "============================================"
    echo "实验:     $exp_id"
    echo "manifest: $manifest_path"
    echo "数据:     $DATA_FILES"
    echo "模型:     $BASE_MODEL"
    echo "输出:     $OUTPUT_DIR"
    echo "============================================"

    exec bash "$train_script"
}
