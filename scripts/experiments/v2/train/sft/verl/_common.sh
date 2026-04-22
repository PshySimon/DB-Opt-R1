#!/bin/bash
set -euo pipefail

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$COMMON_DIR/../_common.sh"

run_verl_sft_train_experiment() {
    local exp_id="$1"
    local manifest_file="$2"
    local train_mode="$3"
    shift 3

    local manifest_path="$V2_MANIFEST_DIR/$manifest_file"
    local data_dir
    local model_path
    local run_name
    local output_dir
    local train_script

    data_dir="$(prepare_experiment_train_parquet "$exp_id" "$manifest_path")"
    model_path="${MODEL_PATH:-${1:-Qwen/Qwen2.5-3B-Instruct}}"
    if [ $# -gt 0 ]; then
        shift
    fi
    if [ $# -gt 0 ]; then
        echo "错误: SFT VERL 包装脚本只支持可选的模型路径参数，额外参数请通过环境变量传递。"
        exit 1
    fi

    run_name="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
    output_dir="${OUTPUT_DIR:-$REPO_ROOT/model_save/experiments/v2/sft/verl/$train_mode/$exp_id/$run_name}"
    train_script="$REPO_ROOT/scripts/train_sft_verl_${train_mode}.sh"

    export BASE_MODEL="$model_path"
    export DATA_DIR="${DATA_DIR:-$data_dir}"
    export PROJECT_NAME="${PROJECT_NAME:-db_opt_r1_v2_sft}"
    export OUTPUT_DIR="${OUTPUT_DIR:-$output_dir}"
    export SFT_OUTPUT_DIR="$OUTPUT_DIR"
    export SFT_EXPERIMENT_NAME="${SFT_EXPERIMENT_NAME:-v2-sft-$train_mode-$exp_id-$run_name}"
    export TRAIN_CONFIG_JSON="${TRAIN_CONFIG_JSON:-$SFT_OUTPUT_DIR/train_config.json}"
    export MAX_LENGTH="${MAX_LENGTH:-8192}"
    export LR="${LR:-1e-6}"
    export EPOCHS="${EPOCHS:-3}"
    export BATCH_SIZE="${BATCH_SIZE:-16}"
    export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
    export N_GPUS="${N_GPUS:-2}"
    export ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
    if [ "$train_mode" = "lora" ]; then
        export LORA_RANK="${LORA_RANK:-64}"
        export LORA_ALPHA="${LORA_ALPHA:-$LORA_RANK}"
        export TARGET_MODULES="${TARGET_MODULES:-all-linear}"
    fi

    echo "============================================"
    echo "  v2 SFT 实验训练 (verl/$train_mode)"
    echo "============================================"
    echo "实验:     $exp_id"
    echo "manifest: $manifest_path"
    echo "数据:     $DATA_DIR"
    echo "模型:     $BASE_MODEL"
    echo "输出:     $SFT_OUTPUT_DIR"
    echo "============================================"

    exec bash "$train_script"
}
