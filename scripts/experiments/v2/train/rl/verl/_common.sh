#!/bin/bash
set -euo pipefail

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$COMMON_DIR/../_common.sh"

run_verl_rl_train_experiment() {
    local exp_id="$1"
    local input_jsonl="$2"
    local train_mode="$3"
    shift 3

    local data_dir
    local model_path
    local run_name
    local output_dir
    local train_script

    data_dir="$(prepare_rl_parquet_dataset "$exp_id" "$input_jsonl")"
    model_path="${MODEL_PATH:-${1:-/root/private_data/DB-Opt-R1/model_save/sft_qwen3_4b_cleaned_merged}}"
    if [ $# -gt 0 ]; then
        shift
    fi

    run_name="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
    output_dir="${OUTPUT_DIR:-$REPO_ROOT/model_save/experiments/v2/rl/verl/$train_mode/$exp_id/$run_name}"
    train_script="$REPO_ROOT/scripts/train_grpo_verl_${train_mode}.sh"

    export SFT_CHECKPOINT="$model_path"
    export OUTPUT_DIR="$output_dir"
    export TRAIN_DATA="${TRAIN_DATA:-$data_dir/train.parquet}"
    export VAL_DATA="${VAL_DATA:-$data_dir/validation.parquet}"
    export TRAIN_CONFIG_JSON="${TRAIN_CONFIG_JSON:-$OUTPUT_DIR/train_config.json}"
    export SCENARIO_FILES="${SCENARIO_FILES:-$V2_RL_SCENARIOS_DEFAULT}"
    export COST_MODEL_PATH="${COST_MODEL_PATH:-$V2_COST_MODEL_DEFAULT}"
    export PROJECT_NAME="${PROJECT_NAME:-db_opt_r1_v2_rl}"
    export GRPO_EXPERIMENT_NAME="${GRPO_EXPERIMENT_NAME:-v2-rl-$train_mode-$exp_id-$run_name}"
    export N_GPUS="${N_GPUS:-2}"
    export CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
    export LR="${LR:-1e-6}"
    export BATCH_SIZE="${BATCH_SIZE:-32}"
    export N_REPEAT="${N_REPEAT:-5}"
    export TOTAL_STEPS="${TOTAL_STEPS:-100}"
    export SAVE_FREQ="${SAVE_FREQ:-5}"
    export TEST_FREQ="${TEST_FREQ:-1}"
    export MAX_TURNS="${MAX_TURNS:-10}"
    export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
    export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
    export MAX_START_LENGTH="${MAX_START_LENGTH:-4096}"
    export MAX_TOOL_RESPONSE_LENGTH="${MAX_TOOL_RESPONSE_LENGTH:-2048}"
    export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
    export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-2}"
    export REF_LOG_PROB_MICRO_BATCH_SIZE="${REF_LOG_PROB_MICRO_BATCH_SIZE:-2}"
    export ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE:-2}"
    export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
    export FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE:-True}"
    export ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
    export DEBUG_ROLLOUT_DIR="${DEBUG_ROLLOUT_DIR:-$REPO_ROOT/debug/rollout/v2/$exp_id}"
    export REWARD_DEBUG_NUM_EXAMINE="${REWARD_DEBUG_NUM_EXAMINE:-0}"
    export VAL_REWARD_DEBUG_NUM_EXAMINE="${VAL_REWARD_DEBUG_NUM_EXAMINE:-1}"
    export EARLY_STOPPING_ENABLED="${EARLY_STOPPING_ENABLED:-False}"
    export EARLY_STOPPING_METRIC="${EARLY_STOPPING_METRIC:-val/compiler_autotuning/db_tuning/test_score}"
    export EARLY_STOPPING_MODE="${EARLY_STOPPING_MODE:-max}"
    export EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-5}"
    export EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0.0}"
    if [ "$train_mode" = "lora" ]; then
        export LORA_RANK="${LORA_RANK:-64}"
        export LORA_ALPHA="${LORA_ALPHA:-$LORA_RANK}"
        export TARGET_MODULES="${TARGET_MODULES:-all-linear}"
    fi

    echo "============================================"
    echo "  v2 RL 实验训练 (verl/$train_mode)"
    echo "============================================"
    echo "实验:         $exp_id"
    echo "数据目录:     $data_dir"
    echo "模型:         $SFT_CHECKPOINT"
    echo "场景文件:     $SCENARIO_FILES"
    echo "Cost Model:   $COST_MODEL_PATH"
    echo "输出目录:     $output_dir"
    echo "============================================"

    exec bash "$train_script" "$@"
}
