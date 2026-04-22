#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../" && pwd)"

V2_RL_DIR="$REPO_ROOT/data_pipeline/data/train/v2/rl"
V2_RL_FRONTIER="$V2_RL_DIR/rl_frontier_1q.jsonl"
V2_RL_HARD="$V2_RL_DIR/rl_hard_1q.jsonl"
V2_RL_FRONTIER_PLUS_HARD="$V2_RL_DIR/rl_frontier_plus_hard_1q.jsonl"

V2_RL_SCENARIOS_DEFAULT="$REPO_ROOT/data_pipeline/data/scenarios/collected/collected_8c16g_hdd_20k.json"
V2_COST_MODEL_DEFAULT="${V2_COST_MODEL:-$REPO_ROOT/cost_model/checkpoints/v10_lgbm}"

ensure_file() {
    local path="$1"
    if [ ! -f "$path" ]; then
        echo "错误: 未找到文件 $path"
        exit 1
    fi
}

prepare_rl_parquet_dataset() {
    local exp_id="$1"
    local input_jsonl="$2"
    local data_dir="${EXPERIMENT_DATA_DIR:-$REPO_ROOT/datasets/rl_v2/$exp_id}"

    ensure_file "$input_jsonl"
    ensure_file "$V2_RL_SCENARIOS_DEFAULT"
    mkdir -p "$data_dir"

    if [ "${FORCE_REBUILD_DATA:-false}" = "true" ] || [ ! -f "$data_dir/train.parquet" ] || [ ! -f "$data_dir/validation.parquet" ]; then
        python -m data_pipeline.preprocess_grpo \
            --input-files "$input_jsonl" \
            --scenarios "$V2_RL_SCENARIOS_DEFAULT" \
            --output-dir "$data_dir" \
            --val-ratio "${RL_VAL_RATIO:-0.1}" \
            --seed "${RL_DATA_SEED:-42}" \
            --questions-per-scene 1
    fi

    echo "$data_dir"
}
