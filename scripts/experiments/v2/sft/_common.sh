#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"

V2_TRAIN_DATA="$REPO_ROOT/data_pipeline/data/train/v2/sft_trajectories_v10_train.jsonl"
V2_MANIFEST_DIR="$REPO_ROOT/data_pipeline/data/train/v2/manifests"
V2_EVAL_QUESTIONS="$REPO_ROOT/data_pipeline/data/eval/v2/eval_trajectories_v2.jsonl"
V2_EVAL_SCENARIOS="$REPO_ROOT/data_pipeline/data/eval/v2/collected_eval_v2.json"
V2_KNOB_SPACE="$REPO_ROOT/configs/knob_space.yaml"
V2_COST_MODEL_DEFAULT="${V2_COST_MODEL:-$REPO_ROOT/cost_model/checkpoints/v8_lgbm}"

ensure_file() {
    local path="$1"
    if [ ! -f "$path" ]; then
        echo "错误: 未找到文件 $path"
        exit 1
    fi
}

prepare_experiment_train_jsonl() {
    local exp_id="$1"
    local manifest_path="$2"
    local data_dir="${EXPERIMENT_DATA_DIR:-$REPO_ROOT/datasets/sft_v2/$exp_id}"
    local output_jsonl="$data_dir/train.jsonl"
    local stats_json="$data_dir/train_stats.json"

    ensure_file "$V2_TRAIN_DATA"
    ensure_file "$manifest_path"
    mkdir -p "$data_dir"

    if [ "${FORCE_REBUILD_DATA:-false}" = "true" ] || [ ! -f "$output_jsonl" ]; then
        python -m data_pipeline.build_sft_experiment_dataset_v2 \
            --train-data "$V2_TRAIN_DATA" \
            --manifest "$manifest_path" \
            --output-jsonl "$output_jsonl" \
            --stats-output "$stats_json"
    fi

    echo "$output_jsonl"
}
