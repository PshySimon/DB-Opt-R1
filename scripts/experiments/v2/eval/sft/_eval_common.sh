#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../train/sft/_common.sh"

run_eval_experiment() {
    local exp_id="$1"
    local run_name="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
    local output_root="${OUTPUT_ROOT:-$REPO_ROOT/eval_results/v2/sft/$exp_id/$run_name}"
    local rollout_dir="$output_root/rollouts"
    local report_dir="$output_root/report"
    local cost_model="${COST_MODEL:-$V2_COST_MODEL_DEFAULT}"

    ensure_file "$V2_EVAL_QUESTIONS"
    ensure_file "$V2_EVAL_SCENARIOS"

    if [ -z "${MODEL:-}" ]; then
        echo "错误: 评估前请设置 MODEL（服务端模型名）"
        exit 1
    fi

    local sampler_args=(
        -m data_pipeline.synthesis.trajectory.sampler
        --mode eval
        --eval-questions "$V2_EVAL_QUESTIONS"
        --scenarios "$V2_EVAL_SCENARIOS"
        --cost-model "$cost_model"
        --knob-space "$V2_KNOB_SPACE"
        --output-dir "$rollout_dir"
        --output-file "sft_trajectories.jsonl"
        --model "$MODEL"
        --parallel "${PARALLEL:-8}"
        --max-turns "${MAX_TURNS:-10}"
    )

    if [ -n "${PROVIDERS_CONFIG:-}" ]; then
        sampler_args+=(--providers-config "$PROVIDERS_CONFIG")
    else
        if [ -z "${API_BASE:-}" ]; then
            echo "错误: 未设置 PROVIDERS_CONFIG 时，必须提供 API_BASE"
            exit 1
        fi
        sampler_args+=(--api-base "$API_BASE" --api-key "${API_KEY:-EMPTY}")
    fi

    mkdir -p "$rollout_dir" "$report_dir"

    echo "============================================"
    echo "  v2 SFT 实验评估"
    echo "============================================"
    echo "实验:       $exp_id"
    echo "模型:       $MODEL"
    echo "rollouts:   $rollout_dir"
    echo "report:     $report_dir"
    echo "cost model: $cost_model"
    echo "============================================"

    python "${sampler_args[@]}"

    local report_args=(
        -m evaluate.run
        --eval-data "$rollout_dir/sft_trajectories.jsonl"
        --scenarios "$V2_EVAL_SCENARIOS"
        --cost-model "$cost_model"
        --knob-space "$V2_KNOB_SPACE"
        --output "$report_dir"
        --skip-bo
    )

    if [ "${WITH_BO:-false}" = "true" ]; then
        report_args=(
            -m evaluate.run
            --eval-data "$rollout_dir/sft_trajectories.jsonl"
            --scenarios "$V2_EVAL_SCENARIOS"
            --cost-model "$cost_model"
            --knob-space "$V2_KNOB_SPACE"
            --output "$report_dir"
            --with-bo
            --bo-trials "${BO_TRIALS:-200}"
        )
    fi

    python "${report_args[@]}"
}
