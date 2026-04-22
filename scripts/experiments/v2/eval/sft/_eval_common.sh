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

    mkdir -p "$rollout_dir" "$report_dir"

    echo "============================================"
    echo "  v2 SFT 实验评估"
    echo "============================================"
    echo "实验:       $exp_id"
    echo "rollouts:   $rollout_dir"
    echo "report:     $report_dir"
    echo "cost model: $cost_model"
    echo "============================================"

    if [ -n "${LOCAL_MODEL_PATH:-}" ]; then
        echo "本地模型:   $LOCAL_MODEL_PATH"
        local local_eval_args=(
            "$REPO_ROOT/scripts/run_local_transformers_eval.py"
            --model-path "$LOCAL_MODEL_PATH"
            --eval-questions "$V2_EVAL_QUESTIONS"
            --scenarios "$V2_EVAL_SCENARIOS"
            --cost-model "$cost_model"
            --knob-space "$V2_KNOB_SPACE"
            --output-dir "$rollout_dir"
            --output-file "sft_trajectories.jsonl"
            --parallel "${PARALLEL:-1}"
            --max-turns "${MAX_TURNS:-10}"
            --device "${LOCAL_DEVICE:-cuda}"
            --dtype "${LOCAL_DTYPE:-bfloat16}"
            --max-new-tokens "${LOCAL_MAX_NEW_TOKENS:-512}"
            --log-interval "${LOCAL_LOG_INTERVAL:-20}"
        )

        if [ -n "${LOCAL_ATTN_IMPL:-}" ]; then
            local_eval_args+=(--attn-implementation "$LOCAL_ATTN_IMPL")
        fi
        if [ -n "${START_INDEX:-}" ]; then
            local_eval_args+=(--start-index "$START_INDEX")
        fi
        if [ -n "${END_INDEX:-}" ]; then
            local_eval_args+=(--end-index "$END_INDEX")
        fi
        if [ -n "${SOURCE_FILTER:-}" ]; then
            local_eval_args+=(--source-filter "$SOURCE_FILTER")
        fi
        if [ -n "${TPS_MIN:-}" ]; then
            local_eval_args+=(--tps-min "$TPS_MIN")
        fi
        if [ -n "${TPS_MAX:-}" ]; then
            local_eval_args+=(--tps-max "$TPS_MAX")
        fi

        python "${local_eval_args[@]}"
    else
        if [ -z "${MODEL:-}" ]; then
            echo "错误: API 评估模式请设置 MODEL；本地评估模式请设置 LOCAL_MODEL_PATH"
            exit 1
        fi

        echo "模型:       $MODEL"
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

        python "${sampler_args[@]}"
    fi

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
