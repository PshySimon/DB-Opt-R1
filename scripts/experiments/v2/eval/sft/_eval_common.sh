#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../train/sft/_common.sh"

ensure_loopback_no_proxy() {
    local api_base="$1"

    case "$api_base" in
        http://127.0.0.1:*|https://127.0.0.1:*|http://localhost:*|https://localhost:*|http://[::1]:*|https://[::1]:*)
            local loopback_hosts="127.0.0.1,localhost,::1"
            export no_proxy="${no_proxy:+$no_proxy,}$loopback_hosts"
            export NO_PROXY="${NO_PROXY:+$NO_PROXY,}$loopback_hosts"
            echo "no_proxy:   $NO_PROXY"
            ;;
    esac
}

verify_sync_commit_blocked() {
    local rollout_file="$1"

    if [ "${BLOCK_SYNC_COMMIT:-true}" != "true" ]; then
        return 0
    fi

    python - "$rollout_file" <<'PY'
import json
import sys

path = sys.argv[1]
rows = 0
tried = 0
ignored = 0
changed = 0
examples = []


def loads(value):
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value or "{}")
    except Exception:
        return {}


with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        rows += 1
        row = json.loads(line)
        history = row.get("tracking", {}).get("tool_history", [])

        initial = {}
        for item in history:
            if item.get("tool") == "get_current_config":
                initial = loads(item.get("result"))
                break
        final = row.get("tracking", {}).get("last_valid_config") or {}
        final_changed = initial.get("synchronous_commit") != final.get("synchronous_commit")

        row_tried = False
        row_ignored = False
        for item in history:
            if item.get("tool") != "set_knob":
                continue
            args = item.get("args") or {}
            knobs = loads(args.get("knobs"))
            if "synchronous_commit" in knobs:
                row_tried = True
            result = loads(item.get("result"))
            ignored_items = result.get("ignored") or []
            if any(isinstance(x, dict) and x.get("name") == "synchronous_commit" for x in ignored_items):
                row_ignored = True

        tried += int(row_tried)
        ignored += int(row_ignored)
        changed += int(final_changed)
        if (row_tried or row_ignored or final_changed) and len(examples) < 5:
            examples.append(
                {
                    "env_sample_idx": row.get("env_sample_idx"),
                    "initial": initial.get("synchronous_commit"),
                    "final": final.get("synchronous_commit"),
                    "tried": row_tried,
                    "ignored": row_ignored,
                }
            )

print(
    "[sync-commit-check] "
    f"rows={rows} tried={tried} ignored={ignored} final_changed={changed} "
    f"examples={examples}"
)

if tried != ignored or changed:
    raise SystemExit(
        "synchronous_commit block check failed: every attempted change must be ignored "
        "and final synchronous_commit must stay unchanged"
    )
PY
}

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
    echo "block sync: ${BLOCK_SYNC_COMMIT:-true}"
    echo "============================================"

    if [ "${BLOCK_SYNC_COMMIT:-true}" = "true" ]; then
        export DBOPT_BLOCK_SYNC_COMMIT=1
    fi

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
            --api-max-concurrent "${API_MAX_CONCURRENT:-${PARALLEL:-8}}"
            --max-turns "${MAX_TURNS:-10}"
        )

        if [ -n "${PROVIDERS_CONFIG:-}" ]; then
            sampler_args+=(--providers-config "$PROVIDERS_CONFIG")
        else
            if [ -z "${API_BASE:-}" ]; then
                echo "错误: 未设置 PROVIDERS_CONFIG 时，必须提供 API_BASE"
                exit 1
            fi
            ensure_loopback_no_proxy "$API_BASE"
            sampler_args+=(--api-base "$API_BASE" --api-key "${API_KEY:-EMPTY}")
        fi

        python "${sampler_args[@]}"
    fi

    verify_sync_commit_blocked "$rollout_dir/sft_trajectories.jsonl"

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
