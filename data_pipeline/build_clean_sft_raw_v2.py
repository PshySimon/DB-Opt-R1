import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import re

import numpy as np
import yaml

from cost_model.model import CostModel


FAMILY_MAP = {
    "shared_buffers": "memory",
    "work_mem": "memory",
    "effective_cache_size": "memory",
    "maintenance_work_mem": "memory",
    "wal_buffers": "memory",
    "temp_buffers": "memory",
    "huge_pages": "memory",
    "max_stack_depth": "memory",
    "random_page_cost": "io_cost",
    "seq_page_cost": "io_cost",
    "cpu_tuple_cost": "io_cost",
    "cpu_index_tuple_cost": "io_cost",
    "cpu_operator_cost": "io_cost",
    "effective_io_concurrency": "io_cost",
    "default_statistics_target": "parallel_stats",
    "from_collapse_limit": "parallel_stats",
    "join_collapse_limit": "parallel_stats",
    "geqo_threshold": "parallel_stats",
    "max_parallel_workers_per_gather": "parallel_stats",
    "max_parallel_workers": "parallel_stats",
    "max_parallel_maintenance_workers": "parallel_stats",
    "checkpoint_completion_target": "wal_checkpoint",
    "max_wal_size": "wal_checkpoint",
    "min_wal_size": "wal_checkpoint",
    "checkpoint_timeout": "wal_checkpoint",
    "wal_compression": "wal_checkpoint",
    "synchronous_commit": "wal_checkpoint",
    "commit_delay": "wal_checkpoint",
    "bgwriter_delay": "bgwriter",
    "bgwriter_lru_maxpages": "bgwriter",
    "bgwriter_lru_multiplier": "bgwriter",
    "max_connections": "connection",
    "superuser_reserved_connections": "connection",
    "idle_in_transaction_session_timeout": "connection",
    "deadlock_timeout": "connection",
    "lock_timeout": "connection",
    "autovacuum": "autovacuum",
    "autovacuum_max_workers": "autovacuum",
    "autovacuum_naptime": "autovacuum",
    "autovacuum_vacuum_threshold": "autovacuum",
    "autovacuum_vacuum_scale_factor": "autovacuum",
    "autovacuum_analyze_threshold": "autovacuum",
    "autovacuum_analyze_scale_factor": "autovacuum",
    "track_activities": "tracking",
    "track_counts": "tracking",
}

ALLOWED_TOOL_NAMES = {
    "finish_tuning",
    "get_hardware_info",
    "get_current_config",
    "get_db_metrics",
    "get_workload_info",
    "view_logs",
    "get_system_stats",
    "set_knob",
    "restart_pg",
    "reload_pg",
    "reset_config",
    "predict_performance",
    "run_benchmark",
}


def quantile(values, p):
    values = sorted(values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[f]
    return values[f] * (c - k) + values[c] * (k - f)


def parse_tool_calls(content: str):
    if "<tool_call>" not in (content or ""):
        return [], False
    matches = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, re.S)
    if not matches:
        return [], True
    calls = []
    malformed = False
    for blob in matches:
        try:
            calls.append(json.loads(blob))
        except Exception:
            malformed = True
    return calls, malformed


def parse_knobs(arguments):
    if not isinstance(arguments, dict):
        return {}
    knobs = arguments.get("knobs", {})
    if isinstance(knobs, str):
        try:
            knobs = json.loads(knobs)
        except Exception:
            return {}
    return knobs if isinstance(knobs, dict) else {}


def action_pattern(families):
    families = sorted(set(families))
    if not families:
        return "no_knob"
    if len(families) == 1:
        return families[0]
    if len(families) == 2:
        return "+".join(families)
    return "mixed_3plus"


def extract_row(record, file_name, line_no, workloads_by_env, static_knobs, family_map=None):
    family_map = family_map or {}
    env = record.get("env_sample_idx")
    workload = workloads_by_env.get(env, "unknown")
    pending_restart = False
    config_effective = True
    has_tool_hallucination = False
    has_protocol_error = False
    has_finish_tuning = False
    has_think = False
    families = []
    predict_payloads = []
    predict_count = 0

    for msg in record.get("messages", []):
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "assistant":
            if "<think>" in content and "</think>" in content:
                has_think = True
            stripped = content.strip()
            if stripped.startswith('{"name"') or stripped.startswith("{'name'"):
                has_protocol_error = True
            tool_calls, malformed = parse_tool_calls(content)
            if malformed:
                has_protocol_error = True
            for tool_call in tool_calls:
                name = tool_call.get("name")
                if name not in ALLOWED_TOOL_NAMES:
                    has_tool_hallucination = True
                    has_protocol_error = True
                arguments = tool_call.get("arguments", {})
                if name == "set_knob":
                    knobs = parse_knobs(arguments)
                    for knob in knobs:
                        families.append(family_map.get(knob, "other"))
                        if knob in static_knobs:
                            pending_restart = True
                elif name in ("restart_pg", "reset_config"):
                    pending_restart = False
                elif name == "predict_performance":
                    predict_count += 1
                    if pending_restart:
                        config_effective = False
                elif name == "finish_tuning":
                    has_finish_tuning = True
        elif role == "tool":
            if "Unknown tool:" in content:
                has_tool_hallucination = True
                has_protocol_error = True
            if "Invalid tool call format" in content:
                has_protocol_error = True
            try:
                payload = json.loads(content)
            except Exception:
                continue
            if isinstance(payload, dict) and {
                "predicted_tps",
                "baseline_tps",
                "actual_tps",
                "improvement_pct",
            } <= payload.keys():
                predict_payloads.append(payload)

    best_predict = None
    if predict_payloads:
        best_predict = max(
            predict_payloads,
            key=lambda payload: float(payload.get("improvement_pct", 0) or 0),
        )

    return {
        "file": file_name,
        "line_no": line_no,
        "env": env,
        "workload": workload,
        "pattern": action_pattern(families),
        "imp": float(best_predict.get("improvement_pct", 0) or 0) if best_predict else 0.0,
        "outer_imp": float(record.get("improvement_pct", 0) or 0),
        "final": best_predict,
        "config_effective": config_effective,
        "has_tool_hallucination": has_tool_hallucination,
        "has_protocol_error": has_protocol_error,
        "has_finish_tuning": has_finish_tuning,
        "has_think": has_think,
        "predict_count": predict_count,
        "record": record,
    }


def classify_high_gain_outlier(row, default_baseline_tps, bucket_upper):
    reasons = []
    improvement = row["imp"]
    final = row["final"] or {}
    predicted_tps = float(final.get("predicted_tps", 0) or 0)
    baseline_tps = float(final.get("baseline_tps", 0) or 0)
    actual_tps = float(final.get("actual_tps", 0) or 0)

    if improvement >= 200:
        reasons.append("cap200")
    if default_baseline_tps is not None and baseline_tps < default_baseline_tps:
        reasons.append("baseline_below_default")
    if bucket_upper:
        pred_over_actual = predicted_tps / max(actual_tps, 1)
        pred_over_baseline = predicted_tps / max(baseline_tps, 1)
        if pred_over_actual > bucket_upper["pred_over_actual_p99"]:
            reasons.append("pred_over_actual_above_p99")
        if pred_over_baseline > bucket_upper["pred_over_baseline_p99"]:
            reasons.append("pred_over_baseline_above_p99")
    return reasons


def build_reference_bounds(rows):
    bucket_ref = defaultdict(list)
    workload_ref = defaultdict(list)

    for row in rows:
        if not row["config_effective"]:
            continue
        if not (3 <= row["imp"] <= 50):
            continue
        final = row["final"]
        if not final:
            continue
        baseline_tps = float(final["baseline_tps"])
        predicted_tps = float(final["predicted_tps"])
        actual_tps = float(final["actual_tps"])
        if baseline_tps <= 1 or predicted_tps <= 0 or actual_tps <= 1:
            continue

        sample = {
            "pred_over_actual": predicted_tps / max(actual_tps, 1),
            "pred_over_baseline": predicted_tps / max(baseline_tps, 1),
        }
        bucket_ref[(row["workload"], row["pattern"])].append(sample)
        workload_ref[row["workload"]].append(sample)

    bucket_bounds = {}
    for key, items in bucket_ref.items():
        if len(items) < 30:
            continue
        bucket_bounds[key] = {
            "pred_over_actual_p99": quantile([x["pred_over_actual"] for x in items], 0.99),
            "pred_over_baseline_p99": quantile([x["pred_over_baseline"] for x in items], 0.99),
            "reference_count": len(items),
        }

    workload_bounds = {}
    for workload, items in workload_ref.items():
        workload_bounds[workload] = {
            "pred_over_actual_p99": quantile([x["pred_over_actual"] for x in items], 0.99),
            "pred_over_baseline_p99": quantile([x["pred_over_baseline"] for x in items], 0.99),
            "reference_count": len(items),
        }

    return bucket_bounds, workload_bounds


def predict_default_baselines(model, scenarios, default_knobs, env_ids):
    env_ids = sorted(env_ids)
    if not env_ids:
        return {}

    matrix = []
    for env in env_ids:
        scenario = scenarios[env]
        knobs = dict(default_knobs)
        knobs["workload"] = scenario["workload"]["type"]
        matrix.append(model.preprocessor.transform(knobs, dict(scenario["hardware"])))

    features = np.vstack(matrix)
    predictions = model.model.predict(features)
    return {env: float(np.expm1(pred)) for env, pred in zip(env_ids, predictions)}


def compute_clean_dataset(rows, scenarios, default_knobs, checkpoint_dir):
    bucket_bounds, workload_bounds = build_reference_bounds(rows)
    high_gain_envs = {
        row["env"]
        for row in rows
        if row["imp"] > 50 and isinstance(row["env"], int) and 0 <= row["env"] < len(scenarios)
    }
    model = CostModel.load(str(checkpoint_dir))
    default_baselines = predict_default_baselines(model, scenarios, default_knobs, high_gain_envs)

    kept = []
    stats = Counter()

    for row in rows:
        if row["imp"] <= 0:
            continue
        stats["positive_total"] += 1

        if row["has_tool_hallucination"]:
            stats["drop_tool_hallucination"] += 1
            continue

        if row.get("has_protocol_error"):
            stats["drop_protocol_error"] += 1
            continue

        if not row["config_effective"]:
            stats["drop_restart_invalid"] += 1
            continue

        if not row.get("has_finish_tuning", False):
            stats["drop_no_finish_tuning"] += 1
            continue

        if not row.get("has_think", False):
            stats["drop_no_think"] += 1
            continue

        if not row["final"]:
            stats["drop_no_predict_call"] += 1
            continue

        if row["imp"] > 50:
            upper = bucket_bounds.get((row["workload"], row["pattern"])) or workload_bounds.get(row["workload"])
            reasons = classify_high_gain_outlier(
                row=row,
                default_baseline_tps=default_baselines.get(row["env"]),
                bucket_upper=upper,
            )
            if reasons:
                stats["drop_high_gain_outlier"] += 1
                for reason in reasons:
                    stats[f"detail_{reason}"] += 1
                continue

        kept.append(row)
        stats["keep"] += 1

    deduped = select_best_per_env(kept)
    stats["drop_same_env_lower_gain"] = len(kept) - len(deduped)
    stats["keep"] = len(deduped)

    return deduped, stats, bucket_bounds, workload_bounds


def select_best_per_env(rows):
    best_by_env = {}
    for row in rows:
        env = row["env"]
        current = best_by_env.get(env)
        if current is None:
            best_by_env[env] = row
            continue
        if row["imp"] > current["imp"]:
            best_by_env[env] = row
            continue
        if row["imp"] == current["imp"]:
            cur_line = (current.get("file", ""), current.get("line_no", 10**18))
            new_line = (row.get("file", ""), row.get("line_no", 10**18))
            if new_line < cur_line:
                best_by_env[env] = row
    return list(best_by_env.values())


def summarize_improvements(rows):
    values = [row["imp"] for row in rows]
    summary = {
        "count": len(values),
        "quantiles": {
            "p10": quantile(values, 0.10),
            "p25": quantile(values, 0.25),
            "p50": quantile(values, 0.50),
            "p75": quantile(values, 0.75),
            "p90": quantile(values, 0.90),
            "p95": quantile(values, 0.95),
            "p99": quantile(values, 0.99),
        },
        "bins": {
            "0_1": sum(1 for v in values if 0 < v <= 1),
            "1_3": sum(1 for v in values if 1 < v < 3),
            "3_5": sum(1 for v in values if 3 <= v < 5),
            "5_10": sum(1 for v in values if 5 <= v < 10),
            "10_20": sum(1 for v in values if 10 <= v < 20),
            "20_30": sum(1 for v in values if 20 <= v < 30),
            "30_50": sum(1 for v in values if 30 <= v <= 50),
        },
        "coarse_bins": {
            "0_3": sum(1 for v in values if 0 < v < 3),
            "3_10": sum(1 for v in values if 3 <= v < 10),
            "10_30": sum(1 for v in values if 10 <= v < 30),
            "30_50": sum(1 for v in values if 30 <= v <= 50),
        },
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="构造 v2 清洗后的 SFT raw 数据集")
    parser.add_argument(
        "--input-files",
        nargs="+",
        default=[
            "data_pipeline/data/train/sft_trajectories_v10_part1.jsonl",
            "data_pipeline/data/train/sft_trajectories_v10_part2.jsonl",
        ],
    )
    parser.add_argument(
        "--scenarios",
        default="data_pipeline/data/scenarios/collected/collected_8c16g_hdd_20k.json",
    )
    parser.add_argument(
        "--checkpoint",
        default="cost_model/checkpoints/v10_lgbm",
    )
    parser.add_argument(
        "--knob-space",
        default="configs/knob_space.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="data_pipeline/data/pool/v2",
    )
    parser.add_argument(
        "--output-file",
        default="sft_trajectories_v10_cleaned_pool.jsonl",
    )
    parser.add_argument(
        "--stats-file",
        default="sft_trajectories_v10_cleaned_stats.json",
    )
    args = parser.parse_args()

    scenarios_path = Path(args.scenarios)
    scenarios = json.loads(scenarios_path.read_text())
    workloads_by_env = {idx: item["workload"]["type"] for idx, item in enumerate(scenarios)}

    knob_cfg = yaml.safe_load(Path(args.knob_space).read_text())["knobs"]
    default_knobs = {k: v["default"] for k, v in knob_cfg.items()}
    static_knobs = {k for k, v in knob_cfg.items() if v.get("restart")}

    rows = []
    for input_file in args.input_files:
        input_path = Path(input_file)
        with input_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                rows.append(
                    extract_row(
                        record=record,
                        file_name=input_path.name,
                        line_no=line_no,
                        workloads_by_env=workloads_by_env,
                        static_knobs=static_knobs,
                        family_map=FAMILY_MAP,
                    )
                )

    kept, stats, bucket_bounds, workload_bounds = compute_clean_dataset(
        rows=rows,
        scenarios=scenarios,
        default_knobs=default_knobs,
        checkpoint_dir=Path(args.checkpoint),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_dir / args.output_file
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in kept:
            record = dict(row["record"])
            record["improvement_pct"] = round(row["imp"], 2)
            record["reward"] = round(row["imp"] / 100.0, 4)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = summarize_improvements(kept)
    stats_payload = {
        "input_files": args.input_files,
        "output_file": str(output_jsonl),
        "stats": dict(stats),
        "improvement_summary": summary,
        "workload_p99_fallback": workload_bounds,
        "bucket_p99_bounds_count": len(bucket_bounds),
    }
    stats_path = output_dir / args.stats_file
    stats_path.write_text(json.dumps(stats_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(
        {
            "output_file": str(output_jsonl),
            "stats_file": str(stats_path),
            "kept": len(kept),
            "stats": dict(stats),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
