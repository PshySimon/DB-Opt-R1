#!/usr/bin/env python3
"""Audit old RL rollout trajectories against the current v12 DB-tool guards.

The script intentionally avoids recomputing the cost model for every rollout.
It scans the trajectory JSONL once, audits every set_knob call with the current
validator, then only replays and re-scores the originally suspicious subset.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback for stripped-down servers
    tqdm = None

from core.db.knob_space import parse_memory
from core.db.knob_validator import KnobValidator
from cost_model.model import CostModel


MEMORY_KNOBS = {
    "shared_buffers",
    "work_mem",
    "effective_cache_size",
    "maintenance_work_mem",
    "wal_buffers",
    "temp_buffers",
}


def progress(iterable=None, **kwargs):
    if tqdm is not None:
        return tqdm(iterable, **kwargs)

    class _NoProgress:
        def __init__(self, it=None, **_):
            self.it = it

        def __iter__(self):
            return iter(self.it)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def update(self, *_):
            return None

        def set_postfix(self, *_args, **_kwargs):
            return None

    return _NoProgress(iterable, **kwargs)


def parse_knobs_arg(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("set_knob.knobs JSON is not an object")
        return parsed
    raise TypeError(f"unexpected set_knob.knobs type: {type(raw).__name__}")


def memory_filter(knobs: dict[str, Any], hw: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    try:
        total_mem_gb = float(hw.get("total_memory_gb") or 0)
    except Exception:
        total_mem_gb = 0
    if total_mem_gb <= 0:
        return knobs, []

    max_kb = total_mem_gb * 0.8 * 1024 * 1024
    accepted: dict[str, Any] = {}
    rejected: list[dict[str, str]] = []
    for name, value in knobs.items():
        if name in MEMORY_KNOBS:
            try:
                if parse_memory(str(value)) > max_kb:
                    rejected.append(
                        {
                            "name": name,
                            "value": str(value),
                            "error": f"超过物理内存限制（{total_mem_gb:.1f}GB 的 80%）",
                        }
                    )
                    continue
            except (TypeError, ValueError):
                pass
        accepted[name] = value
    return accepted, rejected


def iter_json_array_selected(path: Path, wanted_indices: set[int]) -> dict[int, dict[str, Any]]:
    """Stream a top-level JSON array and return selected 0-based items."""
    if not wanted_indices:
        return {}

    decoder = json.JSONDecoder()
    selected: dict[int, dict[str, Any]] = {}
    buffer = ""
    item_idx = 0
    started = False
    done = False
    file_size = path.stat().st_size

    with path.open("r", encoding="utf-8") as f, progress(total=file_size, unit="B", unit_scale=True, desc="load scenarios") as bar:
        while not done:
            chunk = f.read(1024 * 1024)
            if chunk:
                buffer += chunk
                bar.update(len(chunk.encode("utf-8", errors="ignore")))

            while True:
                i = 0
                while i < len(buffer) and buffer[i].isspace():
                    i += 1

                if not started:
                    if i >= len(buffer):
                        break
                    if buffer[i] != "[":
                        raise ValueError(f"{path} is not a JSON array")
                    started = True
                    buffer = buffer[i + 1 :]
                    continue

                i = 0
                while i < len(buffer) and (buffer[i].isspace() or buffer[i] == ","):
                    i += 1
                if i >= len(buffer):
                    buffer = ""
                    break
                if buffer[i] == "]":
                    done = True
                    buffer = buffer[i + 1 :]
                    break

                try:
                    obj, end = decoder.raw_decode(buffer, i)
                except json.JSONDecodeError:
                    if not chunk:
                        raise
                    if i:
                        buffer = buffer[i:]
                    break

                if item_idx in wanted_indices:
                    selected[item_idx] = obj
                    if len(selected) == len(wanted_indices):
                        return selected
                item_idx += 1
                buffer = buffer[end:]

            if not chunk:
                break

    missing = sorted(wanted_indices - set(selected))
    if missing:
        raise IndexError(f"missing scenario indices: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    return selected


def scenario_record(raw: dict[str, Any]) -> dict[str, Any]:
    workload = raw.get("workload") or {}
    return {
        "hardware": raw.get("hardware") or {},
        "knobs": dict(raw.get("knobs") or {}),
        "workload": workload.get("type", "mixed"),
        "actual_tps": workload.get("tps_current", 0),
    }


def coverage_combo(model: CostModel, current: dict[str, Any], baseline: dict[str, Any], hw: dict[str, Any]):
    cur = model.check_input_coverage(current, hw)
    base = model.check_input_coverage(baseline, hw)
    hard = bool(cur.get("hard_ood") or base.get("hard_ood"))
    near = bool(cur.get("near_boundary") or base.get("near_boundary"))
    features = list(cur.get("features") or []) + list(base.get("features") or [])
    if hard:
        return "invalid", features
    if near:
        return "low", features
    if cur.get("confidence") == "unknown" or base.get("confidence") == "unknown":
        return "unknown", features
    return "high", features


def freeze_config(config: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(k), str(v)) for k, v in config.items()))


def predict_v12(
    model: CostModel,
    current: dict[str, Any],
    baseline: dict[str, Any],
    hw: dict[str, Any],
    baseline_cache: dict[int, float],
    current_cache: dict[tuple[int, tuple[tuple[str, str], ...]], tuple[float, str, list[str]]],
    scenario_idx: int,
) -> dict[str, Any]:
    if scenario_idx not in baseline_cache:
        baseline_cache[scenario_idx] = model.predict(baseline, hw)
    pred_baseline = baseline_cache[scenario_idx]

    current_key = (scenario_idx, freeze_config(current))
    if current_key not in current_cache:
        pred_current = model.predict(current, hw)
        confidence, features = coverage_combo(model, current, baseline, hw)
        current_cache[current_key] = (pred_current, confidence, features)
    pred_current, confidence, features = current_cache[current_key]

    raw_improvement = (pred_current - pred_baseline) / max(pred_baseline, 1) * 100
    improvement = min(200.0, max(0.0, raw_improvement))
    if confidence == "invalid":
        improvement = 0.0
    elif confidence == "low":
        improvement = min(improvement, 25.0)

    return {
        "improvement_pct": improvement,
        "raw_improvement_pct": raw_improvement,
        "predicted_tps": pred_current,
        "baseline_tps": pred_baseline,
        "confidence": confidence,
        "coverage_features": features,
    }


def scan_trajectories(args, validator: KnobValidator):
    stats = collections.Counter()
    set_names = collections.Counter()
    set_values = collections.Counter()
    failed_names = collections.Counter()
    failed_reasons = collections.Counter()
    ignored_names = collections.Counter()
    canonical_changed = collections.Counter()
    candidates = []
    wanted_scenarios: set[int] = set()

    file_size = args.trajectories.stat().st_size
    with args.trajectories.open("rb") as f, progress(total=file_size, unit="B", unit_scale=True, desc="scan rollouts") as bar:
        for line_no, raw_line in enumerate(f, 1):
            bar.update(len(raw_line))
            if args.limit and line_no > args.limit:
                break
            if not raw_line.strip():
                continue
            obj = json.loads(raw_line)
            stats["trajectories"] += 1

            orig_imp = float(obj.get("improvement_pct") or 0)
            if orig_imp > args.threshold:
                stats["orig_gt_threshold"] += 1
            if orig_imp >= args.cap_threshold:
                stats["orig_cap"] += 1

            tool_history = (obj.get("tracking") or {}).get("tool_history") or []
            saw_abnormal = False
            for event in tool_history:
                if event.get("tool") != "set_knob":
                    continue
                stats["set_calls"] += 1
                try:
                    raw_knobs = parse_knobs_arg((event.get("args") or {}).get("knobs", "{}"))
                except Exception as exc:
                    stats["set_parse_errors"] += 1
                    failed_reasons[f"parse_error: {type(exc).__name__}"] += 1
                    saw_abnormal = True
                    continue

                stats["set_entries"] += len(raw_knobs)
                for name, value in raw_knobs.items():
                    set_names[name] += 1
                    set_values[(name, str(value))] += 1

                accepted, failed, ignored = validator.validate(raw_knobs)
                stats["accepted_entries_validator_only"] += len(accepted)
                stats["failed_entries_validator_only"] += len(failed)
                stats["ignored_entries_validator_only"] += len(ignored)
                if failed or ignored:
                    saw_abnormal = True
                for item in failed:
                    failed_names[item.get("name")] += 1
                    failed_reasons[item.get("error")] += 1
                for item in ignored:
                    ignored_names[item.get("name")] += 1
                for name, value in accepted.items():
                    if str(raw_knobs.get(name)) != str(value):
                        canonical_changed[(name, str(raw_knobs.get(name)), str(value))] += 1

            if saw_abnormal:
                stats["trajectories_with_validator_abnormal"] += 1

            if orig_imp > args.threshold or orig_imp >= args.cap_threshold:
                if args.max_candidates and len(candidates) >= args.max_candidates:
                    continue
                scenario_idx = int(obj["orig_env_sample_idx"])
                wanted_scenarios.add(scenario_idx)
                candidates.append(
                    {
                        "line_no": line_no,
                        "train_row_idx": obj.get("train_row_idx"),
                        "orig_env_sample_idx": scenario_idx,
                        "rollout_idx": obj.get("rollout_idx"),
                        "orig_improvement_pct": orig_imp,
                        "orig_predicted_tps": obj.get("predicted_tps"),
                        "orig_baseline_tps": obj.get("baseline_tps"),
                        "termination_reason": obj.get("termination_reason"),
                        "tool_history": tool_history,
                    }
                )

    return {
        "stats": stats,
        "set_names": set_names,
        "set_values": set_values,
        "failed_names": failed_names,
        "failed_reasons": failed_reasons,
        "ignored_names": ignored_names,
        "canonical_changed": canonical_changed,
        "candidates": candidates,
        "wanted_scenarios": wanted_scenarios,
    }


def replay_candidate(candidate, scenario, validator, restart_knobs, model, baseline_cache, current_cache):
    hw = scenario["hardware"]
    workload = scenario["workload"]
    baseline = dict(scenario["knobs"])
    baseline["workload"] = workload
    current = dict(scenario["knobs"])
    current["workload"] = workload
    pending: dict[str, Any] = {}
    stats = collections.Counter()
    failed = []
    ignored = []
    memory_failed = []
    predictions = []

    for event in candidate["tool_history"]:
        tool = event.get("tool")
        if tool == "set_knob":
            try:
                raw_knobs = parse_knobs_arg((event.get("args") or {}).get("knobs", "{}"))
            except Exception as exc:
                failed.append({"name": "<parse>", "error": str(exc)})
                stats["set_parse_errors"] += 1
                continue

            accepted, validation_failed, validation_ignored = validator.validate(raw_knobs)
            accepted, mem_failed = memory_filter(accepted, hw)
            failed.extend(validation_failed)
            ignored.extend(validation_ignored)
            memory_failed.extend(mem_failed)
            stats["failed_entries"] += len(validation_failed) + len(mem_failed)
            stats["ignored_entries"] += len(validation_ignored)

            for name, value in accepted.items():
                if name in restart_knobs:
                    pending[name] = value
                else:
                    current[name] = value

        elif tool == "restart_pg":
            if pending:
                current.update(pending)
                pending.clear()
            stats["restart_calls"] += 1

        elif tool == "predict_performance":
            if pending:
                stats["predict_with_pending_restart"] += 1
            pred = predict_v12(
                model=model,
                current=current,
                baseline=baseline,
                hw=hw,
                baseline_cache=baseline_cache,
                current_cache=current_cache,
                scenario_idx=candidate["orig_env_sample_idx"],
            )
            pred["pending_restart"] = dict(pending)
            pred["knobs"] = {
                key: value for key, value in current.items() if key != "workload"
            }
            predictions.append(pred)

    best_prediction = None
    if predictions:
        best_prediction = max(
            predictions,
            key=lambda item: (
                float(item.get("predicted_tps") or 0.0),
                float(item.get("improvement_pct") or 0.0),
            ),
        )
    return {
        "stats": stats,
        "failed": failed,
        "ignored": ignored,
        "memory_failed": memory_failed,
        "best_prediction": best_prediction,
        "num_predictions": len(predictions),
    }


def pct(num, den):
    if not den:
        return 0.0
    return round(num * 100 / den, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--checkpoint", default="cost_model/checkpoints/v12_lgbm")
    parser.add_argument("--knob-space", default="configs/knob_space.yaml")
    parser.add_argument("--pg-catalog", default="configs/pg_settings_pg16_catalog.json")
    parser.add_argument("--threshold", type=float, default=50.0, help="Only re-score original trajectories above this improvement")
    parser.add_argument("--cap-threshold", type=float, default=200.0)
    parser.add_argument("--output", type=Path, default=Path("/tmp/v12_rollout_knob_audit.json"))
    parser.add_argument("--limit", type=int, default=0, help="Debug: scan only first N jsonl lines")
    parser.add_argument("--max-candidates", type=int, default=0, help="Debug: re-score at most N suspicious trajectories")
    args = parser.parse_args()

    validator = KnobValidator(str(args.knob_space), str(args.pg_catalog))
    restart_knobs = {name for name, spec in validator.catalog.items() if spec.get("context") == "postmaster"}

    scanned = scan_trajectories(args, validator)
    candidates = scanned["candidates"]
    wanted = scanned["wanted_scenarios"]

    print(f"\nselected suspicious trajectories: {len(candidates)}")
    print(f"unique baseline scenarios needed: {len(wanted)}")

    selected_raw = iter_json_array_selected(args.scenarios, wanted)
    scenarios = {idx: scenario_record(raw) for idx, raw in selected_raw.items()}

    print("loading v12 cost model...")
    model = CostModel.load(args.checkpoint)

    baseline_cache: dict[int, float] = {}
    current_cache: dict[tuple[int, tuple[tuple[str, str], ...]], tuple[float, str, list[str]]] = {}
    v12_stats = collections.Counter()
    confidence = collections.Counter()
    ood_features = collections.Counter()
    failed_names_subset = collections.Counter()
    failed_reasons_subset = collections.Counter()
    ignored_names_subset = collections.Counter()
    remaining_examples = []

    for candidate in progress(candidates, desc="re-score suspicious", unit="traj"):
        result = replay_candidate(
            candidate=candidate,
            scenario=scenarios[candidate["orig_env_sample_idx"]],
            validator=validator,
            restart_knobs=restart_knobs,
            model=model,
            baseline_cache=baseline_cache,
            current_cache=current_cache,
        )
        v12_stats.update(result["stats"])
        if result["failed"] or result["memory_failed"] or result["ignored"]:
            v12_stats["candidate_with_guard_abnormal"] += 1
        for item in result["failed"] + result["memory_failed"]:
            failed_names_subset[item.get("name")] += 1
            failed_reasons_subset[item.get("error")] += 1
        for item in result["ignored"]:
            ignored_names_subset[item.get("name")] += 1

        pred = result["best_prediction"]
        if pred is None:
            v12_stats["candidate_without_prediction"] += 1
            continue
        v12_stats["candidate_with_prediction"] += 1
        imp = float(pred["improvement_pct"])
        confidence[pred["confidence"]] += 1
        for feature in pred.get("coverage_features") or []:
            ood_features[feature] += 1
        if imp > args.threshold:
            v12_stats["v12_gt_threshold"] += 1
        if imp >= args.cap_threshold:
            v12_stats["v12_cap"] += 1
            if len(remaining_examples) < 50:
                remaining_examples.append(
                    {
                        "line_no": candidate["line_no"],
                        "train_row_idx": candidate["train_row_idx"],
                        "orig_env_sample_idx": candidate["orig_env_sample_idx"],
                        "rollout_idx": candidate["rollout_idx"],
                        "orig_improvement_pct": candidate["orig_improvement_pct"],
                        "v12_improvement_pct": round(imp, 2),
                        "v12_raw_improvement_pct": round(pred["raw_improvement_pct"], 2),
                        "v12_predicted_tps": round(pred["predicted_tps"], 1),
                        "v12_baseline_tps": round(pred["baseline_tps"], 1),
                        "confidence": pred["confidence"],
                        "best_knobs": pred.get("knobs") or {},
                    }
                )

    total = scanned["stats"]["trajectories"]
    orig_gt = scanned["stats"]["orig_gt_threshold"]
    orig_cap = scanned["stats"]["orig_cap"]
    cand_pred = v12_stats["candidate_with_prediction"]
    summary = {
        "inputs": {
            "trajectories": str(args.trajectories),
            "scenarios": str(args.scenarios),
            "checkpoint": str(args.checkpoint),
            "threshold": args.threshold,
            "cap_threshold": args.cap_threshold,
            "limit": args.limit,
            "max_candidates": args.max_candidates,
        },
        "full_scan": {
            **dict(scanned["stats"]),
            "orig_gt_threshold_rate_pct": pct(orig_gt, total),
            "orig_cap_rate_pct": pct(orig_cap, total),
            "top_set_names": scanned["set_names"].most_common(50),
            "top_failed_names_validator_only": scanned["failed_names"].most_common(50),
            "top_failed_reasons_validator_only": scanned["failed_reasons"].most_common(50),
            "top_ignored_names_validator_only": scanned["ignored_names"].most_common(50),
            "top_canonical_changed_validator_only": [
                {"name": k[0], "from": k[1], "to": k[2], "count": v}
                for k, v in scanned["canonical_changed"].most_common(50)
            ],
        },
        "v12_rescore_suspicious": {
            **dict(v12_stats),
            "candidate_count": len(candidates),
            "unique_baseline_scenarios": len(wanted),
            "baseline_predict_cache_size": len(baseline_cache),
            "current_predict_cache_size": len(current_cache),
            "v12_gt_threshold_rate_in_candidates_pct": pct(v12_stats["v12_gt_threshold"], cand_pred),
            "v12_cap_rate_in_candidates_pct": pct(v12_stats["v12_cap"], cand_pred),
            "confidence": dict(confidence),
            "top_ood_features": ood_features.most_common(50),
            "top_failed_names": failed_names_subset.most_common(50),
            "top_failed_reasons": failed_reasons_subset.most_common(50),
            "top_ignored_names": ignored_names_subset.most_common(50),
            "remaining_v12_cap_examples": remaining_examples,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== FULL SCAN ===")
    print(f"trajectories={total}")
    print(f"set_calls={scanned['stats']['set_calls']} set_entries={scanned['stats']['set_entries']}")
    print(
        f"orig >{args.threshold:g}%={orig_gt} ({pct(orig_gt, total)}%) "
        f"orig >={args.cap_threshold:g}%={orig_cap} ({pct(orig_cap, total)}%)"
    )
    print(
        "validator_only: "
        f"failed_entries={scanned['stats']['failed_entries_validator_only']} "
        f"ignored_entries={scanned['stats']['ignored_entries_validator_only']} "
        f"trajectories_abnormal={scanned['stats']['trajectories_with_validator_abnormal']}"
    )

    print("\n=== V12 RE-SCORE SUSPICIOUS ===")
    print(f"candidates={len(candidates)} with_prediction={cand_pred}")
    print(f"baseline_cache={len(baseline_cache)} current_cache={len(current_cache)}")
    print(
        f"v12 >{args.threshold:g}%={v12_stats['v12_gt_threshold']} "
        f"({pct(v12_stats['v12_gt_threshold'], cand_pred)}% of candidates with prediction)"
    )
    print(
        f"v12 >={args.cap_threshold:g}%={v12_stats['v12_cap']} "
        f"({pct(v12_stats['v12_cap'], cand_pred)}% of candidates with prediction)"
    )
    print(f"confidence={dict(confidence)}")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
