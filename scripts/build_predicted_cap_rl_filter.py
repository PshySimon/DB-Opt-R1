#!/usr/bin/env python3
"""Build RL train filters from cost-model re-scoring of rollout best-predict knobs.

The script replays the rollout trajectory only up to the best
``predict_performance`` call, batch-predicts those best-knob snapshots with a
specified cost model, then removes train rows whose sampled rollouts can still
hit a capped improvement threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    import ijson
except Exception:  # pragma: no cover
    ijson = None

from core.db.knob_validator import KnobValidator
from cost_model.model import CostModel


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

        def __exit__(self, *_):
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
        return parsed if isinstance(parsed, dict) else {}
    return {}


def parse_tool_result(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    return {}


def is_predict_payload(payload: dict[str, Any]) -> bool:
    return {"predicted_tps", "baseline_tps", "improvement_pct"} <= payload.keys()


def best_predict_from_tool_history(tool_history: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    current_knobs: dict[str, Any] = {}
    best: tuple[dict[str, Any], dict[str, Any]] | None = None
    best_key: tuple[float, float] | None = None

    for event in tool_history:
        tool = event.get("tool")
        if tool == "set_knob":
            try:
                current_knobs.update(parse_knobs_arg((event.get("args") or {}).get("knobs", {})))
            except Exception:
                continue
            continue

        if tool != "predict_performance":
            continue

        try:
            payload = parse_tool_result(event.get("result"))
        except Exception:
            continue
        if not is_predict_payload(payload):
            continue

        key = (
            float(payload.get("predicted_tps") or 0.0),
            float(payload.get("improvement_pct") or 0.0),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = (dict(current_knobs), dict(payload))

    return best


def freeze_config(config: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(k), str(v)) for k, v in config.items()))


def config_key(config: dict[str, Any]) -> str:
    return json.dumps(dict(freeze_config(config)), ensure_ascii=False, sort_keys=True)


def load_rl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_selected_scenarios(path: Path, wanted: set[int]) -> dict[int, dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    if not wanted:
        return selected

    max_idx = max(wanted)
    if ijson is not None:
        with path.open("rb") as f, progress(total=max_idx + 1, desc="load scenarios", unit="env") as bar:
            for idx, item in enumerate(ijson.items(f, "item")):
                if idx in wanted:
                    selected[idx] = item
                bar.update(1)
                if idx >= max_idx and len(selected) == len(wanted):
                    break
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        for idx in progress(sorted(wanted), desc="load scenarios", unit="env"):
            selected[idx] = data[idx]

    missing = wanted - set(selected)
    if missing:
        sample = sorted(missing)[:10]
        raise IndexError(f"missing scenario indices: {sample}{'...' if len(missing) > 10 else ''}")
    return selected


def scenario_to_prediction_parts(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
    workload = (raw.get("workload") or {}).get("type", "mixed")
    baseline = dict(raw.get("knobs") or {})
    baseline["workload"] = workload
    return baseline, raw.get("hardware") or {}, workload


def compare_threshold(value: float, threshold: float, op: str) -> bool:
    if op == "ge":
        return value >= threshold
    if op == "gt":
        return value > threshold
    raise ValueError(f"unsupported threshold op: {op}")


def build_filtered_outputs(
    input_files: list[Path],
    output_suffix: str,
    output_dir: Path,
    bad_train_rows: set[int],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {}

    for src in input_files:
        rows = load_rl_rows(src)
        if src.stem == "rl_hard_1q":
            active_bad_rows: set[int] = set()
        else:
            active_bad_rows = bad_train_rows
        kept = []
        removed = 0
        for idx, row in enumerate(rows):
            if idx in active_bad_rows:
                removed += 1
            else:
                kept.append(row)

        dst = output_dir / f"{src.stem}_{output_suffix}{src.suffix}"
        with dst.open("w", encoding="utf-8") as f:
            for row in kept:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        result[src.stem] = {
            "src": str(src),
            "dst": str(dst),
            "total": len(rows),
            "removed": removed,
            "kept": len(kept),
        }

    return result


def write_indices(path: Path, values: set[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(i) for i in sorted(values)) + ("\n" if values else ""), encoding="utf-8")


def transform_prediction_items(model: CostModel, predict_items: list[tuple[str, int, str, dict[str, Any], dict[str, Any]]]) -> np.ndarray:
    """Vectorized version of CostModel.preprocessor.transform for mixed hw_info."""
    rows: list[dict[str, Any]] = []
    defaults = model.preprocessor.knob_defaults
    for _kind, _env_idx, _key, config, hw in progress(predict_items, desc="build feature rows", unit="cfg"):
        row: dict[str, Any] = {}
        for name, default in defaults.items():
            row[f"knob_{name}"] = config.get(name, default)
        row["workload"] = config.get("workload", "mixed")
        row["status"] = "success"
        for k, v in hw.items():
            row[f"hw_{k}"] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    features = model.preprocessor._build_features(df)
    for col in model.preprocessor.feature_names:
        if col not in features.columns:
            features[col] = 0
    features = features[model.preprocessor.feature_names]
    return features.values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--knob-space", type=Path, default=Path("configs/knob_space.yaml"))
    parser.add_argument("--pg-catalog", type=Path, default=Path("configs/pg_settings_pg16_catalog.json"))
    parser.add_argument("--threshold", type=float, default=200.0)
    parser.add_argument("--threshold-op", choices=["ge", "gt"], default="ge")
    parser.add_argument(
        "--knob-mode",
        choices=["accepted", "raw"],
        default="accepted",
        help="accepted: pass PG-valid canonical knobs; raw: pass raw best-predict knobs to the cost model.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--filtered-output-dir",
        type=Path,
        default=None,
        help="Directory for filtered RL JSONL files. Defaults to --output-dir.",
    )
    parser.add_argument("--output-suffix", default="no_pred_v14_ge200")
    parser.add_argument(
        "--input-rl",
        type=Path,
        action="append",
        default=[],
        help="RL JSONL to filter. Can be passed multiple times.",
    )
    parser.add_argument("--limit", type=int, default=0, help="debug: scan first N rollout rows")
    parser.add_argument("--write-records", action="store_true", help="Write per-trajectory prediction audit JSONL.")
    args = parser.parse_args()

    validator = KnobValidator(str(args.knob_space), str(args.pg_catalog))

    records: list[dict[str, Any]] = []
    env_to_configs: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    wanted_envs: set[int] = set()
    scan_stats = Counter()
    old_rows_by_threshold: set[int] = set()

    file_size = args.trajectories.stat().st_size
    with args.trajectories.open("rb") as f, progress(total=file_size, unit="B", unit_scale=True, desc="scan rollouts") as bar:
        for line_no, raw in enumerate(f, 1):
            bar.update(len(raw))
            if args.limit and line_no > args.limit:
                break
            if not raw.strip():
                continue

            row = json.loads(raw)
            scan_stats["records"] += 1
            train_row_idx = int(row["train_row_idx"])
            old_imp = float(row.get("improvement_pct") or 0.0)
            if compare_threshold(old_imp, args.threshold, args.threshold_op):
                old_rows_by_threshold.add(train_row_idx)
                scan_stats["old_threshold_trajectories"] += 1

            best = best_predict_from_tool_history((row.get("tracking") or {}).get("tool_history") or [])
            if best is None:
                scan_stats["no_best_predict"] += 1
                continue

            raw_knobs, old_payload = best
            accepted, failed, ignored = validator.validate(raw_knobs)
            scan_stats["validator_failed_entries"] += len(failed)
            scan_stats["validator_ignored_entries"] += len(ignored)
            selected_knobs = accepted if args.knob_mode == "accepted" else raw_knobs
            if not selected_knobs:
                scan_stats["no_best_knobs"] += 1
                continue

            env_idx = int(row["orig_env_sample_idx"])
            best_knobs = dict(sorted((str(k), str(v)) for k, v in selected_knobs.items()))
            key = config_key(best_knobs)
            env_to_configs[env_idx][key] = best_knobs
            wanted_envs.add(env_idx)
            records.append(
                {
                    "line_no": line_no,
                    "train_row_idx": train_row_idx,
                    "orig_env_sample_idx": env_idx,
                    "config_key": key,
                    "old_improvement_pct": old_imp,
                    "old_predicted_tps": old_payload.get("predicted_tps"),
                    "old_baseline_tps": old_payload.get("baseline_tps"),
                }
            )

    scenarios_raw = load_selected_scenarios(args.scenarios, wanted_envs)
    scenarios = {idx: scenario_to_prediction_parts(raw) for idx, raw in scenarios_raw.items()}

    print("loading cost model...")
    model = CostModel.load(str(args.checkpoint))

    baseline_predictions: dict[int, float] = {}
    current_predictions: dict[tuple[int, str], float] = {}
    workload_by_env: dict[int, str] = {}
    predict_items: list[tuple[str, int, str, dict[str, Any], dict[str, Any]]] = []

    for env_idx, keyed_configs in progress(env_to_configs.items(), desc="prepare predict configs", unit="env"):
        baseline, hw, workload = scenarios[env_idx]
        workload_by_env[env_idx] = workload
        predict_items.append(("baseline", env_idx, "__baseline__", baseline, hw))
        for key, knobs in keyed_configs.items():
            current = dict(baseline)
            current.update(knobs)
            predict_items.append(("current", env_idx, key, current, hw))

    unique_pairs = len(predict_items) - len(env_to_configs)
    print(f"transforming {len(predict_items)} unique prediction inputs...")
    X = transform_prediction_items(model, predict_items)
    print("running one cost-model batch predict...")
    mean, _ = model._infer(X)
    preds = np.expm1(mean)
    for (kind, env_idx, key, _config, _hw), pred in zip(predict_items, preds):
        if kind == "baseline":
            baseline_predictions[env_idx] = float(pred)
        else:
            current_predictions[(env_idx, key)] = float(pred)

    threshold_rows: set[int] = set()
    threshold_trajectories = 0
    gt25 = gt50 = ge200 = le0 = 0
    by_workload = Counter()
    output_records_path = args.output_dir / "predicted_rollout_records.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record_out = output_records_path.open("w", encoding="utf-8") if args.write_records else None
    try:
        for rec in progress(records, desc="score trajectories", unit="traj"):
            env_idx = int(rec["orig_env_sample_idx"])
            base = baseline_predictions[env_idx]
            pred = current_predictions[(env_idx, rec["config_key"])]
            raw_imp = (pred - base) / max(base, 1.0) * 100.0
            imp = min(200.0, max(0.0, raw_imp))

            if imp > 25:
                gt25 += 1
            if imp > 50:
                gt50 += 1
            if imp >= 200:
                ge200 += 1
            if imp <= 0:
                le0 += 1
            if compare_threshold(imp, args.threshold, args.threshold_op):
                threshold_trajectories += 1
                threshold_rows.add(int(rec["train_row_idx"]))
                by_workload[workload_by_env[env_idx]] += 1

            if record_out is not None:
                record_out.write(
                    json.dumps(
                        {
                            **rec,
                            "v_predicted_tps": pred,
                            "v_baseline_tps": base,
                            "v_raw_improvement_pct": raw_imp,
                            "v_improvement_pct": imp,
                            "workload": workload_by_env[env_idx],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    finally:
        if record_out is not None:
            record_out.close()

    clean_rows = set(range(max((r["train_row_idx"] for r in records), default=-1) + 1)) - threshold_rows
    write_indices(args.output_dir / f"bad_train_row_indices_pred_{args.threshold_op}{int(args.threshold)}.txt", threshold_rows)
    write_indices(args.output_dir / f"clean_train_row_indices_no_pred_{args.threshold_op}{int(args.threshold)}.txt", clean_rows)

    filtered_outputs = {}
    if args.input_rl:
        filtered_outputs = build_filtered_outputs(
            args.input_rl,
            args.output_suffix,
            args.filtered_output_dir or args.output_dir,
            threshold_rows,
        )

    summary = {
        "inputs": {
            "trajectories": str(args.trajectories),
            "scenarios": str(args.scenarios),
            "checkpoint": str(args.checkpoint),
            "threshold": args.threshold,
            "threshold_op": args.threshold_op,
        },
        "counts": {
            **dict(scan_stats),
            "records_with_best_knobs": len(records),
            "unique_envs": len(wanted_envs),
            "unique_current_env_knob_pairs": unique_pairs,
            "old_train_rows_at_threshold": len(old_rows_by_threshold),
            "pred_threshold_trajectories": threshold_trajectories,
            "pred_train_rows_at_threshold": len(threshold_rows),
            "pred_gt25_trajectories": gt25,
            "pred_gt50_trajectories": gt50,
            "pred_ge200_trajectories": ge200,
            "pred_le0_trajectories": le0,
            "by_workload_at_threshold": dict(by_workload),
        },
        "filtered_outputs": filtered_outputs,
        "artifacts": {
            "predicted_rollout_records": str(output_records_path) if args.write_records else None,
        },
    }

    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== PREDICTED CAP RL FILTER ===")
    print(f"records={scan_stats['records']}")
    print(f"no_best_predict={scan_stats['no_best_predict']} no_best_knobs={scan_stats['no_best_knobs']}")
    print(f"records_with_best_knobs={len(records)} unique_envs={len(wanted_envs)} unique_current_env_knob_pairs={unique_pairs}")
    print(f"old_train_rows_at_threshold={len(old_rows_by_threshold)}")
    print(f"pred_threshold_trajectories={threshold_trajectories}")
    print(f"pred_train_rows_at_threshold={len(threshold_rows)}")
    print(f"pred_gt25={gt25} pred_gt50={gt50} pred_ge200={ge200} pred_le0={le0}")
    print(f"by_workload_at_threshold={dict(by_workload)}")
    if filtered_outputs:
        print("filtered_outputs:")
        for name, item in filtered_outputs.items():
            print(f"  {name}: total={item['total']} removed={item['removed']} kept={item['kept']} -> {item['dst']}")
    print(f"summary={args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
