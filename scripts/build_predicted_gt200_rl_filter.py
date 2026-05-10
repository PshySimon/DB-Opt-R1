#!/usr/bin/env python3
"""Build RL datasets excluding envs with v14-predicted >=200% rollout gains."""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.db.knob_validator import KnobValidator
from cost_model.model import CostModel


def load_audit_helpers():
    helper_path = Path(__file__).with_name("audit_v12_rollout_anomalies.py")
    spec = importlib.util.spec_from_file_location("audit_v12_rollout_anomalies", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_audit = load_audit_helpers()
iter_json_array_selected = _audit.iter_json_array_selected
replay_candidate = _audit.replay_candidate
scenario_record = _audit.scenario_record

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


DEFAULT_RUN_DIR = Path("eval_results/rlvr_filter/rtx3k_filter_train_full_x24_4gpu_20260506_161329")
DEFAULT_TRAJECTORIES = DEFAULT_RUN_DIR / "all_sampled_trajectories.jsonl"
DEFAULT_OVERLAP_INDICES = DEFAULT_RUN_DIR / "v14_cap200_overlap_indices.json"
DEFAULT_SCENARIOS = Path("data_pipeline/data/scenarios/collected/collected_8c16g_hdd_20k.json")
DEFAULT_RL_DIR = Path("data_pipeline/data/train/v2/rl")
DEFAULT_OUTPUT_FILTER_DIR = DEFAULT_RL_DIR / "filters" / "pred_v14_ge200"


RL_DATASETS = {
    "frontier_1q": "rl_frontier_1q.jsonl",
    "hard_1q": "rl_hard_1q.jsonl",
    "frontier_plus_hard_1q": "rl_frontier_plus_hard_1q.jsonl",
}


def progress(iterable=None, **kwargs):
    if tqdm is not None:
        return tqdm(iterable, **kwargs)
    return iterable


def iter_trajectories(path: Path):
    with path.open("rb") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            yield line_no, json.loads(raw_line)


def load_known_new_cap_indices(path: Path | None) -> set[int]:
    if path is None or not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(idx) for idx in ((payload.get("samples") or {}).get("new_only_cap") or [])}


def build_predicted_bad_envs(
    trajectories: Path,
    scenarios_path: Path,
    checkpoint: str,
    knob_space: str,
    pg_catalog: str,
    threshold: float,
    overlap_indices_json: Path | None,
) -> tuple[set[int], dict[str, Any]]:
    known_new_cap_zero_indices = load_known_new_cap_indices(overlap_indices_json)
    old_cap_candidates: list[tuple[int, dict[str, Any]]] = []
    known_new_cap_rows: list[tuple[int, dict[str, Any]]] = []
    wanted_envs: set[int] = set()

    for line_no, row in progress(iter_trajectories(trajectories), desc="scan candidates", unit="traj"):
        zero_idx = line_no - 1
        env = row.get("orig_env_sample_idx")
        if not isinstance(env, int):
            continue

        old_imp = float(row.get("improvement_pct") or 0.0)
        if zero_idx in known_new_cap_zero_indices:
            known_new_cap_rows.append((line_no, row))
        if old_imp >= threshold:
            old_cap_candidates.append((line_no, row))
            wanted_envs.add(env)

    selected_raw = iter_json_array_selected(scenarios_path, wanted_envs)
    scenarios = {idx: scenario_record(raw) for idx, raw in selected_raw.items()}

    validator = KnobValidator(knob_space, pg_catalog)
    restart_knobs = {name for name, spec in validator.catalog.items() if spec.get("context") == "postmaster"}
    model = CostModel.load(checkpoint)

    baseline_cache: dict[int, float] = {}
    current_cache: dict[tuple[int, tuple[tuple[str, str], ...]], tuple[float, str, list[str]]] = {}
    bad_envs: set[int] = set()
    bad_train_rows: set[int] = set()
    stats = collections.Counter()
    confidence = collections.Counter()
    termination = collections.Counter()
    examples: list[dict[str, Any]] = []

    for line_no, row in progress(old_cap_candidates, desc="replay old-cap rollouts", unit="traj"):
        stats["rollouts"] += 1
        env = row.get("orig_env_sample_idx")
        if not isinstance(env, int) or env not in scenarios:
            stats["missing_env"] += 1
            continue

        candidate = {
            "line_no": line_no,
            "train_row_idx": row.get("train_row_idx"),
            "orig_env_sample_idx": env,
            "rollout_idx": row.get("rollout_idx"),
            "orig_improvement_pct": row.get("improvement_pct"),
            "termination_reason": row.get("termination_reason"),
            "tool_history": (row.get("tracking") or {}).get("tool_history") or [],
        }
        termination[str(row.get("termination_reason"))] += 1

        result = replay_candidate(
            candidate=candidate,
            scenario=scenarios[env],
            validator=validator,
            restart_knobs=restart_knobs,
            model=model,
            baseline_cache=baseline_cache,
            current_cache=current_cache,
        )
        pred = result.get("best_prediction")
        if pred is None:
            stats["without_prediction"] += 1
            continue

        stats["with_prediction"] += 1
        confidence[str(pred.get("confidence"))] += 1
        imp = float(pred.get("improvement_pct") or 0.0)
        if imp >= threshold:
            stats["predicted_ge_threshold"] += 1
            bad_envs.add(env)
            train_row_idx = row.get("train_row_idx")
            if isinstance(train_row_idx, int):
                bad_train_rows.add(train_row_idx)
            if len(examples) < 50:
                examples.append(
                    {
                        "line_no": line_no,
                        "train_row_idx": train_row_idx,
                        "orig_env_sample_idx": env,
                        "rollout_idx": row.get("rollout_idx"),
                        "predicted_improvement_pct": round(imp, 2),
                        "predicted_tps": round(float(pred.get("predicted_tps") or 0.0), 1),
                        "baseline_tps": round(float(pred.get("baseline_tps") or 0.0), 1),
                        "confidence": pred.get("confidence"),
                        "best_knobs": pred.get("knobs") or {},
                    }
                )

    for _line_no, row in known_new_cap_rows:
        stats["known_new_cap_from_overlap"] += 1
        env = row.get("orig_env_sample_idx")
        if isinstance(env, int):
            bad_envs.add(env)
        train_row_idx = row.get("train_row_idx")
        if isinstance(train_row_idx, int):
            bad_train_rows.add(train_row_idx)

    summary = {
        "criterion": f"remove rows whose env_sample_idx appears in rollout best-predict improvement >= {threshold:g}% under checkpoint {checkpoint}",
        "inputs": {
            "trajectories": str(trajectories),
            "scenarios": str(scenarios_path),
            "checkpoint": checkpoint,
            "knob_space": knob_space,
            "pg_catalog": pg_catalog,
        },
        "rollout_stats": dict(stats),
        "candidate_stats": {
            "old_logged_ge_threshold_candidates": len(old_cap_candidates),
            "known_new_cap_from_overlap": len(known_new_cap_rows),
            "overlap_indices_json": str(overlap_indices_json) if overlap_indices_json else None,
        },
        "termination": dict(termination),
        "confidence": dict(confidence),
        "bad_env_count": len(bad_envs),
        "bad_train_row_count": len(bad_train_rows),
        "baseline_predict_cache_size": len(baseline_cache),
        "current_predict_cache_size": len(current_cache),
        "examples": examples,
    }
    return bad_envs, summary


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_indices(path: Path, indices: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{idx}\n" for idx in indices), encoding="utf-8")


def filter_rl_datasets(rl_dir: Path, output_filter_dir: Path, bad_envs: set[int], summary: dict[str, Any]) -> dict[str, Any]:
    versions: dict[str, Any] = {}
    for version, filename in RL_DATASETS.items():
        src = rl_dir / filename
        records = load_jsonl(src)
        bad_indices: list[int] = []
        clean_indices: list[int] = []
        clean_records: list[dict[str, Any]] = []

        for idx, record in enumerate(records):
            env = record.get("env_sample_idx")
            if isinstance(env, int) and env in bad_envs:
                bad_indices.append(idx)
            else:
                clean_indices.append(idx)
                clean_records.append(record)

        out_name = f"rl_{version}_no_pred_v14_ge200.jsonl"
        out_path = rl_dir / out_name
        env_ids_path = rl_dir / f"rl_{version}_no_pred_v14_ge200_env_ids.json"
        write_jsonl(out_path, clean_records)
        env_ids_path.write_text(
            json.dumps([item["env_sample_idx"] for item in clean_records], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        bad_path = output_filter_dir / f"{version}_bad_row_indices.txt"
        clean_path = output_filter_dir / f"{version}_clean_row_indices.txt"
        write_indices(bad_path, bad_indices)
        write_indices(clean_path, clean_indices)

        versions[version] = {
            "path": str(src),
            "output_path": str(out_path),
            "env_ids_path": str(env_ids_path),
            "rows": len(records),
            "remove_rows": len(bad_indices),
            "remain_rows": len(clean_records),
            "removed_rate": (len(bad_indices) / len(records)) if records else 0.0,
            "bad_indices_path": str(bad_path),
            "clean_indices_path": str(clean_path),
        }

    summary["versions"] = versions
    output_filter_dir.mkdir(parents=True, exist_ok=True)
    (output_filter_dir / "bad_env_ids.txt").write_text(
        "".join(f"{env}\n" for env in sorted(bad_envs)),
        encoding="utf-8",
    )
    (output_filter_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, default=DEFAULT_TRAJECTORIES)
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--checkpoint", default="cost_model/checkpoints/v14_lgbm")
    parser.add_argument("--knob-space", default="configs/knob_space.yaml")
    parser.add_argument("--pg-catalog", default="configs/pg_settings_pg16_catalog.json")
    parser.add_argument("--rl-dir", type=Path, default=DEFAULT_RL_DIR)
    parser.add_argument("--output-filter-dir", type=Path, default=DEFAULT_OUTPUT_FILTER_DIR)
    parser.add_argument("--threshold", type=float, default=200.0)
    parser.add_argument("--overlap-indices-json", type=Path, default=DEFAULT_OVERLAP_INDICES)
    args = parser.parse_args()

    bad_envs, summary = build_predicted_bad_envs(
        trajectories=args.trajectories,
        scenarios_path=args.scenarios,
        checkpoint=args.checkpoint,
        knob_space=args.knob_space,
        pg_catalog=args.pg_catalog,
        threshold=args.threshold,
        overlap_indices_json=args.overlap_indices_json,
    )
    summary = filter_rl_datasets(
        rl_dir=args.rl_dir,
        output_filter_dir=args.output_filter_dir,
        bad_envs=bad_envs,
        summary=summary,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
