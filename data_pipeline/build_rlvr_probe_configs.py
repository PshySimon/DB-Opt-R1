"""Build targeted RLVR cost-model probe knob configs.

The generator uses high-improvement RLVR rollout trajectories as the mother
distribution, then adds local perturbations and counterfactual controls around
the dominant synchronous_commit / WAL / buffer region. The output is a
``knob_configs_*.json`` file consumable by
``data_pipeline.synthesis.scenarios.pipeline collect``.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import ijson

from core.db.knob_validator import KnobValidator
from training.reward_score import extract_best_predict_from_messages


DEFAULT_TRAJECTORIES = (
    "eval_results/rlvr_filter/"
    "rtx3k_filter_train_full_x24_4gpu_20260506_161329/"
    "all_sampled_trajectories.jsonl"
)
DEFAULT_SCENARIOS = "data_pipeline/data/scenarios/collected/collected_8c16g_hdd_20k.json"
DEFAULT_OUTPUT = "data_pipeline/data/scenarios/knobs/knob_configs_rlvr_sync_off_probe_v1.json"

SYNC_VALUES = ["off", "on", "local", "remote_write"]
AXES = {
    "wal_buffers": ["8MB", "16MB", "32MB", "64MB", "128MB"],
    "shared_buffers": ["2GB", "4GB", "6GB", "8GB"],
    "effective_io_concurrency": ["1", "2", "4", "8"],
    "work_mem": ["4MB", "8MB", "16MB", "32MB"],
    "max_wal_size": ["1GB", "2GB", "4GB"],
    "checkpoint_timeout": ["5min", "10min", "15min"],
    "max_connections": ["50", "100", "200"],
    "commit_delay": ["0", "1000", "10000"],
}
WORKLOAD_WEIGHTS = [("write_heavy", 47), ("mixed", 36), ("high_concurrency", 17)]


def _weighted_choice(rng: random.Random, weighted: list[tuple[Any, int | float]]) -> Any:
    total = sum(weight for _, weight in weighted)
    point = rng.uniform(0, total)
    acc = 0.0
    for value, weight in weighted:
        acc += weight
        if point <= acc:
            return value
    return weighted[-1][0]


def _parse_set_knobs(args: dict) -> dict:
    raw = args.get("knobs") if isinstance(args, dict) else None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _normalise_knobs(knobs: dict, validator: KnobValidator) -> dict:
    accepted, _, _ = validator.validate(knobs)
    return dict(sorted((str(k), str(v)) for k, v in accepted.items()))


def _load_cap_records(path: Path, threshold: float, validator: KnobValidator) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            improvement = float(row.get("improvement_pct") or 0.0)
            if improvement < threshold:
                continue

            best_predict = extract_best_predict_from_messages(row.get("messages") or [])
            best_knobs = _normalise_knobs((best_predict or {}).get("knobs") or {}, validator)
            if not best_knobs:
                continue
            best_payload = (best_predict or {}).get("payload") or {}

            records.append(
                {
                    "line_no": line_no,
                    "train_row_idx": row.get("train_row_idx"),
                    "orig_env_sample_idx": row.get("orig_env_sample_idx"),
                    "rollout_idx": row.get("rollout_idx"),
                    "improvement_pct": improvement,
                    "predicted_tps": best_payload.get("predicted_tps", row.get("predicted_tps")),
                    "baseline_tps": best_payload.get("baseline_tps", row.get("baseline_tps")),
                    "actual_tps": best_payload.get("actual_tps", row.get("actual_tps")),
                    "knobs": best_knobs,
                }
            )
    return records


def _load_workloads(scenarios_path: Path, env_ids: set[int]) -> dict[int, str]:
    workloads: dict[int, str] = {}
    if not env_ids:
        return workloads
    max_env = max(env_ids)
    with scenarios_path.open("rb") as f:
        for idx, item in enumerate(ijson.items(f, "item")):
            if idx in env_ids:
                workloads[idx] = (item.get("workload") or {}).get("type", "mixed")
            if idx >= max_env and len(workloads) == len(env_ids):
                break
    return workloads


def _config_key(knobs: dict, workload: str) -> str:
    return json.dumps({"knobs": knobs, "workload": workload}, sort_keys=True, ensure_ascii=False)


def _make_item(index: int, kind: str, source: dict, knobs: dict, workload: str) -> dict:
    env = source.get("orig_env_sample_idx")
    base = source.get("baseline_tps")
    pred = source.get("predicted_tps")
    return {
        "name": f"rlvr_sync_probe_{index:04d}",
        "variant": 0,
        "source": "rlvr_sync_off_probe_v1",
        "difficulty": 1,
        "category": "rlvr_cost_model_sync_off_probe",
        "description": (
            f"Targeted RLVR sync-off/WAL probe; kind={kind}; "
            f"orig_env_sample_idx={env}; baseline_tps={base}; predicted_tps={pred}"
        ),
        "workload": workload,
        "knobs": knobs,
        "metadata": {
            "kind": kind,
            "source_line_no": source.get("line_no"),
            "train_row_idx": source.get("train_row_idx"),
            "orig_env_sample_idx": env,
            "rollout_idx": source.get("rollout_idx"),
            "source_improvement_pct": source.get("improvement_pct"),
            "source_predicted_tps": pred,
            "source_baseline_tps": base,
            "source_actual_tps": source.get("actual_tps"),
        },
    }


def _build_source_pool(records: list[dict], workloads_by_env: dict[int, str]) -> list[dict]:
    grouped: dict[tuple[str, str], dict] = {}
    for record in records:
        workload = workloads_by_env.get(record.get("orig_env_sample_idx"), "mixed")
        if workload == "read_only":
            continue
        key = (_config_key(record["knobs"], workload), workload)
        entry = grouped.get(key)
        if entry is None:
            entry = dict(record)
            entry["workload"] = workload
            entry["freq"] = 0
            grouped[key] = entry
        entry["freq"] += 1
    return sorted(grouped.values(), key=lambda item: (-item["freq"], str(item["knobs"])))


def _add_unique(
    items: list[dict],
    seen: set[str],
    kind: str,
    source: dict,
    knobs: dict,
    workload: str,
    limit: int,
) -> bool:
    if len(items) >= limit:
        return False
    key = _config_key(knobs, workload)
    if key in seen:
        return False
    seen.add(key)
    items.append(_make_item(len(items), kind, source, knobs, workload))
    return True


def _mutate_neighbor(source: dict, rng: random.Random, validator: KnobValidator) -> dict:
    knobs = dict(source["knobs"])

    knobs["synchronous_commit"] = _weighted_choice(
        rng,
        [("off", 70), ("on", 12), ("local", 12), ("remote_write", 6)],
    )

    mutation_count = rng.choice([1, 2, 2, 3])
    axis_names = list(AXES)
    rng.shuffle(axis_names)
    for axis in axis_names[:mutation_count]:
        knobs[axis] = rng.choice(AXES[axis])

    # Keep the main WAL/buffer axes populated most of the time, matching rollout access.
    if "wal_buffers" not in knobs or rng.random() < 0.55:
        knobs["wal_buffers"] = _weighted_choice(
            rng,
            [("32MB", 38), ("64MB", 42), ("16MB", 10), ("128MB", 8), ("8MB", 2)],
        )
    if "shared_buffers" not in knobs or rng.random() < 0.45:
        knobs["shared_buffers"] = _weighted_choice(
            rng,
            [("4GB", 52), ("6GB", 36), ("8GB", 8), ("2GB", 4)],
        )

    return _normalise_knobs(knobs, validator)


def build_probe_configs(
    trajectories: Path,
    scenarios: Path,
    output: Path,
    count: int,
    seed: int,
    threshold: float,
    knob_space: str,
    pg_catalog: str,
) -> dict:
    rng = random.Random(seed)
    validator = KnobValidator(knob_space, pg_catalog)

    cap_records = _load_cap_records(trajectories, threshold, validator)
    env_ids = {int(item["orig_env_sample_idx"]) for item in cap_records if item.get("orig_env_sample_idx") is not None}
    workloads_by_env = _load_workloads(scenarios, env_ids)
    source_pool = _build_source_pool(cap_records, workloads_by_env)
    if not source_pool:
        raise RuntimeError("no valid cap records found")

    items: list[dict] = []
    seen: set[str] = set()
    targets = {
        "replay_top": int(count * 0.25),
        "replay_diverse": int(count * 0.20),
        "counterfactual": int(count * 0.20),
    }

    for source in source_pool:
        if sum(1 for item in items if item["metadata"]["kind"] == "replay_top") >= targets["replay_top"]:
            break
        _add_unique(items, seen, "replay_top", source, source["knobs"], source["workload"], count)

    by_keyset: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for source in source_pool[targets["replay_top"]:]:
        by_keyset[tuple(sorted(source["knobs"]))].append(source)
    keysets = list(by_keyset)
    replay_diverse_added = 0
    while replay_diverse_added < targets["replay_diverse"] and keysets:
        rng.shuffle(keysets)
        progressed = False
        for keyset in list(keysets):
            bucket = by_keyset[keyset]
            if not bucket:
                keysets.remove(keyset)
                continue
            source = bucket.pop(0)
            if _add_unique(items, seen, "replay_diverse", source, source["knobs"], source["workload"], count):
                replay_diverse_added += 1
                progressed = True
                if replay_diverse_added >= targets["replay_diverse"]:
                    break
        if not progressed:
            break

    counterfactual_added = 0
    for source in source_pool:
        if counterfactual_added >= targets["counterfactual"]:
            break
        for sync_value in SYNC_VALUES:
            if counterfactual_added >= targets["counterfactual"]:
                break
            knobs = dict(source["knobs"])
            knobs["synchronous_commit"] = sync_value
            knobs = _normalise_knobs(knobs, validator)
            if _add_unique(items, seen, "counterfactual_sync", source, knobs, source["workload"], count):
                counterfactual_added += 1

    attempts = 0
    while len(items) < count and attempts < count * 200:
        attempts += 1
        source = rng.choice(source_pool)
        workload = source["workload"]
        if rng.random() < 0.25:
            workload = _weighted_choice(rng, WORKLOAD_WEIGHTS)
        knobs = _mutate_neighbor(source, rng, validator)
        _add_unique(items, seen, "local_neighbor", source, knobs, workload, count)

    if len(items) < count:
        raise RuntimeError(f"only generated {len(items)} unique configs, requested {count}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "output": str(output),
        "count": len(items),
        "cap_records": len(cap_records),
        "source_pool": len(source_pool),
        "kind_counts": Counter(item["metadata"]["kind"] for item in items),
        "workload_counts": Counter(item["workload"] for item in items),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", default=DEFAULT_TRAJECTORIES)
    parser.add_argument("--scenarios", default=DEFAULT_SCENARIOS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--threshold", type=float, default=200.0)
    parser.add_argument("--knob-space", default="configs/knob_space.yaml")
    parser.add_argument("--pg-catalog", default="configs/pg_settings_pg16_catalog.json")
    args = parser.parse_args()

    summary = build_probe_configs(
        trajectories=Path(args.trajectories),
        scenarios=Path(args.scenarios),
        output=Path(args.output),
        count=args.count,
        seed=args.seed,
        threshold=args.threshold,
        knob_space=args.knob_space,
        pg_catalog=args.pg_catalog,
    )
    print("=== RLVR PROBE CONFIGS ===")
    for key, value in summary.items():
        if isinstance(value, Counter):
            value = dict(value.most_common())
        print(f"{key} = {value}")


if __name__ == "__main__":
    main()
