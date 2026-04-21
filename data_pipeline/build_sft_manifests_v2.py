import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


def parse_tool_calls(content: str):
    return re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content or "", re.S)


def valid_tool_call_count(row: dict) -> int:
    count = 0
    for msg in row.get("messages", []):
        if msg.get("role") == "assistant":
            count += len(parse_tool_calls(msg.get("content", "")))
    return count


def set_knob_count(row: dict) -> int:
    count = 0
    for msg in row.get("messages", []):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            count += content.count('"name":"set_knob"') + content.count('"name": "set_knob"')
    return count


def predict_improvements(row: dict) -> list[float]:
    values = []
    for msg in row.get("messages", []):
        if msg.get("role") != "tool":
            continue
        try:
            payload = json.loads(msg.get("content", ""))
        except Exception:
            continue
        if isinstance(payload, dict) and {"predicted_tps", "baseline_tps", "actual_tps", "improvement_pct"} <= payload.keys():
            values.append(float(payload.get("improvement_pct", 0) or 0))
    return values


def classify_shape(row: dict) -> str:
    values = predict_improvements(row)
    first_positive = next((idx for idx, value in enumerate(values) if value > 0), None)
    if first_positive is None:
        return "other_positive"
    if all(value <= 0 for value in values[:first_positive]):
        return "direct_success" if first_positive == 0 else "retry_success"
    return "other_positive"


def classify_gain_bucket(improvement_pct: float) -> str:
    if 0 < improvement_pct <= 1:
        return "0_1"
    if 1 < improvement_pct < 3:
        return "1_3"
    if 3 <= improvement_pct < 10:
        return "3_10"
    return "10_50"


def classify_depth_bucket(row: dict) -> str:
    predict_count = sum(
        msg.get("content", "").count('"name":"predict_performance"') +
        msg.get("content", "").count('"name": "predict_performance"')
        for msg in row.get("messages", [])
        if msg.get("role") == "assistant"
    )
    if predict_count >= 2 or valid_tool_call_count(row) >= 9 or set_knob_count(row) >= 2:
        return "tail"
    return "main"


def build_label_rows(rows: list[dict], scenarios: list[dict]) -> list[dict]:
    labels = []
    for row in rows:
        env = row["env_sample_idx"]
        scenario = scenarios[env]
        labels.append(
            {
                "env_sample_idx": env,
                "name": scenario["name"],
                "variant": scenario.get("variant", 0),
                "workload": scenario.get("workload", {}).get("type", "unknown"),
                "gain_bucket": classify_gain_bucket(float(row["improvement_pct"])),
                "depth_bucket": classify_depth_bucket(row),
                "shape": classify_shape(row),
            }
        )
    return labels


def _allocate_counts(total: int, groups: dict[tuple, list], rng: random.Random) -> dict[tuple, int]:
    if total > sum(len(items) for items in groups.values()):
        raise ValueError("requested sample size exceeds available population")
    total_available = sum(len(items) for items in groups.values())
    raw = {
        key: total * (len(items) / total_available)
        for key, items in groups.items()
    }
    base = {
        key: min(len(groups[key]), int(value))
        for key, value in raw.items()
    }
    assigned = sum(base.values())
    remainders = sorted(
        [
            (raw[key] - base[key], rng.random(), key)
            for key in groups.keys()
            if len(groups[key]) > base[key]
        ],
        reverse=True,
    )
    idx = 0
    while assigned < total and idx < len(remainders):
        _, _, key = remainders[idx]
        if base[key] < len(groups[key]):
            base[key] += 1
            assigned += 1
        idx += 1
        if idx == len(remainders) and assigned < total:
            remainders = sorted(
                [
                    (raw[key] - base[key], rng.random(), key)
                    for key in groups.keys()
                    if len(groups[key]) > base[key]
                ],
                reverse=True,
            )
            idx = 0
    return base


def sample_stratified(items: list[dict], total: int, strata_keys: tuple[str, ...], seed: int) -> list[dict]:
    rng = random.Random(seed)
    groups = defaultdict(list)
    for item in items:
        key = tuple(item[k] for k in strata_keys)
        groups[key].append(item)
    for values in groups.values():
        rng.shuffle(values)
    allocations = _allocate_counts(total, groups, rng)
    sampled = []
    for key in sorted(groups):
        sampled.extend(groups[key][: allocations.get(key, 0)])
    rng.shuffle(sampled)
    return sampled


def build_manifest_sets(labels: list[dict], abc_size: int = 3000, seed: int = 20260421) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    direct = [item for item in labels if item["shape"] == "direct_success"]
    retry = [item for item in labels if item["shape"] == "retry_success"]
    main = [item for item in labels if item["depth_bucket"] == "main"]
    tail = [item for item in labels if item["depth_bucket"] == "tail"]

    manifests = {}

    manifests["sft_manifest_a0_direct_only.jsonl"] = sample_stratified(
        direct, abc_size, ("workload", "gain_bucket", "depth_bucket"), seed + 1
    )

    if len(retry) > abc_size:
        raise ValueError("retry pool exceeds A1 size")
    a1_direct = sample_stratified(
        direct,
        abc_size - len(retry),
        ("workload", "gain_bucket", "depth_bucket"),
        seed + 2,
    )
    a1_full = retry[:] + a1_direct
    rng.shuffle(a1_full)
    manifests["sft_manifest_a1_full3k.jsonl"] = a1_full

    manifests["sft_manifest_b1_depth_trimmed.jsonl"] = sample_stratified(
        main, abc_size, ("workload", "gain_bucket"), seed + 3
    )

    if len(tail) > abc_size:
        raise ValueError("tail pool exceeds B2 size")
    b2_main = sample_stratified(
        main,
        abc_size - len(tail),
        ("workload", "gain_bucket"),
        seed + 4,
    )
    b2_full = tail[:] + b2_main
    rng.shuffle(b2_full)
    manifests["sft_manifest_b2_depth_full.jsonl"] = b2_full

    manifests["sft_manifest_c1_gain_natural.jsonl"] = sample_stratified(
        labels, abc_size, ("workload", "gain_bucket"), seed + 5
    )

    gain_groups = defaultdict(list)
    for item in labels:
        gain_groups[item["gain_bucket"]].append(item)
    per_bucket = abc_size // 4
    c2 = []
    for bucket in ["0_1", "1_3", "3_10", "10_50"]:
        c2.extend(
            sample_stratified(
                gain_groups[bucket],
                per_bucket,
                ("workload", "depth_bucket"),
                seed + 10 + per_bucket + len(c2),
            )
        )
    rng.shuffle(c2)
    manifests["sft_manifest_c2_gain_balanced.jsonl"] = c2

    return manifests


def write_manifests(manifests: dict[str, list[dict]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for file_name, rows in manifests.items():
        path = output_dir / file_name
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        stats[file_name] = {
            "rows": len(rows),
            "workload": dict(sorted(Counter(r.get("workload", "unknown") for r in rows).items())),
            "gain_bucket": dict(sorted(Counter(r.get("gain_bucket", "unknown") for r in rows).items())),
            "depth_bucket": dict(sorted(Counter(r.get("depth_bucket", "unknown") for r in rows).items())),
            "shape": dict(sorted(Counter(r.get("shape", "unknown") for r in rows).items())),
        }
    (output_dir / "manifest_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Build SFT manifests for v2 train-only data.")
    parser.add_argument(
        "--input",
        default="data_pipeline/data/train/v2/sft_trajectories_v10_train.jsonl",
    )
    parser.add_argument(
        "--scenarios",
        default="data_pipeline/data/scenarios/collected/collected_8c16g_hdd_20k.json",
    )
    parser.add_argument(
        "--output-dir",
        default="data_pipeline/data/train/v2/manifests",
    )
    parser.add_argument("--abc-size", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260421)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.input).open("r", encoding="utf-8")
        if line.strip()
    ]
    scenarios = json.loads(Path(args.scenarios).read_text(encoding="utf-8"))
    labels = build_label_rows(rows, scenarios)
    manifests = build_manifest_sets(labels, abc_size=args.abc_size, seed=args.seed)
    write_manifests(manifests, Path(args.output_dir))
    print(
        json.dumps(
            {name: len(items) for name, items in manifests.items()},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
