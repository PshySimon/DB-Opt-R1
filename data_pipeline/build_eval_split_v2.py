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


def predict_count(row: dict) -> int:
    count = 0
    for msg in row.get("messages", []):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            count += content.count('"name":"predict_performance"') + content.count('"name": "predict_performance"')
    return count


def gain_bucket(improvement_pct: float) -> str:
    if 0 < improvement_pct <= 1:
        return "0_1"
    if 1 < improvement_pct < 3:
        return "1_3"
    if 3 <= improvement_pct < 10:
        return "3_10"
    return "10_50"


def depth_bucket(row: dict) -> str:
    if predict_count(row) >= 2 or valid_tool_call_count(row) >= 9 or set_knob_count(row) >= 2:
        return "tail"
    return "main"


def build_group_entries(rows: list[dict], scenarios: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        env = row["env_sample_idx"]
        scenario = scenarios[env]
        name = scenario["name"]
        variant = scenario.get("variant", 0)
        workload = scenario.get("workload", {}).get("type", "unknown")
        grouped[(name, variant, workload)].append(
            {
                "env": env,
                "workload": workload,
                "gain_bucket": gain_bucket(float(row["improvement_pct"])),
                "depth_bucket": depth_bucket(row),
            }
        )

    groups = []
    for (name, variant, workload), items in sorted(grouped.items()):
        groups.append(
            {
                "key": (name, variant, workload),
                "name": name,
                "variant": variant,
                "workload": workload,
                "size": len(items),
                "env_ids": sorted(item["env"] for item in items),
                "workload_counts": Counter(item["workload"] for item in items),
                "gain_counts": Counter(item["gain_bucket"] for item in items),
                "depth_counts": Counter(item["depth_bucket"] for item in items),
            }
        )
    return groups


def _scaled_targets(groups: list[dict], target_rows: int) -> dict:
    total_rows = sum(group["size"] for group in groups)
    workload_counts = Counter()
    gain_counts = Counter()
    depth_counts = Counter()
    for group in groups:
        workload_counts.update(group["workload_counts"])
        gain_counts.update(group["gain_counts"])
        depth_counts.update(group["depth_counts"])

    def scale(counter: Counter) -> dict:
        return {
            key: target_rows * (value / total_rows)
            for key, value in counter.items()
        }

    return {
        "workload": scale(workload_counts),
        "gain": scale(gain_counts),
        "depth": scale(depth_counts),
    }


def _empty_counts() -> dict:
    return {
        "rows": 0,
        "workload": Counter(),
        "gain": Counter(),
        "depth": Counter(),
    }


def _add_group(counts: dict, group: dict) -> dict:
    updated = {
        "rows": counts["rows"] + group["size"],
        "workload": counts["workload"].copy(),
        "gain": counts["gain"].copy(),
        "depth": counts["depth"].copy(),
    }
    updated["workload"].update(group["workload_counts"])
    updated["gain"].update(group["gain_counts"])
    updated["depth"].update(group["depth_counts"])
    return updated


def _score_counts(counts: dict, targets: dict, target_rows: int, tolerance: int) -> float:
    size_diff = abs(counts["rows"] - target_rows)
    score = size_diff * 8.0
    if counts["rows"] < target_rows - tolerance:
        score += (target_rows - tolerance - counts["rows"]) * 20.0
    if counts["rows"] > target_rows + tolerance:
        score += (counts["rows"] - target_rows - tolerance) * 20.0

    for key, target in targets["workload"].items():
        score += abs(counts["workload"].get(key, 0) - target)
    for key, target in targets["gain"].items():
        score += abs(counts["gain"].get(key, 0) - target)
    for key, target in targets["depth"].items():
        score += abs(counts["depth"].get(key, 0) - target)
    return score


def select_eval_groups(
    groups: list[dict],
    target_rows: int = 600,
    seed: int = 20260421,
    tolerance: int = 20,
    trials: int = 256,
) -> list[dict]:
    targets = _scaled_targets(groups, target_rows)
    groups_by_name = defaultdict(list)
    for group in groups:
        groups_by_name[group["name"]].append(group)

    best_selected = None
    best_score = None
    name_keys = sorted(groups_by_name)

    for trial in range(trials):
        rng = random.Random(seed + trial)
        name_order = name_keys[:]
        rng.shuffle(name_order)

        selected_keys = set()
        selected_groups = []
        counts = _empty_counts()

        for name in name_order:
            candidates = groups_by_name[name]
            scored = []
            for group in candidates:
                projected = _add_group(counts, group)
                score = _score_counts(projected, targets, target_rows, tolerance)
                score += group["size"] * 0.05
                scored.append((score, group))
            scored.sort(key=lambda item: item[0])
            top_k = min(3, len(scored))
            _, chosen = scored[rng.randrange(top_k)]
            selected_keys.add(chosen["key"])
            selected_groups.append(chosen)
            counts = _add_group(counts, chosen)

        remaining = [group for group in groups if group["key"] not in selected_keys]
        while remaining:
            current_score = _score_counts(counts, targets, target_rows, tolerance)
            scored = []
            for group in remaining:
                projected = _add_group(counts, group)
                score = _score_counts(projected, targets, target_rows, tolerance)
                score += group["size"] * 0.05
                scored.append((score, group))
            scored.sort(key=lambda item: item[0])
            best_next_score, best_next_group = scored[0]

            must_continue = counts["rows"] < target_rows - tolerance
            if not must_continue and best_next_score >= current_score:
                break

            selected_keys.add(best_next_group["key"])
            selected_groups.append(best_next_group)
            counts = _add_group(counts, best_next_group)
            remaining = [group for group in remaining if group["key"] != best_next_group["key"]]

        final_score = _score_counts(counts, targets, target_rows, tolerance)
        if best_score is None or final_score < best_score:
            best_score = final_score
            best_selected = list(selected_groups)

    return best_selected or []


def summarize_selected_groups(selected_groups: list[dict], target_rows: int) -> dict:
    stats = {
        "target_rows": target_rows,
        "selected_rows": sum(group["size"] for group in selected_groups),
        "selected_groups": len(selected_groups),
        "covered_parent_names": len({group["name"] for group in selected_groups}),
        "workload": Counter(),
        "gain": Counter(),
        "depth": Counter(),
        "parent_name_counts": Counter(group["name"] for group in selected_groups),
    }
    for group in selected_groups:
        stats["workload"].update(group["workload_counts"])
        stats["gain"].update(group["gain_counts"])
        stats["depth"].update(group["depth_counts"])

    for key in ("workload", "gain", "depth", "parent_name_counts"):
        stats[key] = dict(sorted(stats[key].items()))
    return stats


def write_eval_outputs(
    selected_groups: list[dict],
    rows: list[dict],
    scenarios: list[dict],
    output_dir: Path,
    target_rows: int = 600,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_env = {row["env_sample_idx"]: row for row in rows}

    ordered_env_ids = []
    for group in selected_groups:
        ordered_env_ids.extend(group["env_ids"])

    selected_scenarios = []
    question_items = []
    index_map = []
    for new_idx, source_env in enumerate(ordered_env_ids):
        selected_scenarios.append(scenarios[source_env])
        row = rows_by_env[source_env]
        question = row.get("question") or row["messages"][1]["content"]
        question_items.append(
            {
                "messages": [
                    {"role": "system", "content": row["messages"][0]["content"]},
                    {"role": "user", "content": question},
                ],
                "question": question,
                "env_sample_idx": new_idx,
            }
        )

        scenario = scenarios[source_env]
        index_map.append(
            {
                "eval_env_sample_idx": new_idx,
                "source_env_sample_idx": source_env,
                "name": scenario["name"],
                "variant": scenario.get("variant", 0),
                "workload": scenario.get("workload", {}).get("type", "unknown"),
                "improvement_pct": float(row["improvement_pct"]),
            }
        )

    (output_dir / "collected_eval_v2.json").write_text(
        json.dumps(selected_scenarios, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "eval_trajectories_v2.jsonl").open("w", encoding="utf-8") as f:
        for item in question_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (output_dir / "eval_index_map.jsonl").open("w", encoding="utf-8") as f:
        for item in index_map:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    (output_dir / "eval_source_env_ids.json").write_text(
        json.dumps(ordered_env_ids, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    stats = summarize_selected_groups(selected_groups, target_rows)
    (output_dir / "eval_split_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Build v2 eval split from cleaned SFT raw.")
    parser.add_argument(
        "--input",
        default="data_pipeline/data/pool/v2/sft_trajectories_v10_cleaned_pool.jsonl",
    )
    parser.add_argument(
        "--scenarios",
        default="data_pipeline/data/scenarios/collected/collected_8c16g_hdd_20k.json",
    )
    parser.add_argument(
        "--output-dir",
        default="data_pipeline/data/eval/v2",
    )
    parser.add_argument("--target-size", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260421)
    parser.add_argument("--tolerance", type=int, default=20)
    parser.add_argument("--trials", type=int, default=256)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.input).open("r", encoding="utf-8")
        if line.strip()
    ]
    scenarios = json.loads(Path(args.scenarios).read_text(encoding="utf-8"))

    groups = build_group_entries(rows, scenarios)
    selected_groups = select_eval_groups(
        groups=groups,
        target_rows=args.target_size,
        seed=args.seed,
        tolerance=args.tolerance,
        trials=args.trials,
    )
    write_eval_outputs(
        selected_groups=selected_groups,
        rows=rows,
        scenarios=scenarios,
        output_dir=Path(args.output_dir),
        target_rows=args.target_size,
    )

    stats = summarize_selected_groups(selected_groups, args.target_size)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
