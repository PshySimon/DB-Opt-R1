import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from cost_model.model import CostModel
from data_pipeline.build_clean_sft_raw_v2 import (
    FAMILY_MAP,
    build_reference_bounds,
    classify_high_gain_outlier,
    extract_row,
    predict_default_baselines,
)
from training.data_utils import SYSTEM_PROMPT


DEFAULT_OUTPUT_DIR = "data_pipeline/data/train/v2/rl"


def has_valid_rl_prompt(record: dict) -> bool:
    messages = record.get("messages", [])
    question = str(record.get("question", "") or "").strip()
    env = record.get("env_sample_idx")
    return bool(question and messages and messages[0].get("role") == "system" and isinstance(env, int))


def classify_env_bucket(total_rollouts: int, strong_rollouts: int, weak_rollouts: int) -> str:
    strong_success_rate = strong_rollouts / total_rollouts if total_rollouts else 0.0
    weak_success_rate = weak_rollouts / total_rollouts if total_rollouts else 0.0

    if strong_success_rate > 0.8:
        return "easy"
    if strong_success_rate >= 0.1:
        return "frontier"
    if weak_success_rate > 0:
        return "hard_but_learnable"
    return "all_fail"


def choose_question(env_id: int, questions: set[str], question_env_frequency: dict[str, int], seed: int) -> str:
    if not questions:
        raise ValueError(f"env {env_id} 没有可用 question")

    candidates = sorted(
        questions,
        key=lambda q: (question_env_frequency.get(q, 0), len(q), q),
    )
    best_freq = question_env_frequency.get(candidates[0], 0)
    ties = [q for q in candidates if question_env_frequency.get(q, 0) == best_freq]
    if len(ties) == 1:
        return ties[0]

    rng = random.Random(seed + env_id)
    return rng.choice(ties)


def collect_outlier_envs(rows: list[dict], default_baselines: dict[int, float], bucket_bounds: dict, workload_bounds: dict) -> set[int]:
    outlier_envs: set[int] = set()
    for row in rows:
        if row["imp"] <= 50:
            continue
        upper = bucket_bounds.get((row["workload"], row["pattern"])) or workload_bounds.get(row["workload"])
        reasons = classify_high_gain_outlier(
            row=row,
            default_baseline_tps=default_baselines.get(row["env"]),
            bucket_upper=upper,
        )
        if reasons:
            outlier_envs.add(row["env"])
    return outlier_envs


def build_bucket_records(env_infos: dict[int, dict], bucket_name: str, question_env_frequency: dict[str, int], seed: int) -> list[dict]:
    records = []
    for env_id in sorted(env_infos):
        info = env_infos[env_id]
        if info["bucket"] != bucket_name:
            continue
        question = choose_question(
            env_id=env_id,
            questions=info["questions"],
            question_env_frequency=question_env_frequency,
            seed=seed,
        )
        records.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                "question": question,
                "env_sample_idx": env_id,
                "bucket": bucket_name,
            }
        )
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_stats_payload(env_infos: dict[int, dict], outlier_envs: set[int], records_by_name: dict[str, list[dict]]) -> dict:
    pre_bucket = Counter(info["bucket"] for info in env_infos.values())
    post_bucket = Counter(info["bucket"] for env, info in env_infos.items() if env not in outlier_envs)
    return {
        "pre_outlier_bucket_envs": dict(pre_bucket),
        "post_outlier_bucket_envs": dict(post_bucket),
        "dropped_high_gain_outlier_envs": len(outlier_envs),
        "outputs": {
            name: {
                "rows": len(records),
                "unique_envs": len({item["env_sample_idx"] for item in records}),
            }
            for name, records in records_by_name.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="构造 v2 RL 数据集（1q/env）")
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
        "--eval-env-ids",
        default="data_pipeline/data/eval/v2/eval_source_env_ids.json",
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
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    scenarios = json.loads(Path(args.scenarios).read_text(encoding="utf-8"))
    eval_env_ids = set(json.loads(Path(args.eval_env_ids).read_text(encoding="utf-8")))
    workloads_by_env = {idx: item["workload"]["type"] for idx, item in enumerate(scenarios)}

    knob_cfg = yaml.safe_load(Path(args.knob_space).read_text(encoding="utf-8"))["knobs"]
    default_knobs = {k: v["default"] for k, v in knob_cfg.items()}
    static_knobs = {k for k, v in knob_cfg.items() if v.get("restart")}

    rows: list[dict] = []
    env_infos: dict[int, dict] = defaultdict(lambda: {"total": 0, "strong": 0, "weak": 0, "questions": set()})

    for input_file in args.input_files:
        input_path = Path(input_file)
        with input_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                env = record.get("env_sample_idx")
                if env in eval_env_ids:
                    continue
                if not has_valid_rl_prompt(record):
                    continue

                row = extract_row(
                    record=record,
                    file_name=input_path.name,
                    line_no=line_no,
                    workloads_by_env=workloads_by_env,
                    static_knobs=static_knobs,
                    family_map=FAMILY_MAP,
                )
                rows.append(row)

                info = env_infos[env]
                info["total"] += 1
                if row["imp"] >= 3:
                    info["strong"] += 1
                if row["imp"] > 0:
                    info["weak"] += 1
                question = str(record.get("question", "") or "").strip()
                if question:
                    info["questions"].add(question)

    for env_id, info in env_infos.items():
        info["bucket"] = classify_env_bucket(
            total_rollouts=info["total"],
            strong_rollouts=info["strong"],
            weak_rollouts=info["weak"],
        )

    bucket_bounds, workload_bounds = build_reference_bounds(rows)
    high_gain_envs = {
        row["env"]
        for row in rows
        if row["imp"] > 50 and isinstance(row["env"], int) and 0 <= row["env"] < len(scenarios)
    }
    model = CostModel.load(str(Path(args.checkpoint)))
    default_baselines = predict_default_baselines(model, scenarios, default_knobs, high_gain_envs)
    outlier_envs = collect_outlier_envs(
        rows=rows,
        default_baselines=default_baselines,
        bucket_bounds=bucket_bounds,
        workload_bounds=workload_bounds,
    )

    filtered_env_infos = {
        env: info
        for env, info in env_infos.items()
        if env not in outlier_envs
    }

    question_env_frequency = Counter()
    for info in filtered_env_infos.values():
        for question in info["questions"]:
            question_env_frequency[question] += 1

    frontier_records = build_bucket_records(
        env_infos=filtered_env_infos,
        bucket_name="frontier",
        question_env_frequency=question_env_frequency,
        seed=args.seed,
    )
    hard_records = build_bucket_records(
        env_infos=filtered_env_infos,
        bucket_name="hard_but_learnable",
        question_env_frequency=question_env_frequency,
        seed=args.seed,
    )
    frontier_plus_hard_records = sorted(
        frontier_records + hard_records,
        key=lambda item: item["env_sample_idx"],
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "rl_frontier_1q": frontier_records,
        "rl_hard_1q": hard_records,
        "rl_frontier_plus_hard_1q": frontier_plus_hard_records,
    }

    for name, records in outputs.items():
        write_jsonl(output_dir / f"{name}.jsonl", records)
        (output_dir / f"{name}_env_ids.json").write_text(
            json.dumps([item["env_sample_idx"] for item in records], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    stats_payload = build_stats_payload(
        env_infos=env_infos,
        outlier_envs=outlier_envs,
        records_by_name=outputs,
    )
    (output_dir / "rl_stats.json").write_text(
        json.dumps(stats_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(stats_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
