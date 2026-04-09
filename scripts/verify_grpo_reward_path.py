#!/usr/bin/env python3
"""Verify the GRPO reward path with real tokenizer, parquet ground truth and cost model."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from verl import DataProto

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.db.knob_space import KnobSpace, format_memory, parse_memory
from cost_model.model import CostModel
from training.reward_score import compute_score_answer
from training.verl.main_grpo import DBRewardManager


def _memory_candidate(name: str, target_mb: float, knob_space: KnobSpace) -> str:
    info = knob_space.get_knob_info(name)
    min_mb = parse_memory(str(info["min"])) / 1024
    max_mb = parse_memory(str(info["max"])) / 1024
    chosen_mb = max(min_mb, min(max_mb, target_mb))
    return format_memory(int(chosen_mb * 1024))


def iter_candidate_knob_configs(
    ground_truth: dict,
    knob_space_path: str,
    max_random_candidates: int = 64,
    seed: int = 0,
) -> Iterable[dict[str, object]]:
    knob_space = KnobSpace(knob_space_path)
    hardware = ground_truth.get("hardware", {})
    total_memory_gb = float(hardware.get("total_memory_gb") or 16.0)
    cpu_count = int(hardware.get("cpu_count") or 8)

    shared_buffers_25 = _memory_candidate("shared_buffers", total_memory_gb * 1024 * 0.25, knob_space)
    shared_buffers_40 = _memory_candidate("shared_buffers", total_memory_gb * 1024 * 0.40, knob_space)
    effective_cache_50 = _memory_candidate("effective_cache_size", total_memory_gb * 1024 * 0.50, knob_space)
    effective_cache_75 = _memory_candidate("effective_cache_size", total_memory_gb * 1024 * 0.75, knob_space)
    maint_mem = _memory_candidate("maintenance_work_mem", min(total_memory_gb * 1024 * 0.05, 1024), knob_space)

    deterministic = [
        {"shared_buffers": "8GB"},
        {"shared_buffers": shared_buffers_25},
        {"shared_buffers": shared_buffers_25, "effective_cache_size": effective_cache_75},
        {"shared_buffers": shared_buffers_40, "effective_cache_size": effective_cache_75},
        {
            "shared_buffers": shared_buffers_25,
            "effective_cache_size": effective_cache_75,
            "work_mem": "64MB",
            "maintenance_work_mem": maint_mem,
        },
        {
            "shared_buffers": shared_buffers_25,
            "effective_cache_size": effective_cache_50,
            "work_mem": "64MB",
            "effective_io_concurrency": min(200, max(32, cpu_count * 4)),
            "random_page_cost": 1.1,
        },
        {
            "shared_buffers": shared_buffers_25,
            "effective_cache_size": effective_cache_75,
            "max_parallel_workers": min(16, max(4, cpu_count)),
            "max_parallel_workers_per_gather": min(8, max(2, cpu_count // 4)),
        },
    ]

    seen = set()
    for candidate in deterministic:
        key = json.dumps(candidate, sort_keys=True)
        if key not in seen:
            seen.add(key)
            yield candidate

    knob_names = knob_space.get_knob_names()
    rng = random.Random(seed)
    for _ in range(max_random_candidates):
        candidate = {}
        for name in rng.sample(knob_names, k=rng.randint(1, 4)):
            info = knob_space.get_knob_info(name)
            knob_type = info["type"]
            if knob_type == "memory":
                target_ratio = rng.choice([0.1, 0.2, 0.25, 0.4, 0.5, 0.75])
                candidate[name] = _memory_candidate(name, total_memory_gb * 1024 * target_ratio, knob_space)
            elif knob_type == "integer":
                min_value = int(info["min"])
                max_value = int(info["max"])
                candidate[name] = rng.choice([min_value, max_value, (min_value + max_value) // 2])
            elif knob_type == "float":
                min_value = float(info["min"])
                max_value = float(info["max"])
                candidate[name] = round(rng.choice([min_value, max_value, (min_value + max_value) / 2]), 4)
            elif knob_type == "enum":
                candidate[name] = rng.choice(list(info["values"]))
        key = json.dumps(candidate, sort_keys=True)
        if key not in seen:
            seen.add(key)
            yield candidate


def build_solution(knobs: dict[str, object]) -> str:
    return (
        "<|im_start|>assistant\n"
        "<think>Apply a candidate PostgreSQL configuration and verify the expected performance uplift.</think>\n"
        f'<tool_call>{{"name":"set_knob","arguments":{{"knobs":"{json.dumps(knobs).replace(chr(34), chr(92) + chr(34))}"}}}}</tool_call>\n'
        "<|im_end|>"
    )


def find_positive_knob_config(
    ground_truth: dict,
    cost_model,
    knob_space_path: str,
    max_random_candidates: int = 64,
    seed: int = 0,
):
    best_knobs = None
    best_score = float("-inf")
    for knobs in iter_candidate_knob_configs(
        ground_truth=ground_truth,
        knob_space_path=knob_space_path,
        max_random_candidates=max_random_candidates,
        seed=seed,
    ):
        score = compute_score_answer(
            solution_str=build_solution(knobs),
            ground_truth=ground_truth,
            cost_model=cost_model,
        )
        if score > best_score:
            best_knobs = knobs
            best_score = score
        if score > 0:
            return knobs, score
    raise RuntimeError(f"no positive knob config found; best_score={best_score:.6f}, best_knobs={best_knobs}")


def build_reward_batch(tokenizer, prompt_messages, solution: str, ground_truth: dict) -> DataProto:
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        tokenize=True,
    )
    response_ids = tokenizer.encode(solution, add_special_tokens=False)
    return DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([prompt_ids], dtype=torch.long),
            "responses": torch.tensor([response_ids], dtype=torch.long),
            "attention_mask": torch.tensor(
                [[1] * len(prompt_ids) + [1] * len(response_ids)],
                dtype=torch.long,
            ),
        },
        non_tensors={
            "reward_model": np.array([{"ground_truth": ground_truth}], dtype=object),
            "data_source": np.array(["verify"], dtype=object),
        },
    )


def verify_reward_path(
    train_file: str,
    model_path: str,
    cost_model_path: str,
    knob_space_path: str = "configs/knob_space.yaml",
    sample_index: int = 0,
    max_random_candidates: int = 64,
    seed: int = 0,
):
    df = pd.read_parquet(train_file)
    row = df.iloc[sample_index]
    reward_model = row["reward_model"]
    ground_truth = reward_model["ground_truth"]
    prompt_messages = row["prompt"]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    cost_model = CostModel.load(cost_model_path)

    knobs, search_score = find_positive_knob_config(
        ground_truth=ground_truth,
        cost_model=cost_model,
        knob_space_path=knob_space_path,
        max_random_candidates=max_random_candidates,
        seed=seed,
    )
    solution = build_solution(knobs)
    batch = build_reward_batch(tokenizer, prompt_messages, solution, ground_truth)
    reward_tensor, answer_scores, format_scores = DBRewardManager(
        tokenizer=tokenizer,
        cost_model=cost_model,
    )(batch)

    answer_score = float(answer_scores[0])
    format_score = float(format_scores[0])
    final_reward = float(reward_tensor[0, -1].item())

    return {
        "sample_index": sample_index,
        "selected_knobs": knobs,
        "search_answer_score": float(search_score),
        "db_reward_manager_answer_score": answer_score,
        "db_reward_manager_format_score": format_score,
        "db_reward_manager_final_reward": final_reward,
        "status": "PASS" if answer_score > 0.0 and final_reward > format_score else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify GRPO reward path with real artifacts.")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--cost-model-path", required=True)
    parser.add_argument("--knob-space-path", default="configs/knob_space.yaml")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-random-candidates", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    result = verify_reward_path(
        train_file=args.train_file,
        model_path=args.model_path,
        cost_model_path=args.cost_model_path,
        knob_space_path=args.knob_space_path,
        sample_index=args.sample_index,
        max_random_candidates=args.max_random_candidates,
        seed=args.seed,
    )

    for key, value in result.items():
        print(f"{key}={value}")

    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
