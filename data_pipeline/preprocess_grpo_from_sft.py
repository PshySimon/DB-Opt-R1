"""
GRPO 数据预处理：从 SFT 轨迹构造 verl 所需的 prompt parquet。

核心原则：
1. prompt 直接复用 SFT 轨迹中的 system prompt + question
2. env_sample_idx 必须按和 sampler 一致的场景加载顺序解释
3. 场景加载复用 DBToolEnv._load_scenarios，再应用同样的过滤参数

输出：datasets/grpo/train.parquet + datasets/grpo/validation.parquet
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import random
from pathlib import Path
from typing import Iterable

import pandas as pd

from core.db.scenario_filter import add_filter_args, filter_scenarios
from environment.tools import DBToolEnv

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_sft_records(input_files: list[str]) -> list[dict]:
    records: list[dict] = []
    resolved: list[str] = []
    for pattern in input_files:
        matches = sorted(glob.glob(pattern))
        if matches:
            resolved.extend(matches)
        elif Path(pattern).exists():
            resolved.append(pattern)

    if not resolved:
        raise FileNotFoundError(f"未找到输入文件: {input_files}")

    for path in resolved:
        logger.info("加载轨迹文件: %s", path)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))

    logger.info("共加载 %d 条轨迹", len(records))
    return records


def load_aligned_scenarios(
    scenarios: list[str],
    source_filter: str | None = None,
    tps_min: float | None = None,
    tps_max: float | None = None,
) -> list:
    loaded = DBToolEnv._load_scenarios(scenarios)
    logger.info("按 DBToolEnv 顺序加载 %d 个场景", len(loaded))
    filtered = filter_scenarios(
        loaded,
        source_filter=source_filter,
        tps_min=tps_min,
        tps_max=tps_max,
    )
    logger.info("过滤后保留 %d 个场景", len(filtered))
    return filtered


def _first_user_content(messages: Iterable[dict]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return str(message.get("content", "")).strip()
    return ""


def build_grpo_records(records: list[dict], scenarios: list) -> list[dict]:
    grpo_records: list[dict] = []

    for item in records:
        env_idx = item.get("env_sample_idx")
        messages = item.get("messages", [])
        question = str(item.get("question", "") or _first_user_content(messages)).strip()

        if env_idx is None or not isinstance(env_idx, int):
            continue
        if env_idx < 0 or env_idx >= len(scenarios):
            continue
        if not messages or messages[0].get("role") != "system":
            continue
        if not question:
            continue

        scenario = scenarios[env_idx]
        system_prompt = str(messages[0].get("content", "")).strip()

        grpo_records.append(
            {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "scenario_idx": env_idx,
                        "hardware": dict(getattr(scenario, "hardware", {}) or {}),
                    },
                },
                "data_source": "db_tuning",
            }
        )

    return grpo_records


def split_records(records: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


def main() -> None:
    parser = argparse.ArgumentParser(description="从 SFT 轨迹构造 GRPO parquet")
    parser.add_argument(
        "--input-files",
        nargs="+",
        required=True,
        help="SFT 轨迹 JSONL 文件（支持多个和 glob）",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        required=True,
        help="生成这些 SFT 轨迹时使用的场景输入，需与 sampler 保持一致",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/grpo",
        help="输出目录",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="验证集比例",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    add_filter_args(parser)
    args = parser.parse_args()

    records = load_sft_records(args.input_files)
    scenarios = load_aligned_scenarios(
        args.scenarios,
        source_filter=args.source_filter,
        tps_min=args.tps_min,
        tps_max=args.tps_max,
    )
    grpo_records = build_grpo_records(records, scenarios)

    logger.info("转换得到 %d 条 GRPO prompt", len(grpo_records))
    if not grpo_records:
        raise RuntimeError("没有生成任何 GRPO 记录，请检查输入轨迹和场景是否对齐")

    train_records, val_records = split_records(grpo_records, args.val_ratio, args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_records_ in (("train", train_records), ("validation", val_records)):
        output_path = output_dir / f"{split_name}.parquet"
        pd.DataFrame(split_records_).to_parquet(output_path, index=False)
        logger.info("已保存 %s: %s 条", output_path, len(split_records_))


if __name__ == "__main__":
    main()
