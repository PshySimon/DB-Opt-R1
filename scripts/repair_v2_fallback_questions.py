#!/usr/bin/env python3
"""Repair v2 eval/RL datasets that still contain fallback questions.

This script only patches prompt-level datasets whose payload is
`system + user(question) + env_sample_idx`, namely:
  - data_pipeline/data/eval/v2/eval_trajectories_v2.jsonl
  - data_pipeline/data/train/v2/rl/*.jsonl

It intentionally does NOT repair SFT trajectory datasets, because replacing the
user question without regenerating the assistant trajectory would make the
sample inconsistent.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.llm.multi_client import MultiProviderLLMClient
from data_pipeline.synthesis.scenarios.pipeline import generate_questions_for_state

FALLBACK_QUESTION = "请帮我优化一下数据库的性能配置。"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_RAW_INPUTS = [
    ROOT / "data_pipeline/data/train/sft_trajectories_v10_part1.jsonl",
    ROOT / "data_pipeline/data/train/sft_trajectories_v10_part2.jsonl",
]
DEFAULT_SCENARIOS = ROOT / "data_pipeline/data/scenarios/collected/collected_8c16g_hdd_20k.json"

DEFAULT_EVAL_PATH = ROOT / "data_pipeline/data/eval/v2/eval_trajectories_v2.jsonl"
DEFAULT_EVAL_INDEX_MAP = ROOT / "data_pipeline/data/eval/v2/eval_index_map.jsonl"

DEFAULT_RL_PATHS = [
    ROOT / "data_pipeline/data/train/v2/rl/rl_frontier_1q.jsonl",
    ROOT / "data_pipeline/data/train/v2/rl/rl_hard_1q.jsonl",
    ROOT / "data_pipeline/data/train/v2/rl/rl_frontier_plus_hard_1q.jsonl",
]


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_question(record: dict) -> str:
    if "question" in record:
        return str(record.get("question", "") or "").strip()
    messages = record.get("messages", [])
    if len(messages) >= 2 and messages[1].get("role") == "user":
        return str(messages[1].get("content", "") or "").strip()
    return ""


def set_question(record: dict, question: str) -> dict:
    updated = json.loads(json.dumps(record, ensure_ascii=False))
    updated["question"] = question
    messages = updated.get("messages", [])
    if len(messages) >= 2 and messages[1].get("role") == "user":
        messages[1]["content"] = question
    return updated


def repaired_output_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.repaired{path.suffix}")


def load_question_stats_from_raw(
    raw_paths: list[Path],
    fallback_question: str,
    progress_interval: int = 50_000,
) -> tuple[dict[int, set[str]], Counter]:
    alternatives_by_env: dict[int, set[str]] = defaultdict(set)
    question_env_frequency: Counter = Counter()

    for path in raw_paths:
        logger.info("扫描原始轨迹文件: %s", path)
        scanned = 0
        kept = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                scanned += 1
                row = json.loads(line)
                env = row.get("env_sample_idx")
                question = get_question(row)
                if not isinstance(env, int) or not question or question == fallback_question:
                    if scanned % progress_interval == 0:
                        logger.info("  已扫描 %s 行，当前可复用 env=%s", scanned, len(alternatives_by_env))
                    continue
                alternatives_by_env[env].add(question)
                kept += 1
                if scanned % progress_interval == 0:
                    logger.info("  已扫描 %s 行，保留候选 question=%s，可复用 env=%s", scanned, kept, len(alternatives_by_env))
        logger.info("完成扫描 %s: 总行数=%s, 候选 question=%s, 可复用 env=%s", path.name, scanned, kept, len(alternatives_by_env))

    for questions in alternatives_by_env.values():
        for question in questions:
            question_env_frequency[question] += 1

    return alternatives_by_env, question_env_frequency


def choose_existing_question(
    env_id: int,
    alternatives_by_env: dict[int, set[str]],
    question_env_frequency: Counter,
) -> str | None:
    candidates = sorted(
        alternatives_by_env.get(env_id, set()),
        key=lambda q: (question_env_frequency.get(q, 0), len(q), q),
    )
    if not candidates:
        return None
    return candidates[0]


def load_eval_source_map(path: Path) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for row in read_jsonl(path):
        mapping[int(row["eval_env_sample_idx"])] = int(row["source_env_sample_idx"])
    return mapping


def collect_target_source_envs(
    target_name: str,
    rows: list[dict],
    fallback_question: str,
    eval_source_map: dict[int, int] | None = None,
) -> set[int]:
    source_envs: set[int] = set()
    for row in rows:
        question = get_question(row)
        if question != fallback_question:
            continue
        env = row.get("env_sample_idx")
        if not isinstance(env, int):
            continue
        if target_name == "eval":
            if eval_source_map is None or env not in eval_source_map:
                raise ValueError(f"eval env {env} 缺少 source env 映射")
            source_envs.add(eval_source_map[env])
        else:
            source_envs.add(env)
    return source_envs


def build_llm_fn(args: argparse.Namespace) -> Callable[[str], str]:
    client = MultiProviderLLMClient(
        target_model=args.model,
        providers_config=args.providers_config,
        single_api_key=args.api_key,
        single_api_base=args.api_base,
        api_max_concurrent=args.api_max_concurrent,
    )

    def generate(prompt: str) -> str:
        return client.generate(prompt, temperature=args.temperature)

    return generate


def build_repair_map(
    source_env_ids: set[int],
    alternatives_by_env: dict[int, set[str]],
    question_env_frequency: Counter,
    *,
    regenerate_missing: bool,
    scenarios: list[dict] | None,
    llm_fn: Callable[[str], str] | None,
    progress_interval: int = 10,
) -> tuple[dict[int, str], dict[str, object]]:
    repair_map: dict[int, str] = {}
    summary = {
        "requested_envs": len(source_env_ids),
        "reused_from_raw": 0,
        "generated": 0,
        "still_missing": 0,
    }

    missing: list[int] = []
    for env_id in sorted(source_env_ids):
        replacement = choose_existing_question(env_id, alternatives_by_env, question_env_frequency)
        if replacement:
            repair_map[env_id] = replacement
            summary["reused_from_raw"] += 1
        else:
            missing.append(env_id)

    logger.info(
        "fallback env 统计: 总计=%s, raw可直接复用=%s, 仍需重生=%s",
        summary["requested_envs"],
        summary["reused_from_raw"],
        len(missing),
    )

    if missing and regenerate_missing:
        if scenarios is None or llm_fn is None:
            raise ValueError("开启 regenerate_missing 时，必须提供 scenarios 和 llm_fn")
        logger.info("开始重生缺失 question: %s 个 env", len(missing))
        for idx, env_id in enumerate(missing, 1):
            replacement = str(generate_questions_for_state(scenarios[env_id], 1, llm_fn)[0]).strip()
            if not replacement or replacement == FALLBACK_QUESTION:
                summary["still_missing"] += 1
                logger.warning("  env=%s question 重生失败，仍为 fallback/空值", env_id)
                continue
            repair_map[env_id] = replacement
            summary["generated"] += 1
            if idx % progress_interval == 0 or idx == len(missing):
                logger.info(
                    "  重生进度: %s/%s, 成功=%s, 失败=%s",
                    idx,
                    len(missing),
                    summary["generated"],
                    summary["still_missing"],
                )
    else:
        summary["still_missing"] = len(missing)

    return repair_map, summary


def apply_repairs_to_rows(
    target_name: str,
    rows: list[dict],
    repair_map: dict[int, str],
    fallback_question: str,
    eval_source_map: dict[int, int] | None = None,
) -> tuple[list[dict], dict[str, int]]:
    repaired_rows: list[dict] = []
    patched = 0
    unresolved = 0

    for row in rows:
        question = get_question(row)
        if question != fallback_question:
            repaired_rows.append(row)
            continue

        env = row.get("env_sample_idx")
        if not isinstance(env, int):
            repaired_rows.append(row)
            unresolved += 1
            continue

        source_env = eval_source_map[env] if target_name == "eval" else env
        replacement = repair_map.get(source_env)
        if replacement:
            repaired_rows.append(set_question(row, replacement))
            patched += 1
        else:
            repaired_rows.append(row)
            unresolved += 1

    return repaired_rows, {"patched_rows": patched, "unresolved_rows": unresolved}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair fallback questions in v2 eval/RL datasets.")
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=["eval", "rl"],
        default=["eval", "rl"],
        help="要修复的目标数据集，默认同时修 eval 和 rl",
    )
    parser.add_argument(
        "--raw-inputs",
        nargs="+",
        default=[str(path) for path in DEFAULT_RAW_INPUTS],
        help="原始 v10 轨迹文件，用于复用同 env 的现成 question",
    )
    parser.add_argument(
        "--scenarios",
        default=str(DEFAULT_SCENARIOS),
        help="场景文件，只有在 --regenerate-missing 时才会用到",
    )
    parser.add_argument(
        "--eval-path",
        default=str(DEFAULT_EVAL_PATH),
        help="eval question 文件路径",
    )
    parser.add_argument(
        "--eval-index-map",
        default=str(DEFAULT_EVAL_INDEX_MAP),
        help="eval local env 到 source env 的映射文件",
    )
    parser.add_argument(
        "--rl-paths",
        nargs="+",
        default=[str(path) for path in DEFAULT_RL_PATHS],
        help="RL jsonl 文件路径列表",
    )
    parser.add_argument(
        "--regenerate-missing",
        action="store_true",
        help="对 raw 中找不到替代 question 的 env，调用 LLM 重新生成 question",
    )
    parser.add_argument("--providers-config", default=None)
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-max-concurrent", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        help="重生缺失 question 时的日志进度间隔，默认每 10 个 env 打一条",
    )
    parser.add_argument(
        "--scan-progress-interval",
        type=int,
        default=50_000,
        help="扫描 raw 文件时的日志进度间隔，默认每 50000 行打一次",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="原地覆盖目标文件；默认写出 *.repaired.jsonl",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="可选，输出修复统计 JSON",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger.info("开始修复 v2 fallback question")
    logger.info("targets=%s, regenerate_missing=%s, in_place=%s", args.targets, args.regenerate_missing, args.in_place)

    raw_paths = [Path(path) for path in args.raw_inputs]
    alternatives_by_env, question_env_frequency = load_question_stats_from_raw(
        raw_paths=raw_paths,
        fallback_question=FALLBACK_QUESTION,
        progress_interval=args.scan_progress_interval,
    )

    eval_source_map = None
    target_rows: dict[str, list[dict] | dict[str, list[dict]]] = {}
    source_env_ids: set[int] = set()

    if "eval" in args.targets:
        eval_path = Path(args.eval_path)
        eval_rows = read_jsonl(eval_path)
        eval_source_map = load_eval_source_map(Path(args.eval_index_map))
        target_rows["eval"] = eval_rows
        source_env_ids.update(
            collect_target_source_envs(
                target_name="eval",
                rows=eval_rows,
                fallback_question=FALLBACK_QUESTION,
                eval_source_map=eval_source_map,
            )
        )

    if "rl" in args.targets:
        rl_rows = {str(path): read_jsonl(Path(path)) for path in args.rl_paths}
        target_rows["rl"] = rl_rows
        for rows in rl_rows.values():
            source_env_ids.update(
                collect_target_source_envs(
                    target_name="rl",
                    rows=rows,
                    fallback_question=FALLBACK_QUESTION,
                )
            )

    llm_fn = None
    scenarios = None
    if args.regenerate_missing:
        if not args.providers_config and not args.api_key:
            raise SystemExit("开启 --regenerate-missing 时，必须提供 --providers-config 或 --api-key")
        llm_fn = build_llm_fn(args)
        scenarios = json.loads(Path(args.scenarios).read_text(encoding="utf-8"))

    repair_map, repair_summary = build_repair_map(
        source_env_ids=source_env_ids,
        alternatives_by_env=alternatives_by_env,
        question_env_frequency=question_env_frequency,
        regenerate_missing=args.regenerate_missing,
        scenarios=scenarios,
        llm_fn=llm_fn,
        progress_interval=args.progress_interval,
    )

    report: dict[str, object] = {
        "fallback_question": FALLBACK_QUESTION,
        "repair_summary": repair_summary,
        "outputs": {},
    }

    if "eval" in target_rows:
        eval_path = Path(args.eval_path)
        repaired_rows, stats = apply_repairs_to_rows(
            target_name="eval",
            rows=target_rows["eval"],
            repair_map=repair_map,
            fallback_question=FALLBACK_QUESTION,
            eval_source_map=eval_source_map,
        )
        output_path = eval_path if args.in_place else repaired_output_path(eval_path)
        write_jsonl(output_path, repaired_rows)
        logger.info("eval 修复完成: patched=%s, unresolved=%s, output=%s", stats["patched_rows"], stats["unresolved_rows"], output_path)
        report["outputs"]["eval"] = {
            "path": str(output_path),
            **stats,
        }

    if "rl" in target_rows:
        rl_outputs: dict[str, object] = {}
        for path_str, rows in target_rows["rl"].items():
            path = Path(path_str)
            repaired_rows, stats = apply_repairs_to_rows(
                target_name="rl",
                rows=rows,
                repair_map=repair_map,
                fallback_question=FALLBACK_QUESTION,
            )
            output_path = path if args.in_place else repaired_output_path(path)
            write_jsonl(output_path, repaired_rows)
            logger.info(
                "rl 修复完成: file=%s, patched=%s, unresolved=%s, output=%s",
                path.name,
                stats["patched_rows"],
                stats["unresolved_rows"],
                output_path,
            )
            rl_outputs[str(output_path)] = stats
        report["outputs"]["rl"] = rl_outputs

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("统计报告已写出: %s", report_path)

    logger.info("全部完成")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
