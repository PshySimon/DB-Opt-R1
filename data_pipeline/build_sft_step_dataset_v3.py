"""Build v3 step-level SFT datasets from v2 full DB-agent trajectories.

The v2 full SFT data stores a whole multi-turn agent trajectory as one
``messages`` sample.  For v3 we instead create one training sample per
assistant step so that LLaMA-Factory can supervise only the current output via
``mask_history``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any


THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def load_jsonl(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).open("r", encoding="utf-8")
        if line.strip()
    ]


def strip_think_block(content: str) -> str:
    return THINK_BLOCK_RE.sub("", content or "").strip()


def wrap_tool_response(content: str) -> str:
    return f"<tool_response>\n{(content or '').strip()}\n</tool_response>"


def extract_tool_name(content: str) -> str:
    match = TOOL_CALL_RE.search(content or "")
    if not match:
        return ""
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return ""
    name = payload.get("name")
    return name if isinstance(name, str) else ""


def has_closed_think(content: str) -> bool:
    if "<think>" not in content and "</think>" not in content:
        return True
    return bool(re.search(r"<think>.*?</think>", content, re.DOTALL))


def has_closed_tool_call(content: str) -> bool:
    if "<tool_call>" not in content and "</tool_call>" not in content:
        return False
    return bool(TOOL_CALL_RE.search(content))


def build_step_rows(
    trajectories: list[dict],
    *,
    history_turns: int = 4,
    remove_think_history: bool = True,
    require_tool_call: bool = True,
    finish_oversample: int = 1,
    max_trajectories: int | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    if history_turns < 0:
        raise ValueError("history_turns must be >= 0")
    if finish_oversample < 1:
        raise ValueError("finish_oversample must be >= 1")

    selected = list(trajectories)
    if max_trajectories is not None:
        selected = selected[:max_trajectories]

    rows: list[dict] = []
    skipped = Counter()
    tools = Counter()
    finish_steps = 0
    assistant_steps = 0

    for traj_idx, traj in enumerate(selected):
        messages = traj.get("messages") or []
        if len(messages) < 3 or messages[0].get("role") != "system":
            skipped["bad_messages"] += 1
            continue

        system = str(messages[0].get("content", "") or "")
        history: list[list[str]] = []
        pending_instruction = ""
        step_idx = 0

        for msg in messages[1:]:
            role = msg.get("role")
            content = str(msg.get("content", "") or "")

            if role == "user":
                pending_instruction = content.strip()
                continue
            if role == "tool":
                pending_instruction = wrap_tool_response(content)
                continue
            if role != "assistant":
                skipped["unsupported_role"] += 1
                continue

            assistant_steps += 1
            if not pending_instruction:
                skipped["missing_instruction"] += 1
                continue
            if not has_closed_think(content):
                skipped["bad_think"] += 1
                continue
            if require_tool_call and not has_closed_tool_call(content):
                skipped["missing_tool_call"] += 1
                history_output = strip_think_block(content) if remove_think_history else content.strip()
                if history_output:
                    history.append([pending_instruction, history_output])
                pending_instruction = ""
                continue

            tool_name = extract_tool_name(content)
            is_final_step = tool_name == "finish_tuning"
            repeat = finish_oversample if is_final_step else 1
            if is_final_step:
                finish_steps += 1
            tools[tool_name or ""] += 1

            base_row = {
                "system": system,
                "instruction": pending_instruction,
                "input": "",
                "output": content.strip(),
                "history": history[-history_turns:] if history_turns else [],
                "meta": {
                    "source": "db_agent_v3",
                    "source_traj_idx": traj_idx,
                    "env_sample_idx": traj.get("env_sample_idx"),
                    "step_idx": step_idx,
                    "target_tool": tool_name,
                    "is_final_step": is_final_step,
                    "reward": traj.get("reward"),
                    "improvement_pct": traj.get("improvement_pct"),
                },
            }
            for oversample_idx in range(repeat):
                row = json.loads(json.dumps(base_row, ensure_ascii=False))
                row["meta"]["oversample_idx"] = oversample_idx
                rows.append(row)

            history_output = strip_think_block(content) if remove_think_history else content.strip()
            if history_output:
                history.append([pending_instruction, history_output])
            pending_instruction = ""
            step_idx += 1

    stats = {
        "source_trajectory_rows": len(selected),
        "assistant_steps": assistant_steps,
        "output_rows": len(rows),
        "finish_steps": finish_steps,
        "finish_oversample": finish_oversample,
        "oversampled_finish_extra_rows": max(finish_oversample - 1, 0) * finish_steps,
        "history_turns": history_turns,
        "remove_think_history": remove_think_history,
        "require_tool_call": require_tool_call,
        "target_tool_counts": dict(sorted(tools.items())),
        "skipped": dict(sorted(skipped.items())),
    }
    return rows, stats


def write_outputs(
    rows: list[dict],
    stats: dict,
    output_json: str | Path,
    stats_output: str | Path,
    *,
    max_shard_mb: int | None = None,
) -> None:
    output_json = Path(output_json)
    stats_output = Path(stats_output)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    stats_output.parent.mkdir(parents=True, exist_ok=True)
    if output_json.suffix == ".jsonl" and max_shard_mb:
        max_shard_bytes = max_shard_mb * 1024 * 1024
        shard_paths: list[Path] = []
        tmp_paths: list[Path] = []
        shard_idx = 0
        current_size = 0
        current_f = None

        def open_next_shard() -> None:
            nonlocal shard_idx, current_size, current_f
            if current_f is not None:
                current_f.close()
            tmp_path = output_json.with_name(f"{output_json.stem}-{shard_idx:05d}.tmp")
            tmp_paths.append(tmp_path)
            current_f = tmp_path.open("w", encoding="utf-8")
            shard_idx += 1
            current_size = 0

        for stale in output_json.parent.glob(f"{output_json.stem}-*-of-*.jsonl"):
            stale.unlink()
        open_next_shard()
        for row in rows:
            line = json.dumps(row, ensure_ascii=False) + "\n"
            line_size = len(line.encode("utf-8"))
            if current_size and current_size + line_size > max_shard_bytes:
                open_next_shard()
            assert current_f is not None
            current_f.write(line)
            current_size += line_size
        if current_f is not None:
            current_f.close()

        total = len(tmp_paths)
        for idx, tmp_path in enumerate(tmp_paths):
            shard_path = output_json.with_name(f"{output_json.stem}-{idx:05d}-of-{total:05d}.jsonl")
            tmp_path.replace(shard_path)
            shard_paths.append(shard_path)
        stats["output_jsonl_shards"] = [str(path) for path in shard_paths]
        stats["max_shard_mb"] = max_shard_mb
    elif output_json.suffix == ".jsonl":
        with output_json.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        stats["output_jsonl_shards"] = [str(output_json)]
    else:
        output_json.write_text(
            json.dumps(rows, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    stats_output.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build v3 step-level LLaMA-Factory SFT data from v2 trajectories.")
    parser.add_argument("--input-jsonl", default="data_pipeline/data/train/v2/sft_trajectories_v10_train.jsonl")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--stats-output", required=True)
    parser.add_argument("--history-turns", type=int, default=4)
    parser.add_argument("--keep-think-history", action="store_true")
    parser.add_argument("--allow-missing-tool-call", action="store_true")
    parser.add_argument("--finish-oversample", type=int, default=1)
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--max-shard-mb", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    trajectories = load_jsonl(args.input_jsonl)
    rows, stats = build_step_rows(
        trajectories,
        history_turns=args.history_turns,
        remove_think_history=not args.keep_think_history,
        require_tool_call=not args.allow_missing_tool_call,
        finish_oversample=args.finish_oversample,
        max_trajectories=args.max_trajectories,
    )
    stats.update(
        {
            "input_jsonl": str(args.input_jsonl),
            "output_json": str(args.output_json),
        }
    )
    write_outputs(rows, stats, args.output_json, args.stats_output, max_shard_mb=args.max_shard_mb)


if __name__ == "__main__":
    main()
