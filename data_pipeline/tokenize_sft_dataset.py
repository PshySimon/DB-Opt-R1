"""Pre-tokenize TRL SFT JSONL datasets into reusable HF Arrow caches."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import re
from pathlib import Path
import random
import shutil
from statistics import mean
from typing import Any

from datasets import Dataset
from transformers import AutoTokenizer

from training.data_utils import load_sft_data
from training.trl import sft


def as_list(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError(f"Expected a single sequence, got nested list with {len(value)} rows")
        value = value[0]
    if not isinstance(value, list):
        raise TypeError(f"Expected list-like tokenizer output, got {type(value).__name__}")
    return [int(x) for x in value]


def infer_model_tag(model_path: str) -> str:
    lowered = str(model_path).lower()
    if "qwen3" in lowered:
        return "qwen3"
    if "qwen2.5" in lowered or "qwen2_5" in lowered or "qwen2-5" in lowered:
        return "qwen2_5"
    name = Path(str(model_path).rstrip("/")).name.lower()
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_") or "model"


def infer_dataset_name(data_files: list[str]) -> str:
    if len(data_files) == 1:
        path = Path(data_files[0])
        if path.name == "train.jsonl":
            return path.parent.name
        return path.stem
    return "sft_mix"


def default_output_dir(
    project_root: Path,
    dataset_name: str,
    model_tag: str,
    max_length: int,
    seed: int,
) -> Path:
    return (
        project_root
        / "data_pipeline/data/tokenized_sft"
        / f"{dataset_name}__{model_tag}__ml{max_length}__seed{seed}"
    )


def percentile(sorted_values: list[int], q: float) -> int:
    if not sorted_values:
        return 0
    return sorted_values[int((len(sorted_values) - 1) * q)]


def summarize(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "mean": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}
    sorted_values = sorted(values)
    return {
        "n": len(values),
        "mean": mean(values),
        "p50": percentile(sorted_values, 0.50),
        "p90": percentile(sorted_values, 0.90),
        "p95": percentile(sorted_values, 0.95),
        "p99": percentile(sorted_values, 0.99),
        "max": sorted_values[-1],
    }


def tokenize_record(record: dict, tokenizer, max_length: int) -> dict | None:
    processed = tokenizer.apply_chat_template(
        record["messages"],
        tools=sft.extract_tools(record),
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        **sft.extract_chat_template_kwargs(record),
    )
    input_ids = as_list(processed["input_ids"])
    assistant_masks = as_list(processed.get("assistant_masks") or [])
    if len(input_ids) != len(assistant_masks):
        raise ValueError(
            f"assistant_masks length mismatch: input_ids={len(input_ids)} masks={len(assistant_masks)}"
        )
    if len(input_ids) > max_length:
        return None
    assistant_tokens = int(sum(assistant_masks))
    if assistant_tokens <= 0:
        return None

    row = {
        "input_ids": input_ids,
        "assistant_masks": assistant_masks,
        "length": len(input_ids),
        "assistant_tokens": assistant_tokens,
    }
    for key in ("data_source", "source_dataset"):
        if key in record:
            row[key] = str(record[key])
    return row


def split_records(records: list[dict], train_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    train_records = shuffled[:split_idx]
    eval_records = shuffled[split_idx:]
    if not eval_records and len(train_records) > 1:
        eval_records = [train_records.pop()]
    return train_records, eval_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pre-tokenize JSONL SFT data for TRL SFTTrainer")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_files", nargs="+", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--train_ratio", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chat_template_path", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log_every", type=int, default=500)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    project_root = Path.cwd()
    dataset_name = args.dataset_name or infer_dataset_name(args.data_files)
    model_tag = infer_model_tag(args.model_path)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else default_output_dir(project_root, dataset_name, model_tag, args.max_length, args.seed)
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    sft.maybe_apply_training_chat_template(args, tokenizer)

    raw_records = load_sft_data(args.data_files)
    tokenized_records = []
    skipped_overlength_or_empty = 0
    errors = 0
    for idx, record in enumerate(raw_records, start=1):
        try:
            tokenized = tokenize_record(record, tokenizer, args.max_length)
        except Exception as exc:  # noqa: BLE001 - keep preprocessing resilient and report exact count.
            errors += 1
            print(f"ERROR tokenizing record {idx}: {type(exc).__name__}: {exc}", flush=True)
            continue
        if tokenized is None:
            skipped_overlength_or_empty += 1
            continue
        tokenized_records.append(tokenized)
        if args.log_every > 0 and idx % args.log_every == 0:
            print(
                f"processed={idx}/{len(raw_records)} kept={len(tokenized_records)} "
                f"skipped={skipped_overlength_or_empty} errors={errors}",
                flush=True,
            )

    if not tokenized_records:
        raise RuntimeError("No valid tokenized records were produced")

    train_records, eval_records = split_records(tokenized_records, args.train_ratio, args.seed)
    Dataset.from_list(train_records).save_to_disk(str(output_dir / "train"))
    Dataset.from_list(eval_records).save_to_disk(str(output_dir / "eval"))

    lengths = [row["length"] for row in tokenized_records]
    assistant_tokens = [row["assistant_tokens"] for row in tokenized_records]
    by_source: dict[str, list[int]] = {}
    for row in tokenized_records:
        source = row.get("data_source") or row.get("source_dataset") or "unknown"
        by_source.setdefault(source, []).append(row["length"])

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_data_files": args.data_files,
        "dataset_name": dataset_name,
        "model_path": args.model_path,
        "model_tag": model_tag,
        "chat_template_path": args.chat_template_path,
        "max_length": args.max_length,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "records_before_filter": len(raw_records),
        "records_after_filter": len(tokenized_records),
        "train_records": len(train_records),
        "eval_records": len(eval_records),
        "skipped_overlength_or_empty": skipped_overlength_or_empty,
        "tokenize_errors": errors,
    }
    stats = {
        **metadata,
        "length": summarize(lengths),
        "assistant_tokens": summarize(assistant_tokens),
        "by_source_length": {source: summarize(values) for source, values in sorted(by_source.items())},
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
