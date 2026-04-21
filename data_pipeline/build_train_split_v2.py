import argparse
import json
from pathlib import Path


def filter_train_rows(rows: list[dict], eval_env_ids: set[int]) -> list[dict]:
    kept = [row for row in rows if row["env_sample_idx"] not in eval_env_ids]
    kept.sort(key=lambda row: row["env_sample_idx"])
    return kept


def write_train_outputs(rows: list[dict], output_dir: Path, output_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    env_ids = [row["env_sample_idx"] for row in rows]
    (output_dir / "train_env_ids.json").write_text(
        json.dumps(env_ids, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    stats = {
        "rows": len(rows),
        "unique_envs": len(set(env_ids)),
        "output_file": str(output_path),
    }
    (output_dir / "train_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Build v2 train split by excluding eval env ids.")
    parser.add_argument(
        "--input",
        default="data_pipeline/data/pool/v2/sft_trajectories_v10_cleaned_pool.jsonl",
    )
    parser.add_argument(
        "--eval-env-ids",
        default="data_pipeline/data/eval/v2/eval_source_env_ids.json",
    )
    parser.add_argument(
        "--output-dir",
        default="data_pipeline/data/train/v2",
    )
    parser.add_argument(
        "--output-name",
        default="sft_trajectories_v10_train.jsonl",
    )
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.input).open("r", encoding="utf-8")
        if line.strip()
    ]
    eval_env_ids = set(json.loads(Path(args.eval_env_ids).read_text(encoding="utf-8")))
    kept = filter_train_rows(rows, eval_env_ids=eval_env_ids)
    write_train_outputs(
        rows=kept,
        output_dir=Path(args.output_dir),
        output_name=args.output_name,
    )
    print(
        json.dumps(
            {
                "input_rows": len(rows),
                "eval_rows": len(eval_env_ids),
                "train_rows": len(kept),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
