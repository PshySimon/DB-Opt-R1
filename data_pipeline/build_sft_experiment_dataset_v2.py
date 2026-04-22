import argparse
import json
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).open("r", encoding="utf-8")
        if line.strip()
    ]


def select_rows_by_manifest(train_rows: list[dict], manifest_rows: list[dict]) -> list[dict]:
    env_to_row = {int(row["env_sample_idx"]): row for row in train_rows}
    selected = []
    missing = []
    for item in manifest_rows:
        env = int(item["env_sample_idx"])
        row = env_to_row.get(env)
        if row is None:
            missing.append(env)
            continue
        selected.append(row)
    if missing:
        raise ValueError(f"missing env_sample_idx in train data: {missing[:10]}")
    return selected


def build_stats(rows: list[dict], manifest_path: str | Path, train_data_path: str | Path, output_jsonl: str | Path) -> dict:
    envs = [int(row["env_sample_idx"]) for row in rows]
    return {
        "manifest_file": str(manifest_path),
        "train_data_file": str(train_data_path),
        "output_file": str(output_jsonl),
        "rows": len(rows),
        "unique_envs": len(set(envs)),
        "min_env_sample_idx": str(min(envs)) if envs else "",
        "max_env_sample_idx": str(max(envs)) if envs else "",
    }


def write_outputs(rows: list[dict], output_jsonl: str | Path, stats_output: str | Path, manifest_path: str | Path = "", train_data_path: str | Path = "") -> None:
    output_jsonl = Path(output_jsonl)
    stats_output = Path(stats_output)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    stats_output.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = build_stats(rows, manifest_path, train_data_path, output_jsonl)
    stats_output.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build full SFT experiment dataset JSONL from a v2 manifest.")
    parser.add_argument(
        "--train-data",
        default="data_pipeline/data/train/v2/sft_trajectories_v10_train.jsonl",
        help="Train-only cleaned raw JSONL",
    )
    parser.add_argument("--manifest", required=True, help="Experiment manifest JSONL")
    parser.add_argument("--output-jsonl", required=True, help="Output experiment train JSONL")
    parser.add_argument("--stats-output", required=True, help="Output stats JSON")
    args = parser.parse_args()

    train_rows = load_jsonl(args.train_data)
    manifest_rows = load_jsonl(args.manifest)
    selected = select_rows_by_manifest(train_rows, manifest_rows)
    write_outputs(
        selected,
        args.output_jsonl,
        args.stats_output,
        manifest_path=args.manifest,
        train_data_path=args.train_data,
    )


if __name__ == "__main__":
    main()
