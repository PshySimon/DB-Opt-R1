#!/usr/bin/env python3
"""Sample and summarize GPU efficiency during training runs."""

import argparse
import csv
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


FIELDS = [
    "timestamp",
    "index",
    "gpu_util_pct",
    "mem_util_pct",
    "mem_used_mib",
    "mem_total_mib",
    "power_w",
]

NVIDIA_SMI_QUERY = [
    "timestamp",
    "index",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
]


def _float(value: str) -> float:
    cleaned = "".join(ch for ch in str(value) if ch.isdigit() or ch in ".-")
    return float(cleaned) if cleaned else 0.0


def _percentile(values, q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = min(len(values) - 1, int(len(values) * q))
    return values[index]


def sample(args) -> int:
    output_path = Path(args.output)
    if output_path.parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    query = ",".join(NVIDIA_SMI_QUERY)
    command = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]

    with output_path.open("w", newline="", buffering=1) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        while True:
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                for line in result.stdout.splitlines():
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) != len(FIELDS):
                        continue
                    writer.writerow(dict(zip(FIELDS, parts)))
                handle.flush()
            except subprocess.CalledProcessError as exc:
                print(exc.stderr.strip() or str(exc), file=sys.stderr, flush=True)

            time.sleep(args.interval)


def summary(args) -> int:
    input_path = Path(args.input)
    by_gpu = defaultdict(list)

    with input_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            gpu = str(row["index"]).strip()
            mem_total = max(_float(row["mem_total_mib"]), 1.0)
            mem_used = _float(row["mem_used_mib"])
            by_gpu[gpu].append(
                {
                    "gpu_util": _float(row["gpu_util_pct"]),
                    "mem_util": _float(row["mem_util_pct"]),
                    "mem_used_gb": mem_used / 1024.0,
                    "mem_used_pct": mem_used / mem_total * 100.0,
                    "power_w": _float(row["power_w"]),
                }
            )

    print(f"GPU_EFFICIENCY_CSV={input_path}")
    for gpu in sorted(by_gpu, key=lambda item: int(item) if item.isdigit() else item):
        rows = by_gpu[gpu]
        pieces = [f"GPU{gpu}", f"n={len(rows)}"]
        for metric in ["gpu_util", "mem_used_pct", "mem_used_gb", "mem_util", "power_w"]:
            values = [row[metric] for row in rows]
            avg = sum(values) / len(values) if values else 0.0
            suffix = "%" if metric in {"gpu_util", "mem_used_pct", "mem_util"} else ("GB" if metric == "mem_used_gb" else "W")
            pieces.append(
                f"{metric}:avg={avg:.1f}{suffix},"
                f"p10={_percentile(values, 0.1):.1f}{suffix},"
                f"p50={_percentile(values, 0.5):.1f}{suffix},"
                f"p90={_percentile(values, 0.9):.1f}{suffix}"
            )
        print(" ".join(pieces))

    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser("sample", help="sample nvidia-smi once per interval until interrupted")
    sample_parser.add_argument("--output", required=True, help="CSV output path")
    sample_parser.add_argument("--interval", type=float, default=1.0, help="sampling interval in seconds")
    sample_parser.set_defaults(func=sample)

    summary_parser = subparsers.add_parser("summary", help="summarize a CSV produced by sample")
    summary_parser.add_argument("--input", required=True, help="CSV input path")
    summary_parser.set_defaults(func=summary)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
