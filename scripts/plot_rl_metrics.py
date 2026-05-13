#!/usr/bin/env python3
"""Plot verl GRPO step metrics from a Ray worker log."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_file", help="Ray worker .out file containing `step:N - ...` metrics")
    parser.add_argument("output_png", help="Output PNG path")
    return parser.parse_args()


def load_metrics(log_file: Path) -> list[dict[str, float]]:
    metrics: list[dict[str, float]] = []
    for line in log_file.read_text(errors="ignore").splitlines():
        match = re.search(r"step:(\d+)\s+-\s+(.*)", line)
        if not match:
            continue
        row: dict[str, float] = {"step": float(match.group(1))}
        for part in re.split(r"\s+-\s+", match.group(2)):
            if ":" not in part:
                continue
            key, value = part.split(":", 1)
            try:
                row[key.strip()] = float(value.strip())
            except ValueError:
                pass
        if "critic/score/mean" in row:
            metrics.append(row)

    dedup: dict[int, dict[str, float]] = {}
    for row in metrics:
        dedup[int(row["step"])] = row
    return [dedup[step] for step in sorted(dedup)]


def print_latest(metrics: list[dict[str, float]]) -> None:
    latest = metrics[-1]
    print("steps =", [int(row["step"]) for row in metrics])
    print("latest step =", int(latest["step"]))
    for key in [
        "critic/score/mean",
        "critic/answer_score/mean",
        "critic/format_score/mean",
        "critic/advantages/mean",
        "critic/advantages/max",
        "critic/advantages/min",
        "actor/pg_loss",
        "actor/kl_loss",
        "actor/entropy",
        "actor/grad_norm",
        "termination/finish_tuning_rate",
        "termination/max_turns_reached_rate",
        "response_length/mean",
        "response_length/clip_ratio",
        "turns/mean",
        "timing_s/step",
    ]:
        print(f"{key:40s} {latest.get(key)}")


def plot(metrics: list[dict[str, float]], output_png: Path) -> None:
    def series(key: str) -> tuple[list[float], list[float]]:
        xs: list[float] = []
        ys: list[float] = []
        for row in metrics:
            if key in row:
                xs.append(row["step"])
                ys.append(row[key])
        return xs, ys

    panels = [
        ("Reward", ["critic/score/mean", "critic/answer_score/mean", "critic/format_score/mean"]),
        ("Advantage", ["critic/advantages/mean", "critic/advantages/max", "critic/advantages/min"]),
        ("Policy", ["actor/pg_loss", "actor/kl_loss", "actor/entropy", "actor/grad_norm"]),
        (
            "Behavior",
            [
                "termination/finish_tuning_rate",
                "termination/max_turns_reached_rate",
                "response_length/clip_ratio",
                "turns/mean",
            ],
        ),
        ("Perf", ["timing_s/step", "timing_s/gen", "timing_s/update_actor"]),
    ]

    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 16), sharex=True)
    for ax, (title, keys) in zip(axes, panels):
        for key in keys:
            xs, ys = series(key)
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=1.8, label=key)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    axes[-1].set_xlabel("step")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)


def main() -> None:
    args = parse_args()
    log_file = Path(args.log_file)
    output_png = Path(args.output_png)
    metrics = load_metrics(log_file)
    if not metrics:
        raise SystemExit(f"no step metrics found in {log_file}")
    print_latest(metrics)
    plot(metrics, output_png)
    print("saved =", output_png)


if __name__ == "__main__":
    main()
