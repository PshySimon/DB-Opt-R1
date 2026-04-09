#!/usr/bin/env python3
"""Minimal local verification for answer_score reward extraction."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.reward_score import compute_score_answer, extract_final_knobs


class _FakeCostModel:
    def predict(self, knobs, hardware):
        return 120.0 if knobs.get("shared_buffers") == "8GB" else 100.0


def main() -> int:
    solution = (
        "<|im_start|>assistant\n"
        "<think>set knob</think>\n"
        '<tool_call>{"name":"set_knob","arguments":{"knobs":"{\\"shared_buffers\\": \\"8GB\\", \\"work_mem\\": \\"64MB\\"}"}}</tool_call>\n'
        "<|im_end|>"
    )
    ground_truth = {"hardware": {"total_memory_gb": 80.0}}

    knobs = extract_final_knobs(solution)
    score = compute_score_answer(solution, ground_truth, cost_model=_FakeCostModel())
    status = "PASS" if score > 0 else "FAIL"

    print(f"extracted_knobs={knobs}")
    print(f"answer_score={score:.6f}")
    print(f"status={status}")
    return 0 if score > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
