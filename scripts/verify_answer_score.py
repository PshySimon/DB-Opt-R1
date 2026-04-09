#!/usr/bin/env python3
"""Minimal local verification for answer_score reward extraction."""

from pathlib import Path
import sys

import numpy as np
import torch
from verl import DataProto

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.verl.main_grpo import DBRewardManager
from training.reward_score import compute_score_answer, extract_final_knobs


class _FakeCostModel:
    def predict(self, knobs, hardware):
        return 120.0 if knobs.get("shared_buffers") == "8GB" else 100.0


class _FakeTokenizer:
    pad_token_id = 0

    def __init__(self, solution: str):
        self._solution = solution

    def decode(self, ids, skip_special_tokens=False):
        if isinstance(ids, list) and ids == [self.pad_token_id]:
            return "<pad>"
        return self._solution


def main() -> int:
    solution = (
        "<|im_start|>assistant\n"
        "<think>set knob</think>\n"
        '<tool_call>{"name":"set_knob","arguments":{"knobs":"{\\"shared_buffers\\": \\"8GB\\", \\"work_mem\\": \\"64MB\\"}"}}</tool_call>\n'
        "<|im_end|>"
    )
    ground_truth = {"hardware": {"total_memory_gb": 80.0}}
    cost_model = _FakeCostModel()

    knobs = extract_final_knobs(solution)
    direct_score = compute_score_answer(solution, ground_truth, cost_model=cost_model)
    batch = DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([[1, 2]], dtype=torch.long),
            "responses": torch.tensor([[3, 4, 5]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long),
        },
        non_tensors={
            "reward_model": np.array([{"ground_truth": ground_truth}], dtype=object),
            "data_source": np.array(["verify"], dtype=object),
        },
    )
    _, answer_scores, format_scores = DBRewardManager(
        tokenizer=_FakeTokenizer(solution),
        cost_model=cost_model,
    )(batch)
    e2e_score = answer_scores[0]
    status = "PASS" if direct_score > 0 and e2e_score > 0 else "FAIL"

    print(f"extracted_knobs={knobs}")
    print(f"direct_answer_score={direct_score:.6f}")
    print(f"db_reward_manager_answer_score={e2e_score:.6f}")
    print(f"db_reward_manager_format_score={format_scores[0]:.6f}")
    print(f"status={status}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
