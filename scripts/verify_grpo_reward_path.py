#!/usr/bin/env python3
"""Verify the current GRPO reward path produces non-zero answer scores."""

from pathlib import Path
import sys

import numpy as np
import torch
from verl import DataProto

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.verl.main_grpo import DBRewardManager


SOLUTION = (
    "<|im_start|>assistant\n"
    "<think>set knob</think>\n"
    '<tool_call>{"name":"set_knob","arguments":{"knobs":"{\\"shared_buffers\\": \\"8GB\\", \\"work_mem\\": \\"64MB\\"}"}}</tool_call>\n'
    "<|im_end|>"
)


class _FakeTokenizer:
    pad_token_id = 0

    def decode(self, ids, skip_special_tokens=False):
        if isinstance(ids, list) and ids == [self.pad_token_id]:
            return "<pad>"
        return SOLUTION


class _FakeCostModel:
    def predict(self, knobs, hardware):
        return 120.0 if knobs.get("shared_buffers") == "8GB" else 100.0


def main() -> int:
    batch = DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([[1, 2]], dtype=torch.long),
            "responses": torch.tensor([[3, 4, 5]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long),
        },
        non_tensors={
            "reward_model": np.array(
                [{"ground_truth": {"hardware": {"total_memory_gb": 80.0}}}],
                dtype=object,
            ),
            "data_source": np.array(["verify"], dtype=object),
        },
    )

    reward_tensor, answer_scores, format_scores = DBRewardManager(
        tokenizer=_FakeTokenizer(),
        cost_model=_FakeCostModel(),
    )(batch)

    answer_score = float(answer_scores[0])
    format_score = float(format_scores[0])
    final_reward = float(reward_tensor[0, -1].item())
    status = "PASS" if answer_score > 0.0 and final_reward > format_score else "FAIL"

    print(f"db_reward_manager_answer_score={answer_score:.6f}")
    print(f"db_reward_manager_format_score={format_score:.6f}")
    print(f"db_reward_manager_final_reward={final_reward:.6f}")
    print(f"status={status}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
