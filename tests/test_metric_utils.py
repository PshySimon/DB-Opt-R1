import unittest

import numpy as np
import torch
from verl import DataProto

from training.verl.metric_utils import compute_data_metrics


class MetricUtilsTest(unittest.TestCase):
    def test_compute_data_metrics_includes_termination_reason_rates(self):
        batch = DataProto.from_dict(
            tensors={
                "token_level_scores": torch.tensor([[1.0, 0.0], [0.5, 0.0]], dtype=torch.float32),
                "token_level_rewards": torch.tensor([[1.0, 0.0], [0.5, 0.0]], dtype=torch.float32),
                "advantages": torch.tensor([[1.0, 0.0], [0.5, 0.0]], dtype=torch.float32),
                "returns": torch.tensor([[1.0, 0.0], [0.5, 0.0]], dtype=torch.float32),
                "responses": torch.tensor([[11, 12], [21, 22]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
                "format_scores": torch.tensor([0.6, 0.4], dtype=torch.float32),
                "answer_scores": torch.tensor([0.3, 0.1], dtype=torch.float32),
                "turns": torch.tensor([2, 3], dtype=torch.float32),
            },
            non_tensors={
                "termination_reason": np.array(
                    ["finish_tuning", "max_turns_reached"],
                    dtype=object,
                ),
            },
        )

        metrics = compute_data_metrics(batch, use_critic=False)

        self.assertEqual(0.5, metrics["termination/finish_tuning_rate"])
        self.assertEqual(0.5, metrics["termination/max_turns_reached_rate"])


if __name__ == "__main__":
    unittest.main()
