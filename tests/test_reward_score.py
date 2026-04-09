import subprocess
import sys
import unittest
from pathlib import Path

from training.reward_score import compute_score_answer, extract_final_knobs


ROOT = Path(__file__).resolve().parents[1]


class RewardScoreTest(unittest.TestCase):
    def test_extract_final_knobs_supports_set_knob_json_argument(self):
        solution = (
            "<|im_start|>assistant\n"
            "<think>set knob</think>\n"
            '<tool_call>{"name":"set_knob","arguments":{"knobs":"{\\"shared_buffers\\": \\"8GB\\", \\"work_mem\\": \\"64MB\\"}"}}</tool_call>\n'
            "<|im_end|>"
        )

        knobs = extract_final_knobs(solution)

        self.assertEqual(
            knobs,
            {
                "shared_buffers": "8GB",
                "work_mem": "64MB",
            },
        )

    def test_compute_score_answer_returns_positive_for_valid_set_knob_payload(self):
        class FakeCostModel:
            def predict(self, knobs, hardware):
                return 120.0 if knobs.get("shared_buffers") == "8GB" else 100.0

        solution = (
            "<|im_start|>assistant\n"
            "<think>set knob</think>\n"
            '<tool_call>{"name":"set_knob","arguments":{"knobs":"{\\"shared_buffers\\": \\"8GB\\"}"}}</tool_call>\n'
            "<|im_end|>"
        )
        ground_truth = {"hardware": {"total_memory_gb": 80.0}}

        score = compute_score_answer(solution, ground_truth, cost_model=FakeCostModel())

        self.assertGreater(score, 0.0)

    def test_verify_answer_score_script_reports_positive_score(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "verify_answer_score.py")],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("answer_score=", result.stdout)
        self.assertIn("status=PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
