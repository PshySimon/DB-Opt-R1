import math
import unittest

from training.reward_score import (
    compute_score_answer,
    compute_score_format_answer,
    extract_best_predict_knobs,
    extract_final_knobs,
)


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

    def test_compute_score_answer_uses_best_predict_not_final_set_knob(self):
        solution = (
            "<|im_start|>assistant\n"
            "<think>set good knob</think>\n"
            '<tool_call>{"name":"set_knob","arguments":{"knobs":"{\\"shared_buffers\\": \\"8GB\\"}"}}</tool_call>\n'
            "<|im_end|>"
            "<|im_start|>user\n"
            '<tool_response>{"success":["shared_buffers"],"pending_restart":[],"failed":[]}</tool_response>'
            "<|im_end|>"
            "<|im_start|>assistant\n"
            "<think>predict good knob</think>\n"
            '<tool_call>{"name":"predict_performance","arguments":{}}</tool_call>'
            "<|im_end|>"
            "<|im_start|>user\n"
            '<tool_response>{"predicted_tps":120.0,"baseline_tps":100.0,"actual_tps":90.0,"improvement_pct":20.0}</tool_response>'
            "<|im_end|>"
            "<|im_start|>assistant\n"
            "<think>set worse final knob</think>\n"
            '<tool_call>{"name":"set_knob","arguments":{"knobs":"{\\"shared_buffers\\": \\"4GB\\"}"}}</tool_call>'
            "<|im_end|>"
            "<|im_start|>user\n"
            '<tool_response>{"success":["shared_buffers"],"pending_restart":[],"failed":[]}</tool_response>'
            "<|im_end|>"
            "<|im_start|>assistant\n"
            "<think>predict worse final knob</think>\n"
            '<tool_call>{"name":"predict_performance","arguments":{}}</tool_call>'
            "<|im_end|>"
            "<|im_start|>user\n"
            '<tool_response>{"predicted_tps":100.0,"baseline_tps":100.0,"actual_tps":90.0,"improvement_pct":0.0}</tool_response>'
            "<|im_end|>"
        )
        ground_truth = {"hardware": {"total_memory_gb": 80.0}}

        score = compute_score_answer(solution, ground_truth, cost_model=None)

        self.assertEqual({"shared_buffers": "8GB"}, extract_best_predict_knobs(solution))
        self.assertAlmostEqual(1.0, score)

    def test_compute_score_format_answer_applies_repeated_tool_call_penalty(self):
        class FakeCostModel:
            def predict(self, knobs, hardware):
                raise AssertionError("compute_score_answer should use predict_performance payloads")

        solution = (
            "<|im_start|>assistant\n"
            "<think>set knob</think>\n"
            '<tool_call>{"name":"set_knob","arguments":{"knobs":"{\\"shared_buffers\\": \\"8GB\\"}"}}</tool_call>\n'
            "<|im_end|>"
            "<|im_start|>user\n"
            '<tool_response>{"success":["shared_buffers"],"pending_restart":[],"failed":[]}</tool_response>'
            "<|im_end|>"
            "<|im_start|>assistant\n"
            "<think>predict</think>\n"
            '<tool_call>{"name":"predict_performance","arguments":{}}</tool_call>'
            "<|im_end|>"
            "<|im_start|>user\n"
            '<tool_response>{"predicted_tps":120.0,"baseline_tps":100.0,"actual_tps":90.0,"improvement_pct":20.0}</tool_response>'
            "<|im_end|>"
        )
        ground_truth = {"hardware": {"total_memory_gb": 80.0}}

        baseline = compute_score_format_answer(solution, ground_truth, cost_model=FakeCostModel())
        penalized = compute_score_format_answer(
            solution,
            ground_truth,
            cost_model=FakeCostModel(),
            termination_reason="repeated_tool_call",
        )

        self.assertTrue(math.isclose(baseline - 0.1, penalized, rel_tol=0.0, abs_tol=1e-6))

    def test_compute_score_format_answer_applies_max_turns_penalty(self):
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

        baseline = compute_score_format_answer(solution, ground_truth, cost_model=FakeCostModel())
        penalized = compute_score_format_answer(
            solution,
            ground_truth,
            cost_model=FakeCostModel(),
            termination_reason="max_turns_reached",
        )

        self.assertTrue(math.isclose(baseline - 0.05, penalized, rel_tol=0.0, abs_tol=1e-6))

if __name__ == "__main__":
    unittest.main()
