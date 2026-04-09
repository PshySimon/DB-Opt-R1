import unittest

from training.reward_score import extract_final_knobs


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


if __name__ == "__main__":
    unittest.main()
