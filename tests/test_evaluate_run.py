import unittest

from evaluate import run as evaluate_run


class DummyCostModel:
    def predict(self, knobs, hw_info):
        base = 100.0
        if knobs.get("shared_buffers") == "8GB":
            base += 20.0
        if knobs.get("workload") == "oltp":
            base += 5.0
        return base


class DummyKnobSpace:
    def get_default_config(self):
        return {"shared_buffers": "4GB"}


class EvaluateRunTest(unittest.TestCase):
    def test_compute_eval_metrics_skip_bo_does_not_run_bo(self):
        trajectories = [
            {
                "env_sample_idx": 0,
                "messages": [
                    {
                        "role": "assistant",
                        "content": (
                            '<tool_call>{"name":"set_knob","arguments":{"knobs":"{\\"shared_buffers\\":\\"8GB\\"}"}}'
                            "</tool_call>"
                        ),
                    }
                ],
            }
        ]
        scenarios = [
            {
                "name": "s0",
                "hardware": {"cpu_count": 8},
                "workload": {"type": "oltp"},
                "knobs": {"shared_buffers": "4GB"},
            }
        ]

        original_bo_search = evaluate_run.bo_search_optimal

        def fail_if_called(*args, **kwargs):
            raise AssertionError("bo_search_optimal should not be called when skip_bo=True")

        evaluate_run.bo_search_optimal = fail_if_called
        try:
            metrics = evaluate_run.compute_eval_metrics(
                trajectories,
                scenarios,
                DummyCostModel(),
                DummyKnobSpace(),
                n_bo_trials=200,
                skip_bo=True,
            )
        finally:
            evaluate_run.bo_search_optimal = original_bo_search

        summary = metrics["summary"]
        self.assertNotIn("avg_imp_vs_optimal_pct", summary)
        self.assertNotIn("avg_gap_closed_pct", summary)
        self.assertEqual(summary["total_episodes"], 1)


if __name__ == "__main__":
    unittest.main()
