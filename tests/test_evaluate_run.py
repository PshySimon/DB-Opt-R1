import unittest
from unittest.mock import patch

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
    def test_compute_eval_metrics_uses_best_predict_from_trajectory(self):
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
                    },
                    {
                        "role": "tool",
                        "content": (
                            '{'
                            '"predicted_tps": 130.0, '
                            '"baseline_tps": 105.0, '
                            '"improvement_pct": 23.81'
                            '}'
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": (
                            '<tool_call>{"name":"set_knob","arguments":{"knobs":"{\\"shared_buffers\\":\\"4GB\\"}"}}'
                            "</tool_call>"
                        ),
                    },
                    {
                        "role": "tool",
                        "content": (
                            '{'
                            '"predicted_tps": 105.0, '
                            '"baseline_tps": 105.0, '
                            '"improvement_pct": 0.0'
                            '}'
                        ),
                    },
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

        metrics = evaluate_run.compute_eval_metrics(
            trajectories,
            scenarios,
            DummyCostModel(),
            DummyKnobSpace(),
            n_bo_trials=200,
            skip_bo=True,
        )

        episode = metrics["per_episode"][0]
        self.assertEqual(episode["scenario_tps"], 105.0)
        self.assertEqual(episode["model_tps"], 130.0)
        self.assertGreater(episode["imp_vs_scenario_pct"], 20.0)

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

    def test_compute_eval_metrics_logs_progress(self):
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
            for _ in range(3)
        ]
        scenarios = [
            {
                "name": "s0",
                "hardware": {"cpu_count": 8},
                "workload": {"type": "oltp"},
                "knobs": {"shared_buffers": "4GB"},
            }
        ]

        with patch.object(evaluate_run.logger, "info") as mock_info:
            evaluate_run.compute_eval_metrics(
                trajectories,
                scenarios,
                DummyCostModel(),
                DummyKnobSpace(),
                n_bo_trials=200,
                skip_bo=True,
                progress_interval=2,
            )

        joined = "\n".join(str(call.args[0]) for call in mock_info.call_args_list if call.args)
        self.assertIn("评估进度", joined)

    def test_compute_eval_metrics_includes_termination_reason_summary(self):
        trajectories = [
            {
                "env_sample_idx": 0,
                "termination_reason": "finish_tuning",
                "messages": [
                    {
                        "role": "assistant",
                        "content": (
                            '<tool_call>{"name":"set_knob","arguments":{"knobs":"{\\"shared_buffers\\":\\"8GB\\"}"}}'
                            "</tool_call>"
                        ),
                    }
                ],
            },
            {
                "env_sample_idx": 0,
                "termination_reason": "repeated_tool_call",
                "messages": [
                    {
                        "role": "assistant",
                        "content": (
                            '<tool_call>{"name":"set_knob","arguments":{"knobs":"{\\"shared_buffers\\":\\"4GB\\"}"}}'
                            "</tool_call>"
                        ),
                    }
                ],
            },
        ]
        scenarios = [
            {
                "name": "s0",
                "hardware": {"cpu_count": 8},
                "workload": {"type": "oltp"},
                "knobs": {"shared_buffers": "4GB"},
            }
        ]

        metrics = evaluate_run.compute_eval_metrics(
            trajectories,
            scenarios,
            DummyCostModel(),
            DummyKnobSpace(),
            n_bo_trials=200,
            skip_bo=True,
        )

        summary = metrics["summary"]
        self.assertEqual(summary["termination_reason_rate"]["finish_tuning"], 0.5)
        self.assertEqual(summary["termination_reason_rate"]["repeated_tool_call"], 0.5)
        self.assertGreater(
            summary["avg_imp_vs_scenario_pct_by_termination_reason"]["finish_tuning"],
            summary["avg_imp_vs_scenario_pct_by_termination_reason"]["repeated_tool_call"],
        )

    def test_compute_eval_metrics_caches_default_tps_by_hw_and_workload(self):
        class CountingCostModel(DummyCostModel):
            def __init__(self):
                self.default_calls = 0

            def predict(self, knobs, hw_info):
                if knobs.get("shared_buffers") == "4GB":
                    self.default_calls += 1
                return super().predict(knobs, hw_info)

        cost_model = CountingCostModel()
        trajectories = [
            {
                "env_sample_idx": 0,
                "messages": [
                    {
                        "role": "tool",
                        "content": (
                            '{'
                            '"predicted_tps": 130.0, '
                            '"baseline_tps": 105.0, '
                            '"improvement_pct": 23.81'
                            '}'
                        ),
                    }
                ],
            },
            {
                "env_sample_idx": 1,
                "messages": [
                    {
                        "role": "tool",
                        "content": (
                            '{'
                            '"predicted_tps": 130.0, '
                            '"baseline_tps": 105.0, '
                            '"improvement_pct": 23.81'
                            '}'
                        ),
                    }
                ],
            },
        ]
        scenarios = [
            {
                "name": "s0",
                "hardware": {"cpu_count": 8},
                "workload": {"type": "oltp"},
                "knobs": {"shared_buffers": "4GB"},
            },
            {
                "name": "s1",
                "hardware": {"cpu_count": 8},
                "workload": {"type": "oltp"},
                "knobs": {"shared_buffers": "4GB"},
            },
        ]

        evaluate_run.compute_eval_metrics(
            trajectories,
            scenarios,
            cost_model,
            DummyKnobSpace(),
            n_bo_trials=200,
            skip_bo=True,
        )

        self.assertEqual(cost_model.default_calls, 1)


if __name__ == "__main__":
    unittest.main()
