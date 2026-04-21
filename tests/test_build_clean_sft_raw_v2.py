import unittest
from unittest import mock

from data_pipeline import build_clean_sft_raw_v2 as clean_v2


class BuildCleanSftRawV2Test(unittest.TestCase):
    def test_config_effective_false_when_static_knob_not_restarted_before_predict(self):
        record = {
            "env_sample_idx": 0,
            "improvement_pct": 12.0,
            "messages": [
                {
                    "role": "assistant",
                    "content": (
                        '<tool_call>{"name":"set_knob","arguments":{"knobs":"{\\"shared_buffers\\":\\"1GB\\"}"}}'
                        "</tool_call>"
                    ),
                },
                {
                    "role": "assistant",
                    "content": '<tool_call>{"name":"predict_performance","arguments":{}}</tool_call>',
                },
                {
                    "role": "tool",
                    "content": '{"predicted_tps": 120.0, "baseline_tps": 100.0, "actual_tps": 110.0, "improvement_pct": 12.0}',
                },
            ],
        }

        row = clean_v2.extract_row(record, "dummy.jsonl", 1, {0: "mixed"}, {"shared_buffers"})

        self.assertFalse(row["config_effective"])

    def test_extract_row_marks_hallucinated_tool_call(self):
        record = {
            "env_sample_idx": 0,
            "improvement_pct": 8.0,
            "messages": [
                {
                    "role": "assistant",
                    "content": '<tool_call>{"name":"get_hardware_info","arguments":{}}</tool_call>',
                },
                {
                    "role": "assistant",
                    "content": '<tool_call>{"name":"done","arguments":{}}</tool_call>',
                },
                {
                    "role": "tool",
                    "content": '{"predicted_tps": 120.0, "baseline_tps": 100.0, "actual_tps": 110.0, "improvement_pct": 8.0}',
                },
            ],
        }

        row = clean_v2.extract_row(record, "dummy.jsonl", 1, {0: "mixed"}, set(), clean_v2.FAMILY_MAP)

        self.assertTrue(row["has_tool_hallucination"])
        self.assertTrue(row["has_protocol_error"])

    def test_extract_row_marks_protocol_error_from_malformed_tool_call_and_tool_feedback(self):
        record = {
            "env_sample_idx": 0,
            "improvement_pct": 8.0,
            "messages": [
                {
                    "role": "assistant",
                    "content": '<tool_call>{"name":"run_command","arguments":{"cmd":"ls"}',
                },
                {
                    "role": "tool",
                    "content": 'Invalid tool call format. Please use <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>',
                },
            ],
        }

        row = clean_v2.extract_row(record, "dummy.jsonl", 1, {0: "mixed"}, set(), clean_v2.FAMILY_MAP)

        self.assertTrue(row["has_protocol_error"])

    def test_extract_row_uses_best_predict_improvement_instead_of_outer_improvement(self):
        record = {
            "env_sample_idx": 0,
            "improvement_pct": 99.0,
            "messages": [
                {
                    "role": "assistant",
                    "content": '<tool_call>{"name":"predict_performance","arguments":{}}</tool_call>',
                },
                {
                    "role": "tool",
                    "content": '{"predicted_tps": 120.0, "baseline_tps": 100.0, "actual_tps": 110.0, "improvement_pct": 5.0}',
                },
                {
                    "role": "assistant",
                    "content": '<tool_call>{"name":"predict_performance","arguments":{}}</tool_call>',
                },
                {
                    "role": "tool",
                    "content": '{"predicted_tps": 112.0, "baseline_tps": 100.0, "actual_tps": 108.0, "improvement_pct": 2.0}',
                },
                {
                    "role": "assistant",
                    "content": '<tool_call>{"name":"finish_tuning","arguments":{}}</tool_call>',
                },
            ],
        }

        row = clean_v2.extract_row(record, "dummy.jsonl", 1, {0: "mixed"}, set(), clean_v2.FAMILY_MAP)

        self.assertEqual(5.0, row["imp"])
        self.assertEqual(5.0, row["final"]["improvement_pct"])
        self.assertEqual(2, row["predict_count"])
        self.assertTrue(row["has_finish_tuning"])

    def test_high_gain_outlier_due_to_capped_200(self):
        row = {
            "workload": "mixed",
            "pattern": "memory+wal_checkpoint",
            "imp": 200.0,
            "final": {"predicted_tps": 12000.0, "baseline_tps": 3000.0, "actual_tps": 3200.0},
        }

        reasons = clean_v2.classify_high_gain_outlier(
            row=row,
            default_baseline_tps=2500.0,
            bucket_upper={"pred_over_actual_p99": 1.5, "pred_over_baseline_p99": 1.4},
        )

        self.assertIn("cap200", reasons)

    def test_high_gain_outlier_due_to_default_baseline_and_bucket_overshoot(self):
        row = {
            "workload": "write_heavy",
            "pattern": "memory",
            "imp": 80.0,
            "final": {"predicted_tps": 9000.0, "baseline_tps": 3000.0, "actual_tps": 4000.0},
        }

        reasons = clean_v2.classify_high_gain_outlier(
            row=row,
            default_baseline_tps=3500.0,
            bucket_upper={"pred_over_actual_p99": 1.6, "pred_over_baseline_p99": 1.5},
        )

        self.assertIn("baseline_below_default", reasons)
        self.assertIn("pred_over_actual_above_p99", reasons)
        self.assertIn("pred_over_baseline_above_p99", reasons)

    def test_high_gain_not_outlier_when_within_bounds(self):
        row = {
            "workload": "read_only",
            "pattern": "memory",
            "imp": 65.0,
            "final": {"predicted_tps": 83000.0, "baseline_tps": 60000.0, "actual_tps": 65000.0},
        }

        reasons = clean_v2.classify_high_gain_outlier(
            row=row,
            default_baseline_tps=58000.0,
            bucket_upper={"pred_over_actual_p99": 1.35, "pred_over_baseline_p99": 1.40},
        )

        self.assertEqual([], reasons)

    def test_select_best_per_env_keeps_only_highest_improvement(self):
        rows = [
            {"env": 7, "imp": 5.0, "record": {"id": "low"}},
            {"env": 7, "imp": 15.0, "record": {"id": "high"}},
            {"env": 9, "imp": 8.0, "record": {"id": "other"}},
        ]

        kept = clean_v2.select_best_per_env(rows)

        self.assertEqual(2, len(kept))
        by_env = {row["env"]: row for row in kept}
        self.assertEqual("high", by_env[7]["record"]["id"])
        self.assertEqual("other", by_env[9]["record"]["id"])

    @mock.patch.object(clean_v2, "predict_default_baselines", return_value={})
    @mock.patch.object(clean_v2.CostModel, "load")
    def test_compute_clean_dataset_keeps_weak_positive_rows(self, _mock_load, _mock_default_baselines):
        rows = [
            {
                "env": 1,
                "workload": "mixed",
                "pattern": "memory",
                "imp": 2.0,
                "final": {
                    "predicted_tps": 102.0,
                    "baseline_tps": 100.0,
                    "actual_tps": 101.0,
                    "improvement_pct": 2.0,
                },
                "config_effective": True,
                "has_tool_hallucination": False,
                "has_protocol_error": False,
                "has_finish_tuning": True,
                "has_think": True,
                "predict_count": 1,
                "record": {"id": "weak_positive"},
            }
        ]

        kept, stats, _, _ = clean_v2.compute_clean_dataset(
            rows=rows,
            scenarios=[],
            default_knobs={},
            checkpoint_dir="dummy-checkpoint",
        )

        self.assertEqual(1, len(kept))
        self.assertEqual("weak_positive", kept[0]["record"]["id"])
        self.assertEqual(1, stats["positive_total"])
        self.assertEqual(1, stats["keep"])

    @mock.patch.object(clean_v2, "predict_default_baselines", return_value={})
    @mock.patch.object(clean_v2.CostModel, "load")
    def test_compute_clean_dataset_drops_rows_without_finish_tuning(self, _mock_load, _mock_default_baselines):
        rows = [
            {
                "env": 1,
                "workload": "mixed",
                "pattern": "memory",
                "imp": 2.0,
                "final": {
                    "predicted_tps": 102.0,
                    "baseline_tps": 100.0,
                    "actual_tps": 101.0,
                    "improvement_pct": 2.0,
                },
                "config_effective": True,
                "has_tool_hallucination": False,
                "has_protocol_error": False,
                "has_finish_tuning": False,
                "has_think": True,
                "predict_count": 1,
                "record": {"id": "unfinished"},
            },
            {
                "env": 2,
                "workload": "mixed",
                "pattern": "memory",
                "imp": 3.0,
                "final": {
                    "predicted_tps": 103.0,
                    "baseline_tps": 100.0,
                    "actual_tps": 101.0,
                    "improvement_pct": 3.0,
                },
                "config_effective": True,
                "has_tool_hallucination": False,
                "has_protocol_error": False,
                "has_finish_tuning": True,
                "has_think": True,
                "predict_count": 1,
                "record": {"id": "finished"},
            },
        ]

        kept, stats, _, _ = clean_v2.compute_clean_dataset(
            rows=rows,
            scenarios=[],
            default_knobs={},
            checkpoint_dir="dummy-checkpoint",
        )

        self.assertEqual(["finished"], [row["record"]["id"] for row in kept])
        self.assertEqual(1, stats["drop_no_finish_tuning"])

    @mock.patch.object(clean_v2, "predict_default_baselines", return_value={})
    @mock.patch.object(clean_v2.CostModel, "load")
    def test_compute_clean_dataset_drops_rows_without_think(self, _mock_load, _mock_default_baselines):
        rows = [
            {
                "env": 1,
                "workload": "mixed",
                "pattern": "memory",
                "imp": 2.0,
                "final": {
                    "predicted_tps": 102.0,
                    "baseline_tps": 100.0,
                    "actual_tps": 101.0,
                    "improvement_pct": 2.0,
                },
                "config_effective": True,
                "has_tool_hallucination": False,
                "has_protocol_error": False,
                "has_finish_tuning": True,
                "has_think": False,
                "predict_count": 1,
                "record": {"id": "no_think"},
            },
            {
                "env": 2,
                "workload": "mixed",
                "pattern": "memory",
                "imp": 3.0,
                "final": {
                    "predicted_tps": 103.0,
                    "baseline_tps": 100.0,
                    "actual_tps": 101.0,
                    "improvement_pct": 3.0,
                },
                "config_effective": True,
                "has_tool_hallucination": False,
                "has_protocol_error": False,
                "has_finish_tuning": True,
                "has_think": True,
                "predict_count": 1,
                "record": {"id": "with_think"},
            },
        ]

        kept, stats, _, _ = clean_v2.compute_clean_dataset(
            rows=rows,
            scenarios=[],
            default_knobs={},
            checkpoint_dir="dummy-checkpoint",
        )

        self.assertEqual(["with_think"], [row["record"]["id"] for row in kept])
        self.assertEqual(1, stats["drop_no_think"])


if __name__ == "__main__":
    unittest.main()
