import json
import tempfile
import unittest
from pathlib import Path

from core.db.knob_validator import KnobValidator
from data_pipeline.build_rlvr_probe_configs import _load_cap_records


class BuildRlvrProbeConfigsTest(unittest.TestCase):
    def test_load_cap_records_uses_best_predict_knobs_not_top_level_final_knobs(self):
        row = {
            "train_row_idx": 1,
            "orig_env_sample_idx": 2,
            "rollout_idx": 3,
            "improvement_pct": 20.0,
            "final_knobs": {"shared_buffers": "4GB"},
            "messages": [
                {
                    "role": "assistant",
                    "content": '<tool_call>{"name":"set_knob","arguments":{"knobs":"{\\"shared_buffers\\":\\"8GB\\"}"}}</tool_call>',
                },
                {"role": "tool", "content": '{"success":["shared_buffers"],"pending_restart":[],"failed":[]}'},
                {
                    "role": "assistant",
                    "content": '<tool_call>{"name":"predict_performance","arguments":{}}</tool_call>',
                },
                {
                    "role": "tool",
                    "content": '{"predicted_tps":130.0,"baseline_tps":100.0,"actual_tps":90.0,"improvement_pct":30.0}',
                },
                {
                    "role": "assistant",
                    "content": '<tool_call>{"name":"set_knob","arguments":{"knobs":"{\\"shared_buffers\\":\\"4GB\\"}"}}</tool_call>',
                },
                {"role": "tool", "content": '{"success":["shared_buffers"],"pending_restart":[],"failed":[]}'},
                {
                    "role": "assistant",
                    "content": '<tool_call>{"name":"predict_performance","arguments":{}}</tool_call>',
                },
                {
                    "role": "tool",
                    "content": '{"predicted_tps":100.0,"baseline_tps":100.0,"actual_tps":90.0,"improvement_pct":0.0}',
                },
            ],
        }

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "rollouts.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            records = _load_cap_records(
                path,
                threshold=0.0,
                validator=KnobValidator("configs/knob_space.yaml"),
            )

        self.assertEqual(1, len(records))
        self.assertEqual({"shared_buffers": "8GB"}, records[0]["knobs"])
        self.assertEqual(130.0, records[0]["predicted_tps"])


if __name__ == "__main__":
    unittest.main()
