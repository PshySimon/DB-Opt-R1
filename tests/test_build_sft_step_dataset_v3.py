import json
import tempfile
import unittest
from pathlib import Path

from data_pipeline import build_sft_step_dataset_v3 as builder


def _assistant(content: str) -> dict:
    return {"role": "assistant", "content": content}


class BuildSftStepDatasetV3Test(unittest.TestCase):
    def test_build_step_rows_splits_trajectory_and_removes_history_think(self):
        row = {
            "env_sample_idx": 7,
            "improvement_pct": 3.5,
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "task"},
                _assistant('<think>first thought</think>\n<tool_call>{"name":"get_hardware_info","arguments":{}}</tool_call>'),
                {"role": "tool", "content": '{"cpu": 8}'},
                _assistant('<think>second thought</think>\n<tool_call>{"name":"finish_tuning","arguments":{}}</tool_call>'),
            ],
        }

        rows, stats = builder.build_step_rows([row], history_turns=4, remove_think_history=True)

        self.assertEqual(2, len(rows))
        self.assertEqual("task", rows[0]["instruction"])
        self.assertEqual([], rows[0]["history"])
        self.assertIn("<think>first thought</think>", rows[0]["output"])
        self.assertEqual("get_hardware_info", rows[0]["meta"]["target_tool"])
        self.assertFalse(rows[0]["meta"]["is_final_step"])

        self.assertEqual("<tool_response>\n{\"cpu\": 8}\n</tool_response>", rows[1]["instruction"])
        self.assertEqual(
            [["task", '<tool_call>{"name":"get_hardware_info","arguments":{}}</tool_call>']],
            rows[1]["history"],
        )
        self.assertEqual("finish_tuning", rows[1]["meta"]["target_tool"])
        self.assertTrue(rows[1]["meta"]["is_final_step"])
        self.assertEqual(2, stats["output_rows"])
        self.assertEqual(1, stats["finish_steps"])

    def test_build_step_rows_can_oversample_finish_steps(self):
        row = {
            "env_sample_idx": 9,
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "task"},
                _assistant('<think>x</think><tool_call>{"name":"finish_tuning","arguments":{}}</tool_call>'),
            ],
        }

        rows, stats = builder.build_step_rows([row], finish_oversample=3)

        self.assertEqual(3, len(rows))
        self.assertEqual([0, 1, 2], [r["meta"]["oversample_idx"] for r in rows])
        self.assertEqual(2, stats["oversampled_finish_extra_rows"])

    def test_write_outputs_writes_llamafactory_json_and_stats(self):
        rows = [
            {
                "system": "system",
                "instruction": "task",
                "input": "",
                "output": "answer",
                "history": [],
                "meta": {"target_tool": "finish_tuning", "is_final_step": True},
            }
        ]
        stats = {"output_rows": 1}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = Path(tmpdir) / "train.json"
            stats_json = Path(tmpdir) / "stats.json"

            builder.write_outputs(rows, stats, output_json, stats_json)

            written = json.loads(output_json.read_text(encoding="utf-8"))
            written_stats = json.loads(stats_json.read_text(encoding="utf-8"))

        self.assertEqual(rows, written)
        self.assertEqual(1, written_stats["output_rows"])

    def test_write_outputs_can_shard_jsonl(self):
        rows = [
            {
                "system": "system",
                "instruction": f"task {idx}",
                "input": "",
                "output": "answer" * 100,
                "history": [],
                "meta": {"idx": idx},
            }
            for idx in range(20)
        ]
        stats = {"output_rows": len(rows)}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_jsonl = Path(tmpdir) / "train.jsonl"
            stats_json = Path(tmpdir) / "stats.json"

            builder.write_outputs(rows, stats, output_jsonl, stats_json, max_shard_mb=1)

            written = []
            shard_paths = sorted(Path(tmpdir).glob("train-*-of-*.jsonl"))
            for shard_path in shard_paths:
                with shard_path.open(encoding="utf-8") as f:
                    written.extend(json.loads(line) for line in f if line.strip())
            written_stats = json.loads(stats_json.read_text(encoding="utf-8"))

        self.assertEqual(rows, written)
        self.assertEqual([str(path) for path in shard_paths], written_stats["output_jsonl_shards"])


if __name__ == "__main__":
    unittest.main()
