import json
import tempfile
import unittest
from pathlib import Path

from data_pipeline import build_sft_experiment_dataset_v2 as builder


class BuildSftExperimentDatasetV2Test(unittest.TestCase):
    def test_select_rows_by_manifest_preserves_manifest_order(self):
        rows = [
            {"env_sample_idx": 1, "messages": [{"role": "user", "content": "a"}]},
            {"env_sample_idx": 2, "messages": [{"role": "user", "content": "b"}]},
            {"env_sample_idx": 3, "messages": [{"role": "user", "content": "c"}]},
        ]
        manifest_rows = [
            {"env_sample_idx": 3, "name": "x"},
            {"env_sample_idx": 1, "name": "y"},
        ]

        selected = builder.select_rows_by_manifest(rows, manifest_rows)

        self.assertEqual([3, 1], [row["env_sample_idx"] for row in selected])

    def test_select_rows_by_manifest_raises_when_env_missing(self):
        rows = [{"env_sample_idx": 1, "messages": []}]
        manifest_rows = [{"env_sample_idx": 2}]

        with self.assertRaises(ValueError):
            builder.select_rows_by_manifest(rows, manifest_rows)

    def test_write_outputs_writes_jsonl_and_stats(self):
        rows = [
            {"env_sample_idx": 7, "messages": [{"role": "user", "content": "x"}], "improvement_pct": 3.2},
            {"env_sample_idx": 8, "messages": [{"role": "user", "content": "y"}], "improvement_pct": 7.8},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_jsonl = Path(tmpdir) / "train.jsonl"
            stats_json = Path(tmpdir) / "train_stats.json"

            builder.write_outputs(rows, output_jsonl, stats_json)

            written_rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
            stats = json.loads(stats_json.read_text(encoding="utf-8"))

        self.assertEqual(2, len(written_rows))
        self.assertEqual([7, 8], [row["env_sample_idx"] for row in written_rows])
        self.assertEqual(2, stats["rows"])
        self.assertEqual(2, stats["unique_envs"])
        self.assertEqual("7", stats["min_env_sample_idx"])
        self.assertEqual("8", stats["max_env_sample_idx"])


if __name__ == "__main__":
    unittest.main()
