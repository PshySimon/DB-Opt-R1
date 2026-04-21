import json
import tempfile
import unittest
from pathlib import Path

from data_pipeline import build_train_split_v2 as train_v2


class BuildTrainSplitV2Test(unittest.TestCase):
    def test_filter_train_rows_excludes_eval_envs(self):
        rows = [
            {"env_sample_idx": 0, "improvement_pct": 1.0},
            {"env_sample_idx": 1, "improvement_pct": 2.0},
            {"env_sample_idx": 2, "improvement_pct": 3.0},
        ]

        kept = train_v2.filter_train_rows(rows, eval_env_ids={1})

        self.assertEqual([0, 2], [row["env_sample_idx"] for row in kept])

    def test_write_train_outputs_saves_rows_and_ids(self):
        rows = [
            {"env_sample_idx": 2, "improvement_pct": 3.0, "question": "q2"},
            {"env_sample_idx": 5, "improvement_pct": 5.0, "question": "q5"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            train_v2.write_train_outputs(
                rows=rows,
                output_dir=out_dir,
                output_name="train.jsonl",
            )

            saved_rows = [
                json.loads(line)
                for line in (out_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual([2, 5], [row["env_sample_idx"] for row in saved_rows])

            saved_env_ids = json.loads((out_dir / "train_env_ids.json").read_text(encoding="utf-8"))
            self.assertEqual([2, 5], saved_env_ids)

            stats = json.loads((out_dir / "train_stats.json").read_text(encoding="utf-8"))
            self.assertEqual(2, stats["rows"])
            self.assertEqual(2, stats["unique_envs"])


if __name__ == "__main__":
    unittest.main()
