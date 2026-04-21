import json
import tempfile
import unittest
from pathlib import Path

from data_pipeline import build_sft_manifests_v2 as manifests_v2


def _assistant(tool_name: str) -> dict:
    return {
        "role": "assistant",
        "content": (
            "<think>分析</think>\n"
            f'<tool_call>{{"name":"{tool_name}","arguments":{{}}}}</tool_call>'
        ),
    }


def _tool_predict(improvement_pct: float) -> dict:
    return {
        "role": "tool",
        "content": json.dumps(
            {
                "predicted_tps": 100.0 + improvement_pct,
                "baseline_tps": 100.0,
                "actual_tps": 100.0,
                "improvement_pct": improvement_pct,
            },
            ensure_ascii=False,
        ),
    }


class BuildSftManifestsV2Test(unittest.TestCase):
    def test_build_label_rows_extracts_shape_gain_and_depth(self):
        scenarios = [
            {"name": "a", "variant": 0, "workload": {"type": "read_only"}},
            {"name": "b", "variant": 0, "workload": {"type": "mixed"}},
        ]
        rows = [
            {
                "env_sample_idx": 0,
                "improvement_pct": 0.8,
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "q0"},
                    _assistant("predict_performance"),
                    _tool_predict(0.8),
                    _assistant("finish_tuning"),
                ],
            },
            {
                "env_sample_idx": 1,
                "improvement_pct": 6.0,
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "q1"},
                    _assistant("predict_performance"),
                    _tool_predict(0.0),
                    _assistant("set_knob"),
                    _assistant("predict_performance"),
                    _tool_predict(6.0),
                    _assistant("finish_tuning"),
                ],
            },
        ]

        labels = manifests_v2.build_label_rows(rows, scenarios)
        by_env = {item["env_sample_idx"]: item for item in labels}

        self.assertEqual("direct_success", by_env[0]["shape"])
        self.assertEqual("0_1", by_env[0]["gain_bucket"])
        self.assertEqual("main", by_env[0]["depth_bucket"])

        self.assertEqual("retry_success", by_env[1]["shape"])
        self.assertEqual("3_10", by_env[1]["gain_bucket"])
        self.assertEqual("tail", by_env[1]["depth_bucket"])

    def test_build_manifest_sets_respects_sizes_and_balanced_gain(self):
        labels = []
        env = 0
        workloads = ["read_only", "mixed"]
        gains = ["0_1", "1_3", "3_10", "10_50"]
        for gain in gains:
            for workload in workloads:
                for _ in range(4):
                    labels.append(
                        {
                            "env_sample_idx": env,
                            "name": "demo",
                            "variant": 0,
                            "workload": workload,
                            "gain_bucket": gain,
                            "depth_bucket": "main",
                            "shape": "direct_success",
                        }
                    )
                    env += 1
        for workload in workloads:
            for _ in range(2):
                labels.append(
                    {
                        "env_sample_idx": env,
                        "name": "demo",
                        "variant": 0,
                        "workload": workload,
                        "gain_bucket": "3_10",
                        "depth_bucket": "tail",
                        "shape": "retry_success",
                    }
                )
                env += 1

        manifests = manifests_v2.build_manifest_sets(
            labels=labels,
            abc_size=8,
            seed=7,
        )

        self.assertEqual(8, len(manifests["sft_manifest_a0_direct_only.jsonl"]))
        self.assertEqual(8, len(manifests["sft_manifest_a1_full3k.jsonl"]))
        self.assertEqual(8, len(manifests["sft_manifest_b1_depth_trimmed.jsonl"]))
        self.assertEqual(8, len(manifests["sft_manifest_b2_depth_full.jsonl"]))
        self.assertEqual(8, len(manifests["sft_manifest_c1_gain_natural.jsonl"]))
        self.assertEqual(8, len(manifests["sft_manifest_c2_gain_balanced.jsonl"]))

        a1_shapes = {item["shape"] for item in manifests["sft_manifest_a1_full3k.jsonl"]}
        self.assertIn("retry_success", a1_shapes)

        balanced_counts = {}
        for item in manifests["sft_manifest_c2_gain_balanced.jsonl"]:
            balanced_counts[item["gain_bucket"]] = balanced_counts.get(item["gain_bucket"], 0) + 1
        self.assertEqual({"0_1": 2, "1_3": 2, "3_10": 2, "10_50": 2}, balanced_counts)

    def test_write_manifests_outputs_jsonl_and_stats(self):
        manifests = {
            "one.jsonl": [
                {"env_sample_idx": 1, "shape": "direct_success"},
                {"env_sample_idx": 2, "shape": "retry_success"},
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            manifests_v2.write_manifests(manifests, out_dir)

            rows = [
                json.loads(line)
                for line in (out_dir / "one.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual([1, 2], [item["env_sample_idx"] for item in rows])

            stats = json.loads((out_dir / "manifest_stats.json").read_text(encoding="utf-8"))
            self.assertEqual(2, stats["one.jsonl"]["rows"])


if __name__ == "__main__":
    unittest.main()
