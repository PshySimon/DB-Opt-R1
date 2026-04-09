import json
import tempfile
import unittest
from pathlib import Path

from data_pipeline.preprocess_grpo_from_sft import (
    build_grpo_records,
    load_aligned_scenarios,
    sample_records_per_scene,
    split_records,
)


class PreprocessGrpoFromSFTTest(unittest.TestCase):
    def test_build_grpo_records_uses_system_prompt_question_and_hardware(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scenario_path = root / "collected_a.json"
            scenario_path.write_text(
                json.dumps(
                    [
                        {
                            "name": "s0",
                            "source": "llm_generated",
                            "hardware": {"cpu_count": 8, "total_memory_gb": 14.5},
                            "workload": {"tps_current": 100.0},
                            "knobs": {"shared_buffers": "4GB"},
                            "difficulty": "medium",
                            "description": "demo",
                        },
                        {
                            "name": "s1",
                            "source": "random_sampled",
                            "hardware": {"cpu_count": 16, "total_memory_gb": 32.0},
                            "workload": {"tps_current": 200.0},
                            "knobs": {"shared_buffers": "8GB"},
                            "difficulty": "medium",
                            "description": "demo2",
                        },
                    ]
                ),
                encoding="utf-8",
            )

            scenarios = load_aligned_scenarios([str(scenario_path)], source_filter="llm_generated")
            self.assertEqual(len(scenarios), 1)

            records = build_grpo_records(
                [
                    {
                        "env_sample_idx": 0,
                        "question": "请帮我看看为什么 IO 抖动",
                        "messages": [
                            {"role": "system", "content": "new system prompt"},
                            {"role": "user", "content": "old user prompt"},
                        ],
                    },
                    {
                        "env_sample_idx": 1,
                        "question": "这个会被过滤掉",
                        "messages": [
                            {"role": "system", "content": "ignored"},
                            {"role": "user", "content": "ignored"},
                        ],
                    },
                ],
                scenarios,
            )

            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record["prompt"][0]["content"], "new system prompt")
            self.assertEqual(record["prompt"][1]["content"], "请帮我看看为什么 IO 抖动")
            self.assertEqual(record["reward_model"]["ground_truth"]["scenario_idx"], 0)
            self.assertEqual(
                record["reward_model"]["ground_truth"]["hardware"],
                {"cpu_count": 8, "total_memory_gb": 14.5},
            )

    def test_build_grpo_records_falls_back_to_first_user_message(self):
        scenarios = [type("Scenario", (), {"hardware": {"cpu_count": 4}})()]
        records = build_grpo_records(
            [
                {
                    "env_sample_idx": 0,
                    "messages": [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "来自 messages 的问题"},
                    ],
                }
            ],
            scenarios,
        )
        self.assertEqual(records[0]["prompt"][1]["content"], "来自 messages 的问题")

    def test_split_records_respects_ratio(self):
        records = [{"i": i} for i in range(10)]
        train_records, val_records = split_records(records, val_ratio=0.2, seed=123)
        self.assertEqual(len(train_records), 8)
        self.assertEqual(len(val_records), 2)

    def test_sample_records_per_scene_keeps_one_record_per_scene_by_default(self):
        records = [
            {
                "prompt": [{"role": "system", "content": "sys"}, {"role": "user", "content": "q0"}],
                "reward_model": {"ground_truth": {"scenario_idx": 0, "hardware": {}}},
                "data_source": "db_tuning",
            },
            {
                "prompt": [{"role": "system", "content": "sys"}, {"role": "user", "content": "q1"}],
                "reward_model": {"ground_truth": {"scenario_idx": 0, "hardware": {}}},
                "data_source": "db_tuning",
            },
            {
                "prompt": [{"role": "system", "content": "sys"}, {"role": "user", "content": "q2"}],
                "reward_model": {"ground_truth": {"scenario_idx": 1, "hardware": {}}},
                "data_source": "db_tuning",
            },
            {
                "prompt": [{"role": "system", "content": "sys"}, {"role": "user", "content": "q3"}],
                "reward_model": {"ground_truth": {"scenario_idx": 1, "hardware": {}}},
                "data_source": "db_tuning",
            },
        ]

        sampled = sample_records_per_scene(records, questions_per_scene=1, seed=7)

        self.assertEqual(len(sampled), 2)
        kept_scenarios = [r["reward_model"]["ground_truth"]["scenario_idx"] for r in sampled]
        self.assertEqual(sorted(kept_scenarios), [0, 1])

    def test_sample_records_per_scene_can_keep_multiple_records(self):
        records = [
            {
                "prompt": [{"role": "system", "content": "sys"}, {"role": "user", "content": f"q{i}"}],
                "reward_model": {"ground_truth": {"scenario_idx": 0, "hardware": {}}},
                "data_source": "db_tuning",
            }
            for i in range(3)
        ]
        sampled = sample_records_per_scene(records, questions_per_scene=2, seed=1)
        self.assertEqual(len(sampled), 2)


if __name__ == "__main__":
    unittest.main()
