import unittest

from data_pipeline.build_rl_dataset_v2 import (
    DEFAULT_OUTPUT_DIR,
    build_bucket_records,
    choose_question,
    classify_env_bucket,
    collect_outlier_envs,
)


class TestBuildRLDatasetV2(unittest.TestCase):
    def test_default_output_dir_is_under_train_v2(self):
        self.assertEqual(DEFAULT_OUTPUT_DIR, "data_pipeline/data/train/v2/rl")

    def test_classify_env_bucket(self):
        self.assertEqual(classify_env_bucket(total_rollouts=10, strong_rollouts=9, weak_rollouts=9), "easy")
        self.assertEqual(classify_env_bucket(total_rollouts=10, strong_rollouts=3, weak_rollouts=3), "frontier")
        self.assertEqual(classify_env_bucket(total_rollouts=10, strong_rollouts=0, weak_rollouts=2), "hard_but_learnable")
        self.assertEqual(classify_env_bucket(total_rollouts=10, strong_rollouts=0, weak_rollouts=0), "all_fail")

    def test_choose_question_prefers_less_reused_prompt(self):
        question = choose_question(
            env_id=7,
            questions={"常见问题", "更独特的问题"},
            question_env_frequency={
                "常见问题": 12,
                "更独特的问题": 1,
            },
            seed=42,
        )
        self.assertEqual(question, "更独特的问题")

    def test_collect_outlier_envs_marks_env_level(self):
        rows = [
            {
                "env": 1,
                "imp": 60.0,
                "final": {"predicted_tps": 100.0, "baseline_tps": 10.0, "actual_tps": 20.0},
                "workload": "mixed",
                "pattern": "memory",
            },
            {
                "env": 2,
                "imp": 20.0,
                "final": {"predicted_tps": 30.0, "baseline_tps": 20.0, "actual_tps": 22.0},
                "workload": "mixed",
                "pattern": "memory",
            },
        ]
        bucket_bounds = {
            ("mixed", "memory"): {
                "pred_over_actual_p99": 2.0,
                "pred_over_baseline_p99": 2.0,
            }
        }
        workload_bounds = {
            "mixed": {
                "pred_over_actual_p99": 2.0,
                "pred_over_baseline_p99": 2.0,
            }
        }
        default_baselines = {1: 11.0}

        outliers = collect_outlier_envs(
            rows=rows,
            default_baselines=default_baselines,
            bucket_bounds=bucket_bounds,
            workload_bounds=workload_bounds,
        )

        self.assertEqual(outliers, {1})

    def test_build_bucket_records_keeps_one_question_per_env(self):
        env_infos = {
            1: {"bucket": "frontier", "questions": {"Q1", "Q2"}},
            2: {"bucket": "frontier", "questions": {"Q3"}},
            3: {"bucket": "hard_but_learnable", "questions": {"Q4"}},
        }
        question_env_frequency = {
            "Q1": 4,
            "Q2": 1,
            "Q3": 1,
            "Q4": 1,
        }

        records = build_bucket_records(
            env_infos=env_infos,
            bucket_name="frontier",
            question_env_frequency=question_env_frequency,
            seed=42,
        )

        self.assertEqual(len(records), 2)
        self.assertEqual({item["env_sample_idx"] for item in records}, {1, 2})
        self.assertEqual(records[0]["messages"][0]["role"], "system")
        self.assertEqual(records[0]["messages"][1]["role"], "user")
        chosen = {item["env_sample_idx"]: item["question"] for item in records}
        self.assertEqual(chosen[1], "Q2")


if __name__ == "__main__":
    unittest.main()
