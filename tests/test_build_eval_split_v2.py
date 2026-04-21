import json
import tempfile
import unittest
from pathlib import Path

from data_pipeline import build_eval_split_v2 as split_v2


def _assistant_message(tool_name: str) -> dict:
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


class BuildEvalSplitV2Test(unittest.TestCase):
    def test_select_eval_groups_keeps_whole_groups_and_covers_all_names(self):
        scenarios = [
            {"name": "alpha", "variant": 0, "workload": {"type": "read_only"}},
            {"name": "alpha", "variant": 0, "workload": {"type": "read_only"}},
            {"name": "alpha", "variant": 1, "workload": {"type": "mixed"}},
            {"name": "beta", "variant": 0, "workload": {"type": "write_heavy"}},
            {"name": "beta", "variant": 0, "workload": {"type": "write_heavy"}},
            {"name": "gamma", "variant": 0, "workload": {"type": "high_concurrency"}},
        ]
        rows = [
            {
                "env_sample_idx": idx,
                "improvement_pct": imp,
                "question": f"q-{idx}",
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": f"q-{idx}"},
                    _assistant_message("predict_performance"),
                    _tool_predict(imp),
                    _assistant_message("finish_tuning"),
                ],
            }
            for idx, imp in enumerate([0.8, 1.2, 4.0, 2.0, 2.5, 12.0])
        ]

        groups = split_v2.build_group_entries(rows, scenarios)
        selected = split_v2.select_eval_groups(groups, target_rows=4, seed=7)

        selected_envs = {env for group in selected for env in group["env_ids"]}
        selected_names = {group["name"] for group in selected}

        self.assertEqual({"alpha", "beta", "gamma"}, selected_names)

        group_to_envs = {
            (group["name"], group["variant"], group["workload"]): set(group["env_ids"])
            for group in groups
        }
        for key, envs in group_to_envs.items():
            overlap = envs & selected_envs
            self.assertTrue(overlap == set() or overlap == envs, key)

    def test_write_eval_outputs_reindexes_env_ids(self):
        scenarios = [
            {"name": "alpha", "variant": 0, "workload": {"type": "read_only"}},
            {"name": "beta", "variant": 0, "workload": {"type": "mixed"}},
        ]
        rows = [
            {
                "env_sample_idx": 0,
                "improvement_pct": 1.2,
                "question": "question-a",
                "messages": [
                    {"role": "system", "content": "sys-a"},
                    {"role": "user", "content": "question-a"},
                    _assistant_message("predict_performance"),
                    _tool_predict(1.2),
                    _assistant_message("finish_tuning"),
                ],
            },
            {
                "env_sample_idx": 1,
                "improvement_pct": 6.0,
                "question": "question-b",
                "messages": [
                    {"role": "system", "content": "sys-b"},
                    {"role": "user", "content": "question-b"},
                    _assistant_message("predict_performance"),
                    _tool_predict(6.0),
                    _assistant_message("finish_tuning"),
                ],
            },
        ]

        groups = split_v2.build_group_entries(rows, scenarios)
        selected = groups

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            split_v2.write_eval_outputs(
                selected_groups=selected,
                rows=rows,
                scenarios=scenarios,
                output_dir=out_dir,
            )

            scene_items = json.loads((out_dir / "collected_eval_v2.json").read_text(encoding="utf-8"))
            self.assertEqual(2, len(scene_items))
            self.assertEqual("alpha", scene_items[0]["name"])
            self.assertEqual("beta", scene_items[1]["name"])

            question_rows = [
                json.loads(line)
                for line in (out_dir / "eval_trajectories_v2.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual([0, 1], [item["env_sample_idx"] for item in question_rows])
            self.assertEqual(["question-a", "question-b"], [item["question"] for item in question_rows])
            self.assertEqual(["sys-a", "sys-b"], [item["messages"][0]["content"] for item in question_rows])

            index_rows = [
                json.loads(line)
                for line in (out_dir / "eval_index_map.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual([0, 1], [item["eval_env_sample_idx"] for item in index_rows])
            self.assertEqual([0, 1], [item["source_env_sample_idx"] for item in index_rows])


if __name__ == "__main__":
    unittest.main()
