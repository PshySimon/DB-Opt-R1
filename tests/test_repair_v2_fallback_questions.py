import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "repair_v2_fallback_questions.py"
SPEC = importlib.util.spec_from_file_location("repair_v2_fallback_questions", SCRIPT_PATH)
repair = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(repair)


class RepairV2FallbackQuestionsTest(unittest.TestCase):
    def test_choose_existing_question_prefers_less_reused_question(self):
        alternatives_by_env = {
            7: {"常见问题", "更独特的问题"},
        }
        question_env_frequency = {
            "常见问题": 10,
            "更独特的问题": 1,
        }

        chosen = repair.choose_existing_question(
            env_id=7,
            alternatives_by_env=alternatives_by_env,
            question_env_frequency=question_env_frequency,
        )

        self.assertEqual("更独特的问题", chosen)

    def test_build_repair_map_uses_existing_raw_first(self):
        repair_map, summary = repair.build_repair_map(
            source_env_ids={1, 2},
            alternatives_by_env={1: {"替代问题 1"}},
            question_env_frequency={"替代问题 1": 1},
            regenerate_missing=False,
            scenarios=None,
            llm_fn=None,
        )

        self.assertEqual({1: "替代问题 1"}, repair_map)
        self.assertEqual(1, summary["reused_from_raw"])
        self.assertEqual(1, summary["still_missing"])

    def test_apply_repairs_to_rl_rows_updates_question_and_user_message(self):
        rows = [
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": repair.FALLBACK_QUESTION},
                ],
                "question": repair.FALLBACK_QUESTION,
                "env_sample_idx": 11,
            }
        ]

        repaired_rows, stats = repair.apply_repairs_to_rows(
            target_name="rl",
            rows=rows,
            repair_map={11: "新的问题"},
            fallback_question=repair.FALLBACK_QUESTION,
        )

        self.assertEqual(1, stats["patched_rows"])
        self.assertEqual("新的问题", repaired_rows[0]["question"])
        self.assertEqual("新的问题", repaired_rows[0]["messages"][1]["content"])

    def test_apply_repairs_to_eval_rows_uses_source_env_mapping(self):
        rows = [
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": repair.FALLBACK_QUESTION},
                ],
                "question": repair.FALLBACK_QUESTION,
                "env_sample_idx": 3,
            }
        ]

        repaired_rows, stats = repair.apply_repairs_to_rows(
            target_name="eval",
            rows=rows,
            repair_map={103: "映射后的问题"},
            fallback_question=repair.FALLBACK_QUESTION,
            eval_source_map={3: 103},
        )

        self.assertEqual(1, stats["patched_rows"])
        self.assertEqual("映射后的问题", repaired_rows[0]["question"])


if __name__ == "__main__":
    unittest.main()
