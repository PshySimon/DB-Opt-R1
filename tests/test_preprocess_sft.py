import unittest

from data_pipeline import preprocess_sft


class PreprocessSftTest(unittest.TestCase):
    def test_convert_record_keeps_multiturn_messages(self):
        record = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "tool", "content": "t1"},
                {"role": "assistant", "content": "a2"},
            ],
            "tools": [{"name": "foo"}],
            "enable_thinking": True,
            "question": "q",
        }

        converted = preprocess_sft.convert_record(record)

        self.assertEqual(converted["messages"], record["messages"])
        self.assertEqual(converted["tools"], record["tools"])
        self.assertTrue(converted["enable_thinking"])
        self.assertEqual(converted["data_source"], "db_tuning")

    def test_convert_record_requires_two_assistant_turns(self):
        record = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
            ]
        }

        converted = preprocess_sft.convert_record(record, min_turns=2)

        self.assertIsNone(converted)


if __name__ == "__main__":
    unittest.main()
