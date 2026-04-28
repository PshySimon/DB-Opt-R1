import json
import tempfile
import unittest
from pathlib import Path

from data_pipeline import tokenize_sft_dataset as tokenize_sft


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        input_ids = []
        assistant_masks = []
        for message in messages:
            tokens = message["content"].split()
            input_ids.extend(range(len(input_ids), len(input_ids) + len(tokens)))
            assistant_masks.extend([1 if message["role"] == "assistant" else 0] * len(tokens))
        return {"input_ids": input_ids, "assistant_masks": assistant_masks}


class TokenizeSftDatasetTest(unittest.TestCase):
    def test_tokenize_record_preserves_assistant_masks_and_template_kwargs(self):
        tokenizer = FakeTokenizer()
        record = {
            "messages": [
                {"role": "user", "content": "hello now"},
                {"role": "assistant", "content": "answer tokens"},
            ],
            "tools": json.dumps([{"name": "lookup"}]),
            "enable_thinking": False,
            "data_source": "general",
        }

        tokenized = tokenize_sft.tokenize_record(record, tokenizer, max_length=16)

        self.assertEqual(tokenized["input_ids"], [0, 1, 2, 3])
        self.assertEqual(tokenized["assistant_masks"], [0, 0, 1, 1])
        self.assertEqual(tokenized["length"], 4)
        self.assertEqual(tokenized["assistant_tokens"], 2)
        self.assertEqual(tokenized["data_source"], "general")
        self.assertEqual(tokenizer.calls[0]["kwargs"]["tools"], [{"name": "lookup"}])
        self.assertFalse(tokenizer.calls[0]["kwargs"]["enable_thinking"])

    def test_tokenize_record_filters_overlength_and_empty_assistant_masks(self):
        tokenizer = FakeTokenizer()

        overlength = tokenize_sft.tokenize_record(
            {
                "messages": [
                    {"role": "user", "content": "one two"},
                    {"role": "assistant", "content": "three four"},
                ]
            },
            tokenizer,
            max_length=3,
        )
        self.assertIsNone(overlength)

        no_assistant = tokenize_sft.tokenize_record(
            {"messages": [{"role": "user", "content": "one two"}]},
            tokenizer,
            max_length=16,
        )
        self.assertIsNone(no_assistant)

    def test_default_output_dir_uses_tokenized_sft_registry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = tokenize_sft.default_output_dir(
                project_root=Path(tmpdir),
                dataset_name="full_mix_s20_db20_gen80",
                model_tag="qwen3",
                max_length=8192,
                seed=42,
            )

        self.assertTrue(str(output).endswith(
            "data_pipeline/data/tokenized_sft/full_mix_s20_db20_gen80__qwen3__ml8192__seed42"
        ))


if __name__ == "__main__":
    unittest.main()
