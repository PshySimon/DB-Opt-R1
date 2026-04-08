from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TrainingScriptDefaultsTest(unittest.TestCase):
    def test_verl_sft_lora_defaults_to_cleaned_dataset(self):
        content = (ROOT / "scripts" / "train_sft_verl_lora.sh").read_text()
        self.assertIn('DATA_DIR="${DATA_DIR:-./datasets/sft_cleaned}"', content)
        self.assertIn("+data.prompt_key=extra_info", content)
        self.assertIn("+data.response_key=extra_info", content)
        self.assertIn("+model.partial_pretrain=$BASE_MODEL", content)
        self.assertIn(
            "python -m data_pipeline.preprocess_sft",
            content,
        )
        self.assertIn(
            "--input_files data_pipeline/data/train/sft_trajectories_cleaned.jsonl",
            content,
        )
        self.assertIn("--output_dir ./datasets/sft_cleaned/", content)

    def test_verl_sft_full_defaults_to_cleaned_dataset(self):
        content = (ROOT / "scripts" / "train_sft_verl_full.sh").read_text()
        self.assertIn('DATA_DIR="${DATA_DIR:-./datasets/sft_cleaned}"', content)
        self.assertIn("+data.prompt_key=extra_info", content)
        self.assertIn("+data.response_key=extra_info", content)
        self.assertIn("+model.partial_pretrain=$BASE_MODEL", content)
        self.assertIn(
            "python -m data_pipeline.preprocess_sft",
            content,
        )
        self.assertIn(
            "--input_files data_pipeline/data/train/sft_trajectories_cleaned.jsonl",
            content,
        )
        self.assertIn("--output_dir ./datasets/sft_cleaned/", content)


if __name__ == "__main__":
    unittest.main()
