import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path("/Users/caixiaomeng/Projects/Python/Compiler-R1/project/db-opt-r1")
SFT_DIR = REPO_ROOT / "scripts" / "experiments" / "v2" / "sft"
RL_DIR = REPO_ROOT / "scripts" / "experiments" / "v2" / "rl"


class ExperimentScriptsV2Test(unittest.TestCase):
    def test_expected_script_files_exist(self):
        expected = [
            SFT_DIR / "_common.sh",
            SFT_DIR / "_train_common.sh",
            SFT_DIR / "_eval_common.sh",
            SFT_DIR / "train_a0_direct_only.sh",
            SFT_DIR / "train_a1_full3k.sh",
            SFT_DIR / "train_b1_depth_trimmed.sh",
            SFT_DIR / "train_b2_depth_full.sh",
            SFT_DIR / "train_c1_gain_natural.sh",
            SFT_DIR / "train_c2_gain_balanced.sh",
            SFT_DIR / "eval_a0_direct_only.sh",
            SFT_DIR / "eval_a1_full3k.sh",
            SFT_DIR / "eval_b1_depth_trimmed.sh",
            SFT_DIR / "eval_b2_depth_full.sh",
            SFT_DIR / "eval_c1_gain_natural.sh",
            SFT_DIR / "eval_c2_gain_balanced.sh",
            SFT_DIR / "README.md",
            RL_DIR / "README.md",
        ]
        for path in expected:
            self.assertTrue(path.exists(), f"missing file: {path}")

    def test_train_script_references_v2_manifest_and_train_builder(self):
        wrapper = (SFT_DIR / "train_a0_direct_only.sh").read_text(encoding="utf-8")
        common = (SFT_DIR / "_common.sh").read_text(encoding="utf-8")
        train_common = (SFT_DIR / "_train_common.sh").read_text(encoding="utf-8")
        self.assertIn("sft_manifest_a0_direct_only.jsonl", wrapper)
        self.assertIn("data_pipeline.build_sft_experiment_dataset_v2", common)
        self.assertIn("scripts/train_sft_trl_lora.sh", train_common)

    def test_eval_script_references_v2_eval_assets(self):
        wrapper = (SFT_DIR / "eval_a0_direct_only.sh").read_text(encoding="utf-8")
        common = (SFT_DIR / "_common.sh").read_text(encoding="utf-8")
        eval_common = (SFT_DIR / "_eval_common.sh").read_text(encoding="utf-8")
        self.assertIn('run_eval_experiment "a0_direct_only"', wrapper)
        self.assertIn("data_pipeline/data/eval/v2/eval_trajectories_v2.jsonl", common)
        self.assertIn("data_pipeline/data/eval/v2/collected_eval_v2.json", common)
        self.assertIn("python \"${sampler_args[@]}\"", eval_common)
        self.assertIn("python \"${report_args[@]}\"", eval_common)

    def test_shell_scripts_are_valid_bash(self):
        scripts = sorted(str(path) for path in SFT_DIR.glob("*.sh"))
        result = subprocess.run(["bash", "-n", *scripts], cwd=REPO_ROOT, capture_output=True, text=True)
        if result.returncode != 0:
            self.fail(result.stderr)


if __name__ == "__main__":
    unittest.main()
