import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path("/Users/caixiaomeng/Projects/Python/Compiler-R1/project/db-opt-r1")
TRAIN_SFT_DIR = REPO_ROOT / "scripts" / "experiments" / "v2" / "train" / "sft"
TRAIN_RL_DIR = REPO_ROOT / "scripts" / "experiments" / "v2" / "train" / "rl"
EVAL_SFT_DIR = REPO_ROOT / "scripts" / "experiments" / "v2" / "eval" / "sft"


class ExperimentScriptsV2Test(unittest.TestCase):
    def test_expected_script_files_exist(self):
        expected = [
            REPO_ROOT / "scripts" / "run_local_transformers_eval.py",
            TRAIN_SFT_DIR / "_common.sh",
            TRAIN_SFT_DIR / "trl" / "_common.sh",
            TRAIN_SFT_DIR / "trl" / "lora" / "train_a0_direct_only.sh",
            TRAIN_SFT_DIR / "trl" / "lora" / "train_a1_full3k.sh",
            TRAIN_SFT_DIR / "trl" / "lora" / "train_b1_depth_trimmed.sh",
            TRAIN_SFT_DIR / "trl" / "lora" / "train_b2_depth_full.sh",
            TRAIN_SFT_DIR / "trl" / "lora" / "train_c1_gain_natural.sh",
            TRAIN_SFT_DIR / "trl" / "lora" / "train_c2_gain_balanced.sh",
            TRAIN_SFT_DIR / "trl" / "full" / "train_a0_direct_only.sh",
            TRAIN_SFT_DIR / "trl" / "full" / "train_a1_full3k.sh",
            TRAIN_SFT_DIR / "trl" / "full" / "train_b1_depth_trimmed.sh",
            TRAIN_SFT_DIR / "trl" / "full" / "train_b2_depth_full.sh",
            TRAIN_SFT_DIR / "trl" / "full" / "train_c1_gain_natural.sh",
            TRAIN_SFT_DIR / "trl" / "full" / "train_c2_gain_balanced.sh",
            TRAIN_SFT_DIR / "verl" / "_common.sh",
            TRAIN_SFT_DIR / "verl" / "lora" / "train_a0_direct_only.sh",
            TRAIN_SFT_DIR / "verl" / "lora" / "train_a1_full3k.sh",
            TRAIN_SFT_DIR / "verl" / "lora" / "train_b1_depth_trimmed.sh",
            TRAIN_SFT_DIR / "verl" / "lora" / "train_b2_depth_full.sh",
            TRAIN_SFT_DIR / "verl" / "lora" / "train_c1_gain_natural.sh",
            TRAIN_SFT_DIR / "verl" / "lora" / "train_c2_gain_balanced.sh",
            TRAIN_SFT_DIR / "verl" / "full" / "train_a0_direct_only.sh",
            TRAIN_SFT_DIR / "verl" / "full" / "train_a1_full3k.sh",
            TRAIN_SFT_DIR / "verl" / "full" / "train_b1_depth_trimmed.sh",
            TRAIN_SFT_DIR / "verl" / "full" / "train_b2_depth_full.sh",
            TRAIN_SFT_DIR / "verl" / "full" / "train_c1_gain_natural.sh",
            TRAIN_SFT_DIR / "verl" / "full" / "train_c2_gain_balanced.sh",
            EVAL_SFT_DIR / "_eval_common.sh",
            EVAL_SFT_DIR / "eval_a0_direct_only.sh",
            EVAL_SFT_DIR / "eval_a1_full3k.sh",
            EVAL_SFT_DIR / "eval_b1_depth_trimmed.sh",
            EVAL_SFT_DIR / "eval_b2_depth_full.sh",
            EVAL_SFT_DIR / "eval_c1_gain_natural.sh",
            EVAL_SFT_DIR / "eval_c2_gain_balanced.sh",
            TRAIN_SFT_DIR / "README.md",
            TRAIN_RL_DIR / "_common.sh",
            TRAIN_RL_DIR / "verl" / "_common.sh",
            TRAIN_RL_DIR / "verl" / "lora" / "train_frontier_1q.sh",
            TRAIN_RL_DIR / "verl" / "lora" / "train_hard_1q.sh",
            TRAIN_RL_DIR / "verl" / "lora" / "train_frontier_plus_hard_1q.sh",
            TRAIN_RL_DIR / "verl" / "full" / "train_frontier_1q.sh",
            TRAIN_RL_DIR / "verl" / "full" / "train_hard_1q.sh",
            TRAIN_RL_DIR / "verl" / "full" / "train_frontier_plus_hard_1q.sh",
            TRAIN_RL_DIR / "README.md",
        ]
        for path in expected:
            self.assertTrue(path.exists(), f"missing file: {path}")

    def test_trl_train_script_references_v2_manifest_and_train_builder(self):
        wrapper = (TRAIN_SFT_DIR / "trl" / "lora" / "train_a0_direct_only.sh").read_text(encoding="utf-8")
        common = (TRAIN_SFT_DIR / "_common.sh").read_text(encoding="utf-8")
        train_common = (TRAIN_SFT_DIR / "trl" / "_common.sh").read_text(encoding="utf-8")
        self.assertIn("sft_manifest_a0_direct_only.jsonl", wrapper)
        self.assertIn("data_pipeline.build_sft_experiment_dataset_v2", common)
        self.assertIn('train_script="$REPO_ROOT/scripts/train_sft_trl_${train_mode}.sh"', train_common)
        self.assertIn('model_save/experiments/v2/sft/trl/$train_mode', train_common)
        self.assertIn('BASE_MODEL="$model_path"', train_common)

    def test_verl_train_script_references_v2_parquet_builder(self):
        wrapper = (TRAIN_SFT_DIR / "verl" / "full" / "train_a0_direct_only.sh").read_text(encoding="utf-8")
        common = (TRAIN_SFT_DIR / "_common.sh").read_text(encoding="utf-8")
        train_common = (TRAIN_SFT_DIR / "verl" / "_common.sh").read_text(encoding="utf-8")
        self.assertIn("sft_manifest_a0_direct_only.jsonl", wrapper)
        self.assertIn("data_pipeline.preprocess_sft", common)
        self.assertIn('train_script="$REPO_ROOT/scripts/train_sft_verl_${train_mode}.sh"', train_common)
        self.assertIn('model_save/experiments/v2/sft/verl/$train_mode', train_common)
        self.assertIn('BASE_MODEL="$model_path"', train_common)

    def test_eval_script_references_v2_eval_assets(self):
        wrapper = (EVAL_SFT_DIR / "eval_a0_direct_only.sh").read_text(encoding="utf-8")
        common = (TRAIN_SFT_DIR / "_common.sh").read_text(encoding="utf-8")
        eval_common = (EVAL_SFT_DIR / "_eval_common.sh").read_text(encoding="utf-8")
        self.assertIn('run_eval_experiment "a0_direct_only"', wrapper)
        self.assertIn("data_pipeline/data/eval/v2/eval_trajectories_v2.jsonl", common)
        self.assertIn("data_pipeline/data/eval/v2/collected_eval_v2.json", common)
        self.assertIn("../../train/sft/_common.sh", eval_common)
        self.assertIn("python \"${sampler_args[@]}\"", eval_common)
        self.assertIn("python \"${report_args[@]}\"", eval_common)
        self.assertIn('if [ -n "${LOCAL_MODEL_PATH:-}" ]; then', eval_common)
        self.assertIn('scripts/run_local_transformers_eval.py', eval_common)
        self.assertIn('if [ -n "${START_INDEX:-}" ]; then', eval_common)
        self.assertIn('if [ -n "${END_INDEX:-}" ]; then', eval_common)
        self.assertIn('LOCAL_LOG_INTERVAL', eval_common)

    def test_rl_train_script_references_v2_rl_assets(self):
        wrapper = (TRAIN_RL_DIR / "verl" / "lora" / "train_frontier_1q.sh").read_text(encoding="utf-8")
        common = (TRAIN_RL_DIR / "_common.sh").read_text(encoding="utf-8")
        train_common = (TRAIN_RL_DIR / "verl" / "_common.sh").read_text(encoding="utf-8")
        self.assertIn('run_verl_rl_train_experiment "frontier_1q"', wrapper)
        self.assertIn('V2_RL_DIR="$REPO_ROOT/data_pipeline/data/train/v2/rl"', common)
        self.assertIn('V2_RL_FRONTIER="$V2_RL_DIR/rl_frontier_1q.jsonl"', common)
        self.assertIn("data_pipeline/data/scenarios/collected/collected_8c16g_hdd_20k.json", common)
        self.assertIn("cost_model/checkpoints/v10_lgbm", common)
        self.assertIn("data_pipeline.preprocess_grpo", common)
        self.assertIn('train_script="$REPO_ROOT/scripts/train_grpo_verl_${train_mode}.sh"', train_common)
        self.assertIn('model_save/experiments/v2/rl/verl/$train_mode', train_common)
        self.assertIn('SFT_CHECKPOINT="$model_path"', train_common)

    def test_shell_scripts_are_valid_bash(self):
        scripts = sorted(str(path) for path in TRAIN_SFT_DIR.rglob("*.sh"))
        scripts.extend(sorted(str(path) for path in EVAL_SFT_DIR.rglob("*.sh")))
        scripts.extend(sorted(str(path) for path in TRAIN_RL_DIR.rglob("*.sh")))
        result = subprocess.run(["bash", "-n", *scripts], cwd=REPO_ROOT, capture_output=True, text=True)
        if result.returncode != 0:
            self.fail(result.stderr)


if __name__ == "__main__":
    unittest.main()
