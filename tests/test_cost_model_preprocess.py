import unittest

import pandas as pd

from cost_model.preprocess import KnobPreprocessor


class CostModelPreprocessTest(unittest.TestCase):
    def test_wal_compression_is_encoded_as_enum(self):
        prep = KnobPreprocessor("configs/knob_space.yaml")
        df = pd.DataFrame(
            {
                "knob_wal_compression": ["off", "on", "pglz", "lz4", "zstd"],
                "workload": ["mixed"] * 5,
            }
        )

        features = prep._build_features(df)

        self.assertEqual(features["wal_compression"].tolist(), [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_synchronous_commit_accepts_remote_apply(self):
        prep = KnobPreprocessor("configs/knob_space.yaml")
        df = pd.DataFrame(
            {
                "knob_synchronous_commit": ["off", "local", "on", "remote_write", "remote_apply"],
                "workload": ["mixed"] * 5,
            }
        )

        features = prep._build_features(df)

        self.assertEqual(features["synchronous_commit"].tolist(), [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_input_coverage_marks_out_of_training_range(self):
        prep = KnobPreprocessor("configs/knob_space.yaml")
        df = pd.DataFrame(
            {
                "knob_shared_buffers": ["128MB", "256MB", "512MB"],
                "workload": ["mixed", "mixed", "mixed"],
                "hw_total_memory_gb": [16, 16, 16],
            }
        )
        features = prep._build_features(df)
        prep.feature_names = list(features.columns)
        prep.feature_bounds = prep._compute_feature_bounds(features)
        prep._fitted = True

        covered = prep.check_input_coverage(
            {"shared_buffers": "256MB", "workload": "mixed"},
            {"total_memory_gb": 16},
        )
        self.assertEqual(covered["confidence"], "high")

        ood = prep.check_input_coverage(
            {"shared_buffers": "64GB", "workload": "mixed"},
            {"total_memory_gb": 16},
        )
        self.assertEqual(ood["confidence"], "invalid")
        self.assertTrue(ood["hard_ood"])


if __name__ == "__main__":
    unittest.main()
