import os
import tempfile
import time
import unittest
from unittest import mock


class ProgressLoggingTest(unittest.TestCase):
    def test_progress_log_writes_to_configured_file(self):
        from training.progress import progress_log

        with tempfile.TemporaryDirectory() as tmpdir:
            progress_file = os.path.join(tmpdir, "progress.log")
            with mock.patch.dict(
                os.environ,
                {
                    "GRPO_PROGRESS_LOG": "1",
                    "GRPO_PROGRESS_LOG_FILE": progress_file,
                },
                clear=False,
            ):
                progress_log("step=1 gen_start")

            with open(progress_file, encoding="utf-8") as handle:
                content = handle.read()

        self.assertIn("[progress]", content)
        self.assertIn("step=1 gen_start", content)

    def test_progress_heartbeat_writes_while_operation_is_running(self):
        from training.progress import progress_heartbeat

        with tempfile.TemporaryDirectory() as tmpdir:
            progress_file = os.path.join(tmpdir, "progress.log")
            with mock.patch.dict(
                os.environ,
                {
                    "GRPO_PROGRESS_LOG": "1",
                    "GRPO_PROGRESS_LOG_FILE": progress_file,
                    "GRPO_PROGRESS_HEARTBEAT_INTERVAL": "0.01",
                },
                clear=False,
            ):
                with progress_heartbeat("step=1 generate"):
                    time.sleep(0.04)

            with open(progress_file, encoding="utf-8") as handle:
                content = handle.read()

        self.assertIn("step=1 generate still_running", content)


if __name__ == "__main__":
    unittest.main()
