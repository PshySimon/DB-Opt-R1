import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from data_pipeline.synthesis.scenarios import pipeline


class ScenarioSynthesizeResumeTest(unittest.TestCase):
    def _write_dimensions(self, path: Path) -> None:
        dims = {
            "hardware": [{"name": "tiny_hw", "cpu_count": 4, "total_memory_gb": 8, "disk_type": "SSD"}],
            "workloads": ["mixed"],
            "severities": ["mild"],
            "data_volumes": ["small"],
            "concurrent_loads": ["low"],
            "scenarios": [
                {
                    "name": "tiny_case",
                    "category": "memory",
                    "severity_varies": True,
                    "description": "tiny desc",
                    "difficulty": 1,
                    "key_knobs": {"shared_buffers": {"direction": "low", "range": "256MB-1GB"}},
                }
            ],
        }
        path.write_text(yaml.safe_dump(dims, allow_unicode=True), encoding="utf-8")

    def _write_knob_space(self, path: Path) -> None:
        knob_space = {
            "knobs": {
                "shared_buffers": {
                    "type": "memory",
                    "min": "128MB",
                    "max": "8GB",
                    "default": "1GB",
                }
            }
        }
        path.write_text(yaml.safe_dump(knob_space, allow_unicode=True), encoding="utf-8")

    def test_synthesize_recovers_from_truncated_json_using_backup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dims = tmp / "dims.yaml"
            knob_space = tmp / "knob_space.yaml"
            output = tmp / "knob_configs.json"

            self._write_dimensions(dims)
            self._write_knob_space(knob_space)

            hw_output = tmp / "knob_configs_tiny_hw.json"
            backup = tmp / "knob_configs_tiny_hw.json.bak"
            good_items = [
                {
                    "name": "tiny_case",
                    "variant": 0,
                    "source": "llm_generated",
                    "difficulty": 1,
                    "category": "memory",
                    "description": "tiny desc",
                    "workload": "mixed",
                    "severity": "mild",
                    "data_volume": "small",
                    "concurrent_load": "low",
                    "knobs": {"shared_buffers": "512MB"},
                    "hardware_hint": {"name": "tiny_hw", "cpu_count": 4, "total_memory_gb": 8, "disk_type": "SSD"},
                }
            ]
            backup.write_text(json.dumps(good_items, ensure_ascii=False, indent=2), encoding="utf-8")
            hw_output.write_text('{"broken": ', encoding="utf-8")

            with mock.patch.object(pipeline, "logger") as fake_logger:
                pipeline.synthesize_knobs(
                    dimensions_path=str(dims),
                    knob_space_path=str(knob_space),
                    output_path=str(output),
                    llm_generate=lambda prompt: '{"shared_buffers": "768MB"}',
                    per_cell=1,
                    workers=1,
                )

            restored = json.loads(hw_output.read_text(encoding="utf-8"))
            self.assertEqual(good_items, restored)
            warning_calls = [call.args[0] for call in fake_logger.warning.call_args_list]
            self.assertTrue(any("回退到备份文件" in message for message in warning_calls))


if __name__ == "__main__":
    unittest.main()
