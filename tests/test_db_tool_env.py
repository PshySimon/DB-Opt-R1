import unittest
import json
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

from environment.tools import DBToolEnv


class DBToolEnvCompatibilityTest(unittest.TestCase):
    def test_db_tool_env_exposes_tool_desc_for_verl_dataset(self):
        env = DBToolEnv(mode="real", config=None, max_turns=2)
        self.assertTrue(hasattr(env, "tool_desc"))
        self.assertIsInstance(env.tool_desc, list)
        self.assertGreater(len(env.tool_desc), 0)

    def test_db_tool_env_copy_preserves_type_and_scenarios(self):
        env = DBToolEnv(mode="train", cost_model=None, max_turns=2)
        env.scenarios = [object(), object()]

        copied = env.copy()

        self.assertIsInstance(copied, DBToolEnv)
        self.assertIsNot(copied, env)
        self.assertEqual(copied.scenarios, env.scenarios)
        self.assertEqual(copied.max_turns, env.max_turns)

    def test_load_scenarios_accepts_hydra_listconfig(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path_a = root / "collected_a.json"
            path_b = root / "collected_b.json"
            payload = [
                {
                    "name": "demo",
                    "source": "llm_generated",
                    "hardware": {"cpu_count": 8},
                    "knobs": {},
                    "workload": {},
                }
            ]
            path_a.write_text(json.dumps(payload), encoding="utf-8")
            path_b.write_text(json.dumps(payload), encoding="utf-8")

            scenario_list = OmegaConf.create([str(path_a), str(path_b)])
            scenarios = DBToolEnv._load_scenarios(scenario_list)

            self.assertEqual(len(scenarios), 2)


if __name__ == "__main__":
    unittest.main()
