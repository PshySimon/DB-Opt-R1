import unittest

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


if __name__ == "__main__":
    unittest.main()
