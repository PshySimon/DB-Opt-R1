import unittest

from environment.tools import DBToolEnv


class DBToolEnvCompatibilityTest(unittest.TestCase):
    def test_db_tool_env_exposes_tool_desc_for_verl_dataset(self):
        env = DBToolEnv(mode="real", config=None, max_turns=2)
        self.assertTrue(hasattr(env, "tool_desc"))
        self.assertIsInstance(env.tool_desc, list)
        self.assertGreater(len(env.tool_desc), 0)


if __name__ == "__main__":
    unittest.main()
