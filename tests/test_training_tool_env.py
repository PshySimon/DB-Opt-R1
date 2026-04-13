import unittest

from core.tool.tool_base import Tool
from core.tool.tool_env import ToolEnv as CoreToolEnv
from core.tool.tool_env import step as core_step
from training.tool.tool_env import ToolEnv as TrainingToolEnv
from training.tool.tool_env import step_batch as training_step_batch


class EchoTool(Tool):
    def __init__(self):
        super().__init__(
            name="echo",
            description="Echo input",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string", "description": "value"}},
                "required": ["value"],
            },
        )

    def execute(self, args):
        return f"echo:{args['value']}"


class ExplodingTool(Tool):
    def __init__(self):
        super().__init__(
            name="explode",
            description="Always fails",
            parameters={"type": "object", "properties": {}, "required": []},
        )

    def execute(self, args):
        raise RuntimeError("boom")


class TrainingToolEnvParityTest(unittest.TestCase):
    def setUp(self):
        self.core_env = CoreToolEnv(tools=[EchoTool()], max_turns=4)
        self.training_env = TrainingToolEnv(tools=[EchoTool()], max_turns=4)

    def test_training_batch_matches_core_step_when_tool_call_has_trailing_text(self):
        action = '<tool_call>{"name":"echo","arguments":{"value":"ok"}} trailing</tool_call>'

        core_result = core_step(self.core_env, action)
        batch_result = training_step_batch([self.training_env], [action])[0]

        self.assertEqual(core_result, batch_result)

    def test_training_batch_matches_core_step_for_same_valid_call(self):
        action = '<tool_call>{"name":"echo","arguments":{"value":"ok"}}</tool_call>'

        core_result = core_step(self.core_env, action)
        batch_result = training_step_batch([self.training_env], [action])[0]

        self.assertEqual(core_result, batch_result)

    def test_malformed_tool_name_dict_is_treated_as_invalid_not_crash(self):
        action = '<tool_call>{"name":{"tool":"echo"},"arguments":{"value":"ok"}}</tool_call>'

        result = core_step(self.core_env, action)

        self.assertEqual(result[1], CoreToolEnv.PENALTY_FOR_INVALID)
        self.assertFalse(result[3]["action_is_valid"])

    def test_training_batch_handles_malformed_tool_name_dict(self):
        action = '<tool_call>{"name":{"tool":"echo"},"arguments":{"value":"ok"}}</tool_call>'

        result = training_step_batch([self.training_env], [action])[0]

        self.assertEqual(result[1], CoreToolEnv.PENALTY_FOR_INVALID)
        self.assertFalse(result[3]["action_is_valid"])

    def test_tool_execution_error_requires_two_consecutive_failures_to_end_episode(self):
        env = CoreToolEnv(tools=[ExplodingTool()], max_turns=4)
        action = '<tool_call>{"name":"explode","arguments":{}}</tool_call>'

        first_result = core_step(env, action)
        self.assertIn("Error executing 'explode'", first_result[0])
        self.assertFalse(first_result[2])
        self.assertIsNone(env.termination_reason)

        second_result = core_step(env, action)

        self.assertTrue(second_result[2])
        self.assertEqual("tool_execution_error", env.termination_reason)


if __name__ == "__main__":
    unittest.main()
