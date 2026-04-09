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


if __name__ == "__main__":
    unittest.main()
