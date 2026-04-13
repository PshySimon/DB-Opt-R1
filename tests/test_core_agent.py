import unittest

from core.agent import rollout


class _StubEnv:
    def __init__(self):
        self.actions = []
        self.termination_reason = None

    def tools_format_func(self):
        return "# tools"

    def step(self, action_text):
        self.actions.append(action_text)
        if "Invalid tool call format" in action_text:
            raise AssertionError("rollout should not feed tool error messages back as actions")
        return "ok", 0.0, False, {"action_is_valid": True}

    def get_tracking_variables(self):
        return {"termination_reason": self.termination_reason}


class CoreAgentRolloutTest(unittest.TestCase):
    def test_rollout_stops_on_missing_tool_call(self):
        env = _StubEnv()

        def llm_fn(messages, temperature):
            return "<think>done</think>"

        messages, tracking = rollout(
            env=env,
            llm_fn=llm_fn,
            system_prompt="sys",
            user_message="user",
            max_turns=4,
        )

        self.assertEqual([], env.actions)
        self.assertEqual("no_tool_call", tracking["termination_reason"])
        self.assertEqual("<think>done</think>", messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
