import unittest

from training.verl.agent_rl_dataset import ToolRLDataset


class _FakeTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, chat, **kwargs):
        self.calls.append({"chat": chat, "kwargs": kwargs})
        return "rendered"


class _FakeToolEnv:
    tool_desc = [{"name": "get_hardware_info"}]

    def tools_format_func(self):
        return "# Tools\n\n<tools></tools>"


class ToolRLDatasetPromptRenderingTest(unittest.TestCase):
    def test_custom_tool_prompt_matches_sft_protocol_without_mutating_chat(self):
        dataset = ToolRLDataset.__new__(ToolRLDataset)
        dataset.tokenizer = _FakeTokenizer()
        dataset.tool_env = _FakeToolEnv()
        dataset.tools = dataset.tool_env.tool_desc
        dataset.use_custom_tool_format_func = True

        chat = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "slow writes"},
        ]

        rendered = dataset._render_prompt_with_tools(chat)

        self.assertEqual("rendered", rendered)
        self.assertEqual("system prompt", chat[0]["content"])
        call = dataset.tokenizer.calls[-1]
        self.assertNotIn("tools", call["kwargs"])
        self.assertTrue(call["kwargs"]["add_generation_prompt"])
        self.assertFalse(call["kwargs"]["tokenize"])
        self.assertEqual(
            "system prompt\n\n# Tools\n\n<tools></tools>",
            call["chat"][0]["content"],
        )

    def test_default_tool_prompt_uses_tokenizer_tool_schema(self):
        dataset = ToolRLDataset.__new__(ToolRLDataset)
        dataset.tokenizer = _FakeTokenizer()
        dataset.tool_env = _FakeToolEnv()
        dataset.tools = dataset.tool_env.tool_desc
        dataset.use_custom_tool_format_func = False

        dataset._render_prompt_with_tools([{"role": "user", "content": "hello"}])

        call = dataset.tokenizer.calls[-1]
        self.assertEqual(dataset.tools, call["kwargs"]["tools"])


if __name__ == "__main__":
    unittest.main()
