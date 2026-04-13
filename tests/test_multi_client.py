import threading
import unittest
from unittest.mock import patch

from core.llm.multi_client import ClientStats, MultiProviderLLMClient


class _DummyCompletions:
    def __init__(self, error=None, response_text="ok"):
        self.error = error
        self.response_text = response_text
        self.calls = 0
        self.last_kwargs = None

    def create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        if self.error is None:
            message = type("Message", (), {"content": self.response_text})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()
        raise Exception(self.error)


class _DummyChat:
    def __init__(self, completions):
        self.completions = completions


class _DummyClient:
    def __init__(self, error=None, response_text="ok"):
        self.chat = _DummyChat(_DummyCompletions(error, response_text))


class MultiClientRetryPolicyTest(unittest.TestCase):
    def test_context_length_error_does_not_retry(self):
        error = (
            "Error code: 400 - {'object': 'error', 'message': "
            "\"This model's maximum context length is 16384 tokens. "
            "However, you requested 16463 tokens (14415 in the messages, "
            "2048 in the completion). Please reduce the length of the messages "
            "or completion.\", 'type': 'BadRequestError', 'code': 400}"
        )
        dummy_client = _DummyClient(error)

        client = MultiProviderLLMClient.__new__(MultiProviderLLMClient)
        client.providers_config = None
        client.last_reload_time = 0.0
        client.reload_interval = 60.0
        client._stats_lock = threading.RLock()
        client.total_clients = 1
        client.stats = [
            ClientStats(
                idx=0,
                api_base="http://127.0.0.1:8000/v1",
                client=dummy_client,
                max_concurrent=1,
                model_name="dummy",
                api_key_str="dummy",
            )
        ]

        with patch("core.llm.multi_client.time.sleep", return_value=None):
            with self.assertRaises(Exception) as ctx:
                client.generate("prompt")

        self.assertIn("maximum context length", str(ctx.exception).lower())
        self.assertEqual(dummy_client.chat.completions.calls, 1)
        self.assertEqual(client.stats[0].error_streak, 0)
        self.assertEqual(client.stats[0].cooldown_until, 0.0)

    def test_generate_accepts_chat_messages_without_flattening(self):
        dummy_client = _DummyClient(response_text="<tool_call>{}</tool_call>")

        client = MultiProviderLLMClient.__new__(MultiProviderLLMClient)
        client.providers_config = None
        client.last_reload_time = 0.0
        client.reload_interval = 60.0
        client._stats_lock = threading.RLock()
        client.total_clients = 1
        client.stats = [
            ClientStats(
                idx=0,
                api_base="http://127.0.0.1:8000/v1",
                client=dummy_client,
                max_concurrent=1,
                model_name="dummy",
                api_key_str="dummy",
            )
        ]

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "user"},
        ]
        result = client.generate(messages)

        self.assertEqual("<tool_call>{}</tool_call>", result)
        self.assertEqual(messages, dummy_client.chat.completions.last_kwargs["messages"])


if __name__ == "__main__":
    unittest.main()
