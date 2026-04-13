import unittest
from unittest import mock

from data_pipeline.synthesis.trajectory import sampler


class SamplerContractTest(unittest.TestCase):
    def test_messages_to_sft_does_not_append_terminal_assistant(self):
        converted = sampler._messages_to_sft(
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "u"},
                {"role": "tool", "content": "obs"},
            ],
            improvement_pct=1.0,
            sample_idx=0,
        )

        self.assertEqual("tool", converted["messages"][-1]["role"])

    def test_build_llm_fn_passes_message_list_without_flattening(self):
        fake_client = mock.Mock()
        fake_client.generate.return_value = "ok"
        args = mock.Mock(
            model="dummy",
            providers_config=None,
            api_key=None,
            api_base=None,
            api_max_concurrent=1,
        )

        with mock.patch(
            "core.llm.multi_client.MultiProviderLLMClient",
            return_value=fake_client,
        ):
            llm_fn = sampler.build_llm_fn(args)
            messages = [{"role": "user", "content": "hi"}]
            llm_fn(messages, temperature=0.2)

        fake_client.generate.assert_called_once_with(messages, temperature=0.2)


if __name__ == "__main__":
    unittest.main()
