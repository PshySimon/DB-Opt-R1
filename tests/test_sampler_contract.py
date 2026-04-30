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

    def test_best_predict_improvement_uses_max_predict_from_messages(self):
        messages = [
            {
                "role": "user",
                "content": '<tool_response>{"predicted_tps": 110.0, "baseline_tps": 100.0, "actual_tps": 105.0, "improvement_pct": 2.0}</tool_response>',
            },
            {
                "role": "user",
                "content": '<tool_response>{"predicted_tps": 120.0, "baseline_tps": 100.0, "actual_tps": 106.0, "improvement_pct": 5.0}</tool_response>',
            },
        ]

        best = sampler._best_predict_improvement(messages)

        self.assertEqual(5.0, best)

    def test_best_predict_improvement_is_zero_when_messages_have_no_predict(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]

        best = sampler._best_predict_improvement(messages)

        self.assertEqual(0.0, best)

    def test_messages_to_eval_record_preserves_tracking_termination_reason(self):
        converted = sampler._messages_to_eval_record(
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "u"},
                {"role": "tool", "content": '{"ok": true}'},
            ],
            tracking={
                "termination_reason": "finish_tuning",
                "steps_taken": 3,
            },
            sample_idx=7,
            question="q",
        )

        self.assertEqual("finish_tuning", converted["termination_reason"])
        self.assertEqual({"termination_reason": "finish_tuning", "steps_taken": 3}, converted["tracking"])
        self.assertEqual("tool", converted["messages"][-1]["role"])


if __name__ == "__main__":
    unittest.main()
