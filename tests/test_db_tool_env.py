import unittest
import json
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

from environment.tools import DBToolEnv
from data_pipeline.synthesis.scenarios.schema import ScenarioState


class _FakeCostModel:
    def __init__(self):
        self.calls = []

    def predict(self, knobs, hw_info):
        self.calls.append((dict(knobs), dict(hw_info)))
        return 100.0 + len(self.calls)


class _GuardedCostModel:
    def __init__(self, confidence):
        self.confidence = confidence
        self.calls = []

    def predict(self, knobs, hw_info):
        self.calls.append((dict(knobs), dict(hw_info)))
        return 100.0 if len(self.calls) % 2 == 1 else 200.0

    def check_input_coverage(self, knobs, hw_info):
        return {
            "confidence": self.confidence,
            "hard_ood": self.confidence == "invalid",
            "near_boundary": self.confidence == "low",
            "features": ["shared_buffers"] if self.confidence != "high" else [],
        }


class DBToolEnvCompatibilityTest(unittest.TestCase):
    def _make_train_env(self, cost_model=None):
        env = DBToolEnv(
            mode="train",
            cost_model=cost_model or _FakeCostModel(),
            max_turns=8,
            knob_space_path="configs/knob_space.yaml",
        )
        env.scenarios = [
            ScenarioState(
                name="demo",
                source="llm_generated",
                hardware={"total_memory_gb": 16, "cpu_count": 8},
                knobs={"shared_buffers": "128MB", "work_mem": "4MB"},
                workload={"type": "mixed"},
                db_metrics={},
            )
        ]
        env.reset(sample_idx=0)
        return env

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

    def test_finish_tuning_marks_episode_done(self):
        env = DBToolEnv(mode="train", cost_model=_FakeCostModel(), max_turns=5)

        result, reward, done, info = env.step(
            '<tool_call>{"name":"finish_tuning","arguments":{}}</tool_call>'
        )

        self.assertTrue(done)
        self.assertTrue(info["action_is_valid"])
        self.assertEqual("finish_tuning", env.termination_reason)
        self.assertEqual(json.loads(result)["status"], "finished")
        self.assertEqual(reward, 0.0)

    def test_predict_budget_exhausted_after_third_predict(self):
        env = DBToolEnv(mode="train", cost_model=_FakeCostModel(), max_turns=8)
        env.env_state.update({"hw_cpu_count": 8, "knob_shared_buffers": "1GB"})

        for idx in range(3):
            _, _, done, _ = env.step(
                '<tool_call>{"name":"predict_performance","arguments":{}}</tool_call>'
            )
            if idx < 2:
                self.assertFalse(done)

        self.assertTrue(done)
        self.assertEqual(3, env.predict_calls_used)
        self.assertEqual("predict_budget_exhausted", env.termination_reason)

    def test_repeated_same_tool_same_args_ends_episode(self):
        env = DBToolEnv(mode="train", cost_model=_FakeCostModel(), max_turns=8)

        for idx in range(3):
            _, _, done, _ = env.step(
                '<tool_call>{"name":"get_hardware_info","arguments":{}}</tool_call>'
            )
            if idx < 2:
                self.assertFalse(done)

        self.assertTrue(done)
        self.assertEqual("repeated_tool_call", env.termination_reason)

    def test_invalid_tool_call_streak_ends_episode_after_threshold(self):
        env = DBToolEnv(mode="train", cost_model=_FakeCostModel(), max_turns=8)

        _, _, done_first, info_first = env.step("<think>bad</think>")
        _, _, done_second, info_second = env.step("<think>still bad</think>")

        self.assertFalse(done_first)
        self.assertFalse(info_first["action_is_valid"])
        self.assertTrue(done_second)
        self.assertFalse(info_second["action_is_valid"])
        self.assertEqual("invalid_tool_call", env.termination_reason)

    def test_static_knob_requires_restart_before_predict_uses_new_value(self):
        env = self._make_train_env()

        result, _, _, _ = env.step(
            '<tool_call>{"name":"set_knob","arguments":{"knobs":"{\\"shared_buffers\\": \\"512MB\\"}"}}</tool_call>'
        )
        payload = json.loads(result)
        self.assertEqual(payload["success"], [])
        self.assertEqual(payload["pending_restart"], ["shared_buffers"])

        predict_result, _, _, _ = env.step('<tool_call>{"name":"predict_performance","arguments":{}}</tool_call>')
        current_knobs = env.cost_model.calls[-1][0]
        self.assertEqual(current_knobs["shared_buffers"], "128MB")
        predict_payload = json.loads(predict_result)
        self.assertIn("未使用这些待生效值", predict_payload["warnings"][-1])

        env.step('<tool_call>{"name":"restart_pg","arguments":{}}</tool_call>')
        env.step('<tool_call>{"name":"predict_performance","arguments":{}}</tool_call>')
        current_knobs = env.cost_model.calls[-1][0]
        self.assertEqual(current_knobs["shared_buffers"], "512MB")

    def test_dynamic_knob_applies_immediately_without_restart(self):
        env = self._make_train_env()

        result, _, _, _ = env.step(
            '<tool_call>{"name":"set_knob","arguments":{"knobs":"{\\"work_mem\\": \\"32MB\\"}"}}</tool_call>'
        )
        payload = json.loads(result)
        self.assertEqual(payload["success"], ["work_mem"])
        self.assertEqual(payload["pending_restart"], [])
        self.assertEqual(payload["applied"], {"work_mem": "32MB"})

        env.step('<tool_call>{"name":"predict_performance","arguments":{}}</tool_call>')
        current_knobs = env.cost_model.calls[-1][0]
        self.assertEqual(current_knobs["work_mem"], "32MB")

    def test_invalid_pg_value_is_rejected_in_simulated_set_knob(self):
        env = self._make_train_env()

        result, _, _, _ = env.step(
            '<tool_call>{"name":"set_knob","arguments":{"knobs":"{\\"commit_delay\\": \\"10ms\\"}"}}</tool_call>'
        )
        payload = json.loads(result)

        self.assertEqual(payload["success"], [])
        self.assertEqual(payload["pending_restart"], [])
        self.assertEqual(payload["ignored"], [])
        self.assertEqual(payload["failed"][0]["name"], "commit_delay")
        self.assertNotIn("knob_commit_delay", env.env_state)
        self.assertNotIn("commit_delay", env.tools[0].scenario.knobs)

    def test_unknown_knob_is_ignored_in_simulated_set_knob(self):
        env = self._make_train_env()

        result, _, _, _ = env.step(
            '<tool_call>{"name":"set_knob","arguments":{"knobs":"{\\"checkpoint_segments\\": \\"32\\"}"}}</tool_call>'
        )
        payload = json.loads(result)

        self.assertEqual(payload["success"], [])
        self.assertEqual(payload["pending_restart"], [])
        self.assertEqual(payload["failed"], [])
        self.assertEqual(payload["ignored"][0]["name"], "checkpoint_segments")
        self.assertNotIn("knob_checkpoint_segments", env.env_state)
        self.assertNotIn("checkpoint_segments", env.tools[0].scenario.knobs)

    def test_enum_values_follow_pg_catalog_and_action_space(self):
        env = self._make_train_env()

        result, _, _, _ = env.step(
            '<tool_call>{"name":"set_knob","arguments":{"knobs":"{\\"synchronous_commit\\": \\"remote_apply\\"}"}}</tool_call>'
        )
        payload = json.loads(result)

        self.assertEqual(payload["success"], ["synchronous_commit"])
        self.assertEqual(payload["failed"], [])
        self.assertEqual(payload["applied"], {"synchronous_commit": "remote_apply"})
        self.assertEqual(env.env_state["knob_synchronous_commit"], "remote_apply")

    def test_removed_enum_alias_is_rejected(self):
        env = self._make_train_env()

        result, _, _, _ = env.step(
            '<tool_call>{"name":"set_knob","arguments":{"knobs":"{\\"synchronous_commit\\": \\"remote\\"}"}}</tool_call>'
        )
        payload = json.loads(result)

        self.assertEqual(payload["success"], [])
        self.assertEqual(payload["failed"][0]["name"], "synchronous_commit")
        self.assertNotIn("knob_synchronous_commit", env.env_state)

    def test_time_value_is_validated_against_pg_range(self):
        env = self._make_train_env()

        result, _, _, _ = env.step(
            '<tool_call>{"name":"set_knob","arguments":{"knobs":"{\\"checkpoint_timeout\\": \\"10s\\"}"}}</tool_call>'
        )
        payload = json.loads(result)

        self.assertEqual(payload["success"], [])
        self.assertEqual(payload["failed"][0]["name"], "checkpoint_timeout")
        self.assertNotIn("knob_checkpoint_timeout", env.env_state)

        env = self._make_train_env()
        result, _, _, _ = env.step(
            '<tool_call>{"name":"set_knob","arguments":{"knobs":"{\\"checkpoint_timeout\\": \\"5min\\"}"}}</tool_call>'
        )
        payload = json.loads(result)

        self.assertEqual(payload["success"], ["checkpoint_timeout"])
        self.assertEqual(payload["failed"], [])
        self.assertEqual(payload["applied"], {"checkpoint_timeout": "5min"})
        self.assertEqual(env.env_state["knob_checkpoint_timeout"], "5min")

        env = self._make_train_env()
        result, _, _, _ = env.step(
            '<tool_call>{"name":"set_knob","arguments":{"knobs":"{\\"checkpoint_timeout\\": \\"5m\\"}"}}</tool_call>'
        )
        payload = json.loads(result)

        self.assertEqual(payload["success"], ["checkpoint_timeout"])
        self.assertEqual(payload["failed"], [])
        self.assertEqual(payload["applied"], {"checkpoint_timeout": "5min"})
        self.assertEqual(env.env_state["knob_checkpoint_timeout"], "5min")

    def test_memory_value_is_saved_in_show_like_form_after_restart(self):
        env = self._make_train_env()

        result, _, _, _ = env.step(
            '<tool_call>{"name":"set_knob","arguments":{"knobs":"{\\"shared_buffers\\": \\"6.24GB\\"}"}}</tool_call>'
        )
        payload = json.loads(result)

        self.assertEqual(payload["success"], [])
        self.assertEqual(payload["pending_restart"], ["shared_buffers"])
        self.assertEqual(payload["pending_restart_values"], {"shared_buffers": "6390MB"})
        restart_result, _, _, _ = env.step('<tool_call>{"name":"restart_pg","arguments":{}}</tool_call>')
        self.assertEqual(json.loads(restart_result)["applied"], {"shared_buffers": "6390MB"})

        self.assertEqual(env.env_state["knob_shared_buffers"], "6390MB")
        self.assertEqual(env.tools[0].scenario.knobs["shared_buffers"], "6390MB")

    def test_predict_invalid_coverage_zeroes_improvement(self):
        env = self._make_train_env(cost_model=_GuardedCostModel("invalid"))

        result, reward, _, _ = env.step(
            '<tool_call>{"name":"predict_performance","arguments":{}}</tool_call>'
        )
        payload = json.loads(result)

        self.assertEqual(payload["confidence"], "invalid")
        self.assertEqual(payload["improvement_pct"], 0.0)
        self.assertEqual(reward, 0.0)
        self.assertTrue(payload["warnings"])

    def test_predict_low_confidence_caps_improvement(self):
        env = self._make_train_env(cost_model=_GuardedCostModel("low"))

        result, reward, _, _ = env.step(
            '<tool_call>{"name":"predict_performance","arguments":{}}</tool_call>'
        )
        payload = json.loads(result)

        self.assertEqual(payload["confidence"], "low")
        self.assertEqual(payload["improvement_pct"], 25.0)
        self.assertEqual(reward, 0.25)
        self.assertTrue(payload["warnings"])


if __name__ == "__main__":
    unittest.main()
