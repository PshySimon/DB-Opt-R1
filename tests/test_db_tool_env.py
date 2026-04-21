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


class DBToolEnvCompatibilityTest(unittest.TestCase):
    def _make_train_env(self):
        env = DBToolEnv(
            mode="train",
            cost_model=_FakeCostModel(),
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

        env.step('<tool_call>{"name":"predict_performance","arguments":{}}</tool_call>')
        current_knobs = env.cost_model.calls[-1][0]
        self.assertEqual(current_knobs["shared_buffers"], "128MB")

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

        env.step('<tool_call>{"name":"predict_performance","arguments":{}}</tool_call>')
        current_knobs = env.cost_model.calls[-1][0]
        self.assertEqual(current_knobs["work_mem"], "32MB")


if __name__ == "__main__":
    unittest.main()
