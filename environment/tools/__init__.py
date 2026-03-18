"""
DB 工具环境
"""

import os
import random
import logging

import pandas as pd

from core.tool.tool_env import ToolEnv
from .db_tools import (
    GetHardwareInfoTool,
    GetCurrentConfigTool,
    GetDBMetricsTool,
    GetWorkloadInfoTool,
    GetRecentLogsTool,
    GetSystemStatsTool,
    SetKnobTool,
    RestartPGTool,
    ReloadPGTool,
    ResetConfigTool,
    PredictPerformanceTool,
    RunBenchmarkTool,
)

logger = logging.getLogger(__name__)


class DBToolEnv(ToolEnv):
    """DB 调优工具环境"""

    def __init__(self, mode="train", config=None,
                 dataset_path=None, scenario_dir=None,
                 cost_model=None, max_turns=10,
                 knob_space_path=None):
        """
        Args:
            mode: "train"（模拟环境）或 "real"（真实 PG）
            config: Config 对象（real 模式必须）
            dataset_path: CSV 数据集路径（train 模式，旧模式）
            scenario_dir: YAML 场景目录（train 模式，新模式，优先于 dataset_path）
            cost_model: Cost Model 对象（train 模式必须）
            max_turns: 最大交互轮数
            knob_space_path: knob_space.yaml 路径
        """
        self.mode = mode
        self.env_state = {}
        self.dataset = None
        self.scenario_dir = scenario_dir
        self.scenario_files = []
        self._original_knobs = {}

        if mode == "train":
            if scenario_dir and os.path.isdir(scenario_dir):
                # 新模式：扫描 YAML 场景
                self.scenario_files = sorted([
                    os.path.join(scenario_dir, f)
                    for f in os.listdir(scenario_dir)
                    if f.endswith(".json")
                ])
                logger.info(f"加载 {len(self.scenario_files)} 个场景文件 from {scenario_dir}")
            elif dataset_path:
                # 旧模式：CSV
                self.dataset = pd.read_csv(dataset_path, on_bad_lines="skip")

        # 加载可调 knob 列表
        tunable_knobs = []
        if knob_space_path:
            import yaml
            with open(knob_space_path) as f:
                ks = yaml.safe_load(f)
            tunable_knobs = list(ks.get("knobs", {}).keys())

        common = {"mode": mode, "config": config, "env_state": self.env_state}

        tools = [
            GetHardwareInfoTool(**common),
            GetCurrentConfigTool(tunable_knobs=tunable_knobs, **common),
            GetDBMetricsTool(**common),
            GetWorkloadInfoTool(**common),
            GetRecentLogsTool(**common),
            GetSystemStatsTool(**common),
            SetKnobTool(**common),
            RestartPGTool(**common),
            ReloadPGTool(**common),
            ResetConfigTool(**common),
        ]

        if mode == "train":
            tools.append(PredictPerformanceTool(cost_model=cost_model, **common))
        else:
            tools.append(RunBenchmarkTool(**common))

        super().__init__(tools=tools, max_turns=max_turns)

    @property
    def num_samples(self):
        """返回可用样本数"""
        if self.scenario_files:
            return len(self.scenario_files)
        elif self.dataset is not None:
            return len(self.dataset)
        return 0

    def reset(self, sample_idx=None):
        """重置环境，train 模式从数据集加载一条样本"""
        super().reset()

        if self.mode == "train":
            if self.scenario_files:
                # 新模式：加载 YAML 场景
                self._reset_from_scenario(sample_idx)
            elif self.dataset is not None:
                # 旧模式：从 CSV 加载
                self._reset_from_csv(sample_idx)

    def _reset_from_scenario(self, sample_idx=None):
        """从 YAML 场景文件加载"""
        from datasets.synthesis.scenarios.schema import ScenarioState

        if sample_idx is None:
            sample_idx = random.randint(0, len(self.scenario_files) - 1)

        json_path = self.scenario_files[sample_idx]
        scenario = ScenarioState.from_json(json_path)

        # 注入 scenario 到所有工具
        for tool in self.tools:
            tool.scenario = scenario

        # 同时更新 env_state 兼容旧代码
        self.env_state.clear()
        for k, v in scenario.hardware.items():
            self.env_state[f"hw_{k}"] = v
        for k, v in scenario.knobs.items():
            self.env_state[f"knob_{k}"] = v
        if scenario.workload:
            self.env_state["workload"] = scenario.workload.get("type", "mixed")
            self.env_state["tps"] = scenario.workload.get("tps_current", 0)

        # 保存原始 knob
        self._original_knobs = {k: v for k, v in self.env_state.items() if k.startswith("knob_")}
        for tool in self.tools:
            if isinstance(tool, ResetConfigTool):
                tool._original_knobs = dict(self._original_knobs)

        logger.debug(f"Episode reset: scenario={scenario.name}, "
                    f"difficulty={scenario.difficulty}")

    def _reset_from_csv(self, sample_idx=None):
        """从 CSV 加载（旧模式）"""
        if sample_idx is None:
            sample_idx = random.randint(0, len(self.dataset) - 1)

        row = self.dataset.iloc[sample_idx]
        self.env_state.clear()
        self.env_state.update(row.to_dict())

        # 尝试构造 ScenarioState 兼容
        try:
            from datasets.synthesis.scenarios.schema import ScenarioState
            scenario = ScenarioState.from_csv_row(row.to_dict())
            for tool in self.tools:
                tool.scenario = scenario
        except Exception:
            pass

        # 保存原始 knob 供 reset_config 使用
        self._original_knobs = {k: v for k, v in self.env_state.items() if k.startswith("knob_")}
        for tool in self.tools:
            if isinstance(tool, ResetConfigTool):
                tool._original_knobs = dict(self._original_knobs)

        logger.debug(f"Episode reset: sample_idx={sample_idx}, "
                    f"baseline_tps={self.env_state.get('tps', 'N/A')}")
