"""
DB 工具环境
"""

import os
import json
import random
import logging
from collections.abc import Sequence

import pandas as pd

from core.tool.tool_env import ToolEnv
from .db_tools import (
    FinishTuningTool,
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
    PENDING_RESTART_KNOBS_KEY,
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
            scenario_dir: 场景数据源（JSON 文件或目录，train 模式，优先于 dataset_path）
            cost_model: Cost Model 对象（train 模式必须）
            max_turns: 最大交互轮数
            knob_space_path: knob_space.yaml 路径
        """
        self.mode = mode
        self.config = config
        self.dataset_path = dataset_path
        self.scenario_dir = scenario_dir
        self.cost_model = cost_model
        self.knob_space_path = knob_space_path
        self.env_state = {}
        self.dataset = None
        self.scenarios = []  # ScenarioState 列表
        self._original_knobs = {}

        if mode == "train":
            if scenario_dir:
                self.scenarios = self._load_scenarios(scenario_dir)
                logger.info(f"加载 {len(self.scenarios)} 个场景")
            elif dataset_path:
                # 旧模式：CSV
                self.dataset = pd.read_csv(dataset_path, on_bad_lines="skip")

        # 加载可调 knob 列表
        tunable_knobs = []
        knob_defaults = {}
        restart_knobs = set()
        if knob_space_path:
            import yaml
            with open(knob_space_path) as f:
                ks = yaml.safe_load(f)
            knob_defs = ks.get("knobs", {})
            tunable_knobs = list(knob_defs.keys())
            knob_defaults = {name: info.get("default") for name, info in knob_defs.items()}
            restart_knobs = {name for name, info in knob_defs.items() if info.get("restart", False)}

        common = {"mode": mode, "config": config, "env_state": self.env_state}

        tools = [
            FinishTuningTool(**common),
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
        for tool in self.tools:
            tool.knob_defaults = knob_defaults
            tool.restart_knobs = restart_knobs
        # training.verl.agent_rl_dataset expects the training ToolEnv interface.
        self.tool_desc = [tool.get_description() for tool in self.tools]

    def copy(self):
        env = DBToolEnv(
            mode=self.mode,
            config=self.config,
            dataset_path=None,
            scenario_dir=None,
            cost_model=self.cost_model,
            max_turns=self.max_turns,
            knob_space_path=self.knob_space_path,
        )
        env.scenarios = self.scenarios
        env.dataset = self.dataset
        return env

    @staticmethod
    def _load_scenarios(source) -> list:
        """加载场景：支持单 JSON 文件、目录、或多个路径列表"""
        import glob
        from data_pipeline.synthesis.scenarios.schema import ScenarioState
        from data_pipeline.synthesis.scenarios.loader import dedup_scenarios

        # 多路径列表：逐个加载后合并
        if isinstance(source, Sequence) and not isinstance(source, (str, bytes, os.PathLike)):
            all_scenarios = []
            for s in source:
                all_scenarios.extend(DBToolEnv._load_scenarios(s))
            return all_scenarios

        def _parse_items(items):
            return [ScenarioState(**{k: v for k, v in item.items()
                    if k in ScenarioState.__dataclass_fields__})
                    for item in items]

        if os.path.isfile(source):
            # 单文件（JSON 数组）
            with open(source, "r", encoding="utf-8") as f:
                items = json.load(f)
            items = dedup_scenarios(items, fname=source, logger=logger)
            scenarios = _parse_items(items)
            logger.info(f"  加载 {source}: {len(scenarios)} 个场景")
            return scenarios

        elif os.path.isdir(source):
            # 目录：优先匹配 collected_*.json（每个是数组），否则逐个 .json
            collected_files = sorted(glob.glob(os.path.join(source, "collected_*.json")))
            if collected_files:
                scenarios = []
                for fpath in collected_files:
                    with open(fpath, "r", encoding="utf-8") as f:
                        items = json.load(f)
                    items = dedup_scenarios(items, fname=fpath, logger=logger)
                    batch = _parse_items(items)
                    logger.info(f"  加载 {os.path.basename(fpath)}: {len(batch)} 个场景")
                    scenarios.extend(batch)
                return scenarios
            else:
                # 每个 .json 是单个场景
                scenarios = []
                for fname in sorted(os.listdir(source)):
                    if fname.endswith(".json"):
                        scenarios.append(ScenarioState.from_json(os.path.join(source, fname)))
                return scenarios

        return []

    @property
    def num_samples(self):
        """返回可用样本数"""
        if self.scenarios:
            return len(self.scenarios)
        elif self.dataset is not None:
            return len(self.dataset)
        return 0

    def reset(self, sample_idx=None):
        """重置环境，train 模式从数据集加载一条样本"""
        super().reset()

        if self.mode == "train":
            if self.scenarios:
                self._reset_from_scenario(sample_idx)
            elif self.dataset is not None:
                self._reset_from_csv(sample_idx)
        self.last_valid_config = self.get_current_config_snapshot()

    def _reset_from_scenario(self, sample_idx=None):
        """从场景列表加载（深拷贝，防止搜索过程修改污染原始数据）"""
        import copy
        if sample_idx is None:
            sample_idx = random.randint(0, len(self.scenarios) - 1)

        scenario = copy.deepcopy(self.scenarios[sample_idx])

        # 注入 scenario 到所有工具
        for tool in self.tools:
            tool.scenario = scenario

        # 同时更新 env_state 兼容旧代码
        self.env_state.clear()
        for k, v in scenario.hardware.items():
            self.env_state[f"hw_{k}"] = v
        for k, v in scenario.knobs.items():
            self.env_state[f"knob_{k}"] = v
        self.env_state[PENDING_RESTART_KNOBS_KEY] = {}
        if scenario.workload:
            if isinstance(scenario.workload, dict):
                self.env_state["workload"] = scenario.workload.get("type", "mixed")
                self.env_state["tps"] = scenario.workload.get("tps_current", 0)
            else:
                self.env_state["workload"] = str(scenario.workload)
                self.env_state["tps"] = 0

        # 保存原始 knob
        self._original_knobs = {k: v for k, v in self.env_state.items() if k.startswith("knob_")}
        for tool in self.tools:
            if isinstance(tool, ResetConfigTool):
                tool._original_knobs = dict(self._original_knobs)
            if hasattr(tool, '_original_knobs_snapshot'):
                tool._original_knobs_snapshot = dict(self._original_knobs)

        logger.debug(f"Episode reset: scenario={scenario.name}, "
                    f"difficulty={scenario.difficulty}")

    def _reset_from_csv(self, sample_idx=None):
        """从 CSV 加载（旧模式）"""
        if sample_idx is None:
            sample_idx = random.randint(0, len(self.dataset) - 1)

        row = self.dataset.iloc[sample_idx]
        self.env_state.clear()
        self.env_state.update(row.to_dict())
        self.env_state[PENDING_RESTART_KNOBS_KEY] = {}

        # 尝试构造 ScenarioState 兼容
        try:
            from data_pipeline.synthesis.scenarios.schema import ScenarioState
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
            if hasattr(tool, '_original_knobs_snapshot'):
                tool._original_knobs_snapshot = dict(self._original_knobs)

        logger.debug(f"Episode reset: sample_idx={sample_idx}, "
                    f"baseline_tps={self.env_state.get('tps', 'N/A')}")

    def get_current_config_snapshot(self):
        return {
            key.replace("knob_", ""): value
            for key, value in self.env_state.items()
            if key.startswith("knob_")
        }
