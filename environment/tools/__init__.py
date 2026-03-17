"""
DB 工具环境
"""

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
                 dataset_path=None, cost_model=None, max_turns=10):
        """
        Args:
            mode: "train"（模拟环境）或 "real"（真实 PG）
            config: Config 对象（real 模式必须）
            dataset_path: CSV 数据集路径（train 模式必须）
            cost_model: Cost Model 对象（train 模式必须）
            max_turns: 最大交互轮数
        """
        self.mode = mode
        self.env_state = {}
        self.dataset = None
        self._original_knobs = {}

        if mode == "train" and dataset_path:
            self.dataset = pd.read_csv(dataset_path)

        common = {"mode": mode, "config": config, "env_state": self.env_state}

        tools = [
            GetHardwareInfoTool(**common),
            GetCurrentConfigTool(**common),
            GetDBMetricsTool(**common),
            GetWorkloadInfoTool(**common),
            GetRecentLogsTool(**common),
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

    def reset(self, sample_idx=None):
        """重置环境，train 模式从数据集加载一条样本"""
        super().reset()

        if self.mode == "train" and self.dataset is not None:
            if sample_idx is None:
                sample_idx = random.randint(0, len(self.dataset) - 1)

            row = self.dataset.iloc[sample_idx]
            self.env_state.clear()
            self.env_state.update(row.to_dict())

            # 保存原始 knob 供 reset_config 使用
            self._original_knobs = {
                k: v for k, v in self.env_state.items()
                if k.startswith("knob_")
            }

            # 注入给 ResetConfigTool
            for tool in self.tools:
                if isinstance(tool, ResetConfigTool):
                    tool._original_knobs = dict(self._original_knobs)

            logger.debug(f"Episode reset: sample_idx={sample_idx}, "
                        f"baseline_tps={self.env_state.get('tps', 'N/A')}")
