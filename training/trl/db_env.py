"""
TRL GRPOTrainer 多轮工具调用环境

通过 environment_factory 接入 TRL 原生的多轮 agent 训练。
将 DBToolEnv 的工具暴露为 TRL 识别的公共方法。

Usage:
    trainer = GRPOTrainer(
        ...,
        environment_factory=lambda: DBTuningTRLEnv(
            cost_model=cost_model,
            scenario_dir="data_pipeline/data/scenarios/collected/",
            knob_space_path="configs/knob_space.yaml",
        ),
    )
"""

import json
import random
import logging
from typing import Optional
from copy import deepcopy

logger = logging.getLogger(__name__)


class DBTuningTRLEnv:
    """
    TRL 兼容的 DB 调优环境。

    TRL 的 environment_factory 协议:
    - reset(**kwargs) -> str | None: 重置环境，返回追加到 user message 的文本
    - 其他公共方法自动暴露为 tools，TRL 用 docstring + type hints 构建 tool schema
    """

    def __init__(
        self,
        cost_model=None,
        scenario_dir: str = None,
        scenario_files: list = None,
        knob_space_path: str = "configs/knob_space.yaml",
        max_turns: int = 10,
    ):
        self.cost_model = cost_model
        self.scenario_dir = scenario_dir
        self.scenario_files = scenario_files
        self.knob_space_path = knob_space_path
        self.max_turns = max_turns

        # 延迟加载场景和 knob space
        self._scenarios = None
        self._knob_space = None
        self._env = None  # 底层 DBToolEnv 实例

        # 追踪变量（供 reward 函数读取）
        self.improvement_pct = 0.0
        self.total_reward = 0.0
        self.steps_taken = 0

    def _ensure_scenarios(self):
        """延迟加载场景数据"""
        if self._scenarios is not None:
            return

        from environment.tools import DBToolEnv
        if self.scenario_files:
            self._scenarios = DBToolEnv._load_scenarios(self.scenario_files)
        elif self.scenario_dir:
            self._scenarios = DBToolEnv._load_scenarios(self.scenario_dir)
        else:
            self._scenarios = []

        if not self._scenarios:
            raise ValueError("没有加载到任何场景数据")

        logger.info(f"DBTuningTRLEnv: 加载 {len(self._scenarios)} 个场景")

    def _ensure_knob_space(self):
        """延迟加载 knob space"""
        if self._knob_space is not None:
            return

        import yaml
        with open(self.knob_space_path) as f:
            ks = yaml.safe_load(f)

        self._knob_space = ks.get("knobs", {})

    def reset(self, **kwargs) -> Optional[str]:
        """
        重置环境，随机选择一个场景。

        Returns:
            追加到 user message 的文本（场景描述）
        """
        self._ensure_scenarios()
        self._ensure_knob_space()

        # 随机选场景
        idx = random.randint(0, len(self._scenarios) - 1)
        self._current_scenario = deepcopy(self._scenarios[idx])

        # 创建底层 DBToolEnv
        from environment.tools import DBToolEnv
        self._env = DBToolEnv(
            mode="train",
            cost_model=self.cost_model,
            max_turns=self.max_turns,
            knob_space_path=self.knob_space_path,
        )
        self._env.scenarios = self._scenarios
        self._env.reset(sample_idx=idx)

        # 重置追踪
        self.improvement_pct = 0.0
        self.total_reward = 0.0
        self.steps_taken = 0

        # 返回场景描述，追加到 user message
        hw = self._current_scenario.hardware
        return (
            f"请优化这个 PostgreSQL 数据库的性能。\n\n"
            f"硬件环境：CPU {hw.get('cpu_count', '?')} 核，"
            f"内存 {hw.get('total_memory_gb', '?')} GB，"
            f"存储 {hw.get('storage_type', '?')}"
        )

    # ================================================================
    # 以下公共方法会被 TRL 自动发现并暴露为 tools
    # ================================================================

    def get_hardware_info(self) -> str:
        """
        获取服务器硬件信息，包括 CPU、内存、存储类型等。

        Returns:
            包含硬件信息的 JSON 字符串。
        """
        return self._call_tool("get_hardware_info", {})

    def get_current_config(self) -> str:
        """
        获取当前 PostgreSQL 的关键配置参数值。

        Returns:
            包含当前配置的 JSON 字符串。
        """
        return self._call_tool("get_current_config", {})

    def get_db_metrics(self) -> str:
        """
        获取数据库运行指标，包括缓存命中率、连接数等。

        Returns:
            包含数据库指标的 JSON 字符串。
        """
        return self._call_tool("get_db_metrics", {})

    def get_workload_info(self) -> str:
        """
        获取当前数据库负载信息，包括查询类型分布等。

        Returns:
            包含负载信息的 JSON 字符串。
        """
        return self._call_tool("get_workload_info", {})

    def get_recent_logs(self) -> str:
        """
        获取 PostgreSQL 最近的日志信息。

        Returns:
            最近的日志内容。
        """
        return self._call_tool("get_recent_logs", {})

    def get_system_stats(self) -> str:
        """
        获取系统级别统计信息，如 CPU 使用率、IO 等待等。

        Returns:
            包含系统统计的 JSON 字符串。
        """
        return self._call_tool("get_system_stats", {})

    def set_knob(self, knob_name: str, value: str) -> str:
        """
        设置一个 PostgreSQL 配置参数。

        Args:
            knob_name: 要设置的参数名，如 shared_buffers、work_mem 等。
            value: 参数值，如 "4GB"、"256MB"、"1.1" 等。

        Returns:
            设置结果，包括参数的旧值和新值。
        """
        return self._call_tool("set_knob", {"knob_name": knob_name, "value": value})

    def restart_pg(self) -> str:
        """
        重启 PostgreSQL 服务，使 postmaster 级别参数（如 shared_buffers）生效。

        Returns:
            重启结果。
        """
        return self._call_tool("restart_pg", {})

    def reload_pg(self) -> str:
        """
        重新加载 PostgreSQL 配置，使 sighup 级别参数生效（无需重启）。

        Returns:
            重载结果。
        """
        return self._call_tool("reload_pg", {})

    def reset_config(self) -> str:
        """
        将所有配置参数重置为默认值。

        Returns:
            重置结果。
        """
        return self._call_tool("reset_config", {})

    def predict_performance(self) -> str:
        """
        使用 Cost Model 预测当前配置下的性能（TPS），并返回相比默认配置的提升百分比。

        Returns:
            包含预测 TPS 和提升百分比的 JSON 字符串。
        """
        result = self._call_tool("predict_performance", {})

        # 解析 improvement_pct 供 reward 使用
        try:
            parsed = json.loads(result) if isinstance(result, str) else result
            self.improvement_pct = float(parsed.get("improvement_pct", 0.0))
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass

        return result

    # ================================================================
    # 内部方法
    # ================================================================

    def _call_tool(self, tool_name: str, args: dict) -> str:
        """通过底层 DBToolEnv 执行工具调用"""
        if self._env is None:
            return f"Error: 环境未初始化，请先调用 reset()"

        self.steps_taken += 1

        tool = self._env.tool_map.get(tool_name)
        if tool is None:
            return f"Error: 未知工具 {tool_name}"

        try:
            result = tool.execute(args)
            reward = tool.calculate_reward(args, result)
            self.total_reward += reward
            return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return f"Error: {str(e)}"
