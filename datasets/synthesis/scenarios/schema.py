"""
场景数据结构定义

ScenarioState 表示一个完整的故障场景快照，
包含硬件信息、knob 配置、系统指标、数据库指标、等待事件、慢查询、日志等。
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ScenarioState:
    """一个完整的故障场景"""

    # 元信息
    name: str = ""
    variant: int = 0  # 同名配置的变体编号（用于断点续采匹配）
    source: str = "llm_generated"  # llm_generated | random_sampled
    difficulty: int = 1
    root_cause: List[str] = field(default_factory=list)
    description: str = ""
    question: str = ""  # 用户自然语言问题（基于采集到的真实DB症状生成）

    # 硬件
    hardware: Dict[str, Any] = field(default_factory=dict)

    # knob 配置（当前有问题的配置）
    knobs: Dict[str, Any] = field(default_factory=dict)

    # 系统级指标（CPU/内存/IO/swap）
    system: Dict[str, Any] = field(default_factory=dict)

    # 数据库指标（人可读格式）
    db_metrics: Dict[str, Any] = field(default_factory=dict)

    # 等待事件 Top N
    wait_events: List[Dict[str, Any]] = field(default_factory=list)

    # 慢查询 Top N
    slow_queries: List[Dict[str, Any]] = field(default_factory=list)

    # 日志
    logs: List[Dict[str, str]] = field(default_factory=list)

    # Workload 信息
    workload: Dict[str, Any] = field(default_factory=dict)

    # 标准答案（评估用，不给 LLM 看）
    solution: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str) -> "ScenarioState":
        """从 JSON 文件加载"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_json(self, path: str):
        """保存为 JSON"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)
        logger.info(f"场景已保存: {path}")

    @classmethod
    def from_csv_row(cls, row: dict, knob_defaults: dict = None) -> "ScenarioState":
        """从旧 CSV 行数据兼容加载（向后兼容）"""
        state = cls()

        # 硬件
        state.hardware = {k.replace("hw_", ""): v for k, v in row.items()
                         if k.startswith("hw_") and v is not None and str(v) != "nan"}

        # knobs
        state.knobs = {k.replace("knob_", ""): v for k, v in row.items()
                      if k.startswith("knob_")}

        # 从原始 metric 计数器计算可读指标
        state.db_metrics = cls._parse_csv_metrics(row)

        # workload
        state.workload = {
            "type": row.get("workload", "mixed"),
            "tps_current": float(row.get("tps", 0) or 0),
            "latency_avg_ms": float(row.get("latency_avg", 0) or 0),
        }

        # 系统指标和日志在 CSV 中不存在，留空
        state.system = {}
        state.logs = []
        state.wait_events = []
        state.slow_queries = []

        return state

    @staticmethod
    def _parse_csv_metrics(row: dict) -> dict:
        """从 CSV metric_ 前缀字段解析出人可读指标"""
        raw = {k.replace("metric_", ""): v for k, v in row.items() if k.startswith("metric_")}
        metrics = {}

        # 缓冲命中率
        blks_hit = float(raw.get("pg_stat_database_sum_blks_hit", 0) or 0)
        blks_read = float(raw.get("pg_stat_database_sum_blks_read", 0) or 0)
        total = blks_hit + blks_read
        if total > 0:
            metrics["buffer_hit_rate"] = round(blks_hit / total, 4)

        # 临时文件
        metrics["temp_files_count"] = int(float(raw.get("pg_stat_database_sum_temp_files", 0) or 0))
        metrics["temp_bytes_mb"] = round(float(raw.get("pg_stat_database_sum_temp_bytes", 0) or 0) / 1024 / 1024, 2)

        # 死元组
        live = float(raw.get("pg_stat_user_tables_sum_n_live_tup", 0) or 0)
        dead = float(raw.get("pg_stat_user_tables_sum_n_dead_tup", 0) or 0)
        if live + dead > 0:
            metrics["dead_tuple_ratio"] = round(dead / (live + dead), 4)

        # 扫描比例
        seq = float(raw.get("pg_stat_user_tables_sum_seq_scan", 0) or 0)
        idx = float(raw.get("pg_stat_user_tables_sum_idx_scan", 0) or 0)
        if seq + idx > 0:
            metrics["seq_scan_ratio"] = round(seq / (seq + idx), 4)

        # 死锁
        metrics["deadlocks"] = int(float(raw.get("pg_stat_database_sum_deadlocks", 0) or 0))

        return metrics

    def apply_knob_change(self, new_knobs: dict):
        """根据 knob 变化更新指标（简单规则引擎，方向正确即可）"""
        from core.db.knob_space import parse_memory

        for name, value in new_knobs.items():
            self.knobs[name] = value

        # shared_buffers 增大 → buffer_hit_rate 提高
        if "shared_buffers" in new_knobs and "buffer_hit_rate" in self.db_metrics:
            try:
                mem_gb = self.hardware.get("total_memory_gb", 16)
                new_sb_kb = parse_memory(str(new_knobs["shared_buffers"]))
                ratio = new_sb_kb / (mem_gb * 1024 * 1024)  # 占总内存比例
                self.db_metrics["buffer_hit_rate"] = min(0.999, 0.5 + ratio * 2)
            except Exception:
                pass

        # work_mem 增大 → temp_files 减少
        if "work_mem" in new_knobs:
            try:
                new_wm_kb = parse_memory(str(new_knobs["work_mem"]))
                if new_wm_kb >= 32 * 1024:  # >= 32MB
                    self.db_metrics["temp_files_count"] = 0
                    self.db_metrics["temp_bytes_mb"] = 0
                elif new_wm_kb >= 8 * 1024:  # >= 8MB
                    self.db_metrics["temp_files_count"] = max(0, self.db_metrics.get("temp_files_count", 0) // 4)
            except Exception:
                pass
