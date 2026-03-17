"""
Knob 生成器：在搜索空间内生成合法的 knob 配置
"""

import random
import math
import yaml
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 内存单位换算表（统一到 kB）
MEMORY_UNITS = {
    "kB": 1,
    "KB": 1,
    "MB": 1024,
    "GB": 1024 * 1024,
    "TB": 1024 * 1024 * 1024,
}


def parse_memory(value: str) -> int:
    """将内存字符串解析为 kB"""
    value = value.strip()
    for unit, multiplier in MEMORY_UNITS.items():
        if value.upper().endswith(unit.upper()):
            num = value[:-len(unit)]
            return int(float(num) * multiplier)
    return int(value)


def format_memory(value_kb: int) -> str:
    """将 kB 值格式化为人类可读的内存字符串"""
    if value_kb >= MEMORY_UNITS["GB"]:
        return f"{value_kb // MEMORY_UNITS['GB']}GB"
    elif value_kb >= MEMORY_UNITS["MB"]:
        return f"{value_kb // MEMORY_UNITS['MB']}MB"
    else:
        return f"{value_kb}kB"


class KnobSpace:
    """Knob 搜索空间"""

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.knobs = config["knobs"]
        self.benchmark = config.get("benchmark", {})
        self.collection = config.get("collection", {})

    def get_knob_names(self) -> list:
        return list(self.knobs.keys())

    def get_knob_info(self, name: str) -> dict:
        return self.knobs[name]

    def get_default_config(self) -> dict:
        """返回所有 knob 的默认值"""
        config = {}
        for name, info in self.knobs.items():
            config[name] = info["default"]
        return config

    def needs_restart(self, knob_config: dict) -> bool:
        """检查是否有需要重启的 knob 被修改"""
        for name, value in knob_config.items():
            info = self.knobs.get(name, {})
            if info.get("restart", False):
                default = info.get("default")
                if str(value) != str(default):
                    return True
        return False


class KnobGenerator:
    """Knob 配置生成器"""

    def __init__(self, knob_space: KnobSpace, seed: Optional[int] = None):
        self.space = knob_space
        if seed is not None:
            random.seed(seed)

    def sample_random(self) -> dict:
        """随机采样一组 knob 配置"""
        config = {}
        for name, info in self.space.knobs.items():
            config[name] = self._sample_knob(info)
        return config

    def sample_lhs(self, n_samples: int) -> list[dict]:
        """Latin Hypercube Sampling：生成 n 组配置，保证各维度均匀覆盖"""
        knob_names = self.space.get_knob_names()
        n_knobs = len(knob_names)

        # 为每个维度生成均匀分段
        intervals = []
        for i in range(n_knobs):
            perm = list(range(n_samples))
            random.shuffle(perm)
            intervals.append(perm)

        configs = []
        for i in range(n_samples):
            config = {}
            for j, name in enumerate(knob_names):
                info = self.space.knobs[name]
                # 在第 intervals[j][i] 个分段内随机取值
                segment = intervals[j][i]
                ratio = (segment + random.random()) / n_samples
                config[name] = self._sample_knob_at_ratio(info, ratio)
            configs.append(config)

        return configs

    def _sample_knob(self, info: dict):
        """根据类型随机采样一个 knob 值"""
        ratio = random.random()
        return self._sample_knob_at_ratio(info, ratio)

    def _sample_knob_at_ratio(self, info: dict, ratio: float):
        """在 [min, max] 范围内按比例取值"""
        knob_type = info["type"]

        if knob_type == "memory":
            min_kb = parse_memory(str(info["min"]))
            max_kb = parse_memory(str(info["max"]))
            # 对数空间采样（内存跨多个数量级）
            log_min = math.log2(max(min_kb, 1))
            log_max = math.log2(max(max_kb, 1))
            value_kb = int(2 ** (log_min + ratio * (log_max - log_min)))
            # 对齐到 MB
            value_kb = max(min_kb, min(max_kb, (value_kb // 1024) * 1024))
            if value_kb < 1024:
                value_kb = min_kb
            return format_memory(value_kb)

        elif knob_type == "integer":
            min_val = int(info["min"])
            max_val = int(info["max"])
            return min_val + int(ratio * (max_val - min_val))

        elif knob_type == "float":
            min_val = float(info["min"])
            max_val = float(info["max"])
            value = min_val + ratio * (max_val - min_val)
            return round(value, 3)

        elif knob_type == "enum":
            values = info["values"]
            idx = int(ratio * len(values)) % len(values)
            return values[idx]

        else:
            return info["default"]
