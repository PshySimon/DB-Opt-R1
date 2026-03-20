"""
Knob 生成器：在搜索空间内生成合法的 knob 配置
"""

import random
import math
import logging
from typing import Optional

from core.db.knob_space import KnobSpace, parse_memory, format_memory

logger = logging.getLogger(__name__)



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

    def sample_near_default(self) -> dict:
        """在默认值附近高斯扰动（±30%），让 Cost Model 见过'正常'配置"""
        config = {}
        for name, info in self.space.knobs.items():
            config[name] = self._sample_knob_near_default(info)
        return config

    def sample_mixed(self, n: int) -> list[dict]:
        """混合采样：40% random + 40% near_default + 20% LHS"""
        n_random = int(n * 0.4)
        n_near = int(n * 0.4)
        n_lhs = n - n_random - n_near

        configs = []
        # 纯随机
        for _ in range(n_random):
            configs.append(self.sample_random())
        # 默认值附近
        for _ in range(n_near):
            configs.append(self.sample_near_default())
        # LHS
        if n_lhs > 0:
            configs.extend(self.sample_lhs(n_lhs))

        random.shuffle(configs)
        return configs

    def _sample_knob_near_default(self, info: dict):
        """在默认值附近高斯扰动（sigma=0.15，即 ±30% 内覆盖 95%）"""
        knob_type = info["type"]
        default = info.get("default")

        if knob_type == "memory":
            min_kb = parse_memory(str(info["min"]))
            max_kb = parse_memory(str(info["max"]))
            def_kb = parse_memory(str(default)) if default else min_kb
            log_min = math.log2(max(min_kb, 1))
            log_max = math.log2(max(max_kb, 1))
            log_def = math.log2(max(def_kb, 1))
            # 高斯扰动
            span = log_max - log_min
            ratio = (log_def - log_min) / span if span > 0 else 0.5
            ratio = max(0, min(1, random.gauss(ratio, 0.15)))
            return self._sample_knob_at_ratio(info, ratio)

        elif knob_type == "integer":
            min_val = int(info["min"])
            max_val = int(info["max"])
            def_val = int(default) if default is not None else min_val
            span = max_val - min_val
            ratio = (def_val - min_val) / span if span > 0 else 0.5
            ratio = max(0, min(1, random.gauss(ratio, 0.15)))
            return min_val + int(ratio * span)

        elif knob_type == "float":
            min_val = float(info["min"])
            max_val = float(info["max"])
            def_val = float(default) if default is not None else min_val
            span = max_val - min_val
            ratio = (def_val - min_val) / span if span > 0 else 0.5
            ratio = max(0, min(1, random.gauss(ratio, 0.15)))
            return round(min_val + ratio * span, 3)

        elif knob_type == "enum":
            # 枚举类型 80% 返回默认值，20% 随机
            values = info["values"]
            if random.random() < 0.8 and default in values:
                return default
            return random.choice(values)

        else:
            return default

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
