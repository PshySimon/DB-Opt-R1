"""
Knob 搜索空间定义与工具函数

从 knob_space.yaml 加载 knob 元信息，提供验证、默认值查询等功能。
"""

import yaml
import logging
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

    def validate(self, config: dict) -> dict:
        """校验 LLM 生成的 knob 配置，过滤非法项

        Returns:
            合法的 knob 子集
        """
        valid = {}
        for name, value in config.items():
            if name not in self.knobs:
                logger.warning(f"跳过未知 knob: {name}")
                continue
            info = self.knobs[name]
            try:
                if self._is_valid_value(info, value):
                    valid[name] = value
                else:
                    logger.warning(f"值超出范围: {name}={value}")
            except Exception as e:
                logger.warning(f"校验 {name}={value} 失败: {e}")
        return valid

    def _is_valid_value(self, info: dict, value) -> bool:
        """检查值是否在合法范围内"""
        knob_type = info["type"]

        if knob_type == "memory":
            try:
                val_kb = parse_memory(str(value))
                min_kb = parse_memory(str(info["min"]))
                max_kb = parse_memory(str(info["max"]))
                return min_kb <= val_kb <= max_kb
            except (ValueError, TypeError):
                return False

        elif knob_type == "integer":
            try:
                val = int(value)
                return int(info["min"]) <= val <= int(info["max"])
            except (ValueError, TypeError):
                return False

        elif knob_type == "float":
            try:
                val = float(value)
                return float(info["min"]) <= val <= float(info["max"])
            except (ValueError, TypeError):
                return False

        elif knob_type == "enum":
            return str(value) in info["values"]

        return True

    def summarize_for_prompt(self) -> str:
        """生成可放入 LLM prompt 的 knob 空间摘要"""
        lines = []
        for name, info in self.knobs.items():
            t = info["type"]
            if t == "memory":
                lines.append(f"  {name}: type=memory, min={info['min']}, max={info['max']}, default={info['default']}")
            elif t in ("integer", "float"):
                unit = f", unit={info['unit']}" if 'unit' in info else ""
                lines.append(f"  {name}: type={t}, min={info['min']}, max={info['max']}, default={info['default']}{unit}")
            elif t == "enum":
                lines.append(f"  {name}: type=enum, values={info['values']}, default={info['default']}")
        return "\n".join(lines)

    # ─────────────── 空间转换 ───────────────


    def build_skopt_space(self):
        """构建 skopt 搜索空间。

        Returns:
            (dimensions, dim_names, knob_types)
        """
        from skopt.space import Real, Integer, Categorical

        dimensions = []
        dim_names = []
        knob_types = {}

        for name, info in self.knobs.items():
            ktype = info["type"]
            knob_types[name] = info

            if ktype == "memory":
                lo = parse_memory(str(info["min"]))
                hi = parse_memory(str(info["max"]))
                dimensions.append(Real(float(lo), float(hi), name=name))
            elif ktype == "integer":
                dimensions.append(Integer(int(info["min"]), int(info["max"]), name=name))
            elif ktype == "float":
                dimensions.append(Real(float(info["min"]), float(info["max"]), name=name))
            elif ktype == "enum":
                dimensions.append(Categorical(info["values"], name=name))

            dim_names.append(name)

        return dimensions, dim_names, knob_types

    def knobs_to_vector(self, knobs_dict: dict, dim_names: list) -> list:
        """knob dict → skopt 向量"""
        x = []
        for name in dim_names:
            info = self.knobs[name]
            val = knobs_dict.get(name, info["default"])
            if info["type"] == "memory":
                x.append(float(parse_memory(str(val))))
            elif info["type"] == "integer":
                x.append(int(val))
            elif info["type"] == "float":
                x.append(float(val))
            elif info["type"] == "enum":
                x.append(str(val))
        return x

    def vector_to_knobs(self, x: list, dim_names: list) -> dict:
        """skopt 向量 → knob dict"""
        knobs = {}
        for i, name in enumerate(dim_names):
            info = self.knobs[name]
            val = x[i]
            if info["type"] == "memory":
                knobs[name] = format_memory(int(round(val)))
            elif info["type"] == "integer":
                knobs[name] = int(round(val))
            elif info["type"] == "float":
                knobs[name] = round(float(val), 4)
            elif info["type"] == "enum":
                knobs[name] = str(val)
        return knobs

