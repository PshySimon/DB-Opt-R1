"""
PostgreSQL knob validation shared by real and simulated DB tools.

The validator combines the task action space from ``knob_space.yaml`` with a
PostgreSQL ``pg_settings`` catalog snapshot. It intentionally reports messages
from an operator-facing perspective and does not expose implementation details
about training or offline models.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import yaml

from core.db.knob_space import parse_memory


MEMORY_UNIT_TO_KB = {
    "B": 1 / 1024,
    "kB": 1,
    "KB": 1,
    "8kB": 8,
    "MB": 1024,
    "GB": 1024 * 1024,
    "TB": 1024 * 1024 * 1024,
}

TIME_UNIT_TO_MS = {
    "us": 0.001,
    "ms": 1,
    "s": 1000,
    "m": 60 * 1000,
    "min": 60 * 1000,
    "h": 60 * 60 * 1000,
    "d": 24 * 60 * 60 * 1000,
}

BOOL_TRUE = {"on", "true", "yes", "1"}
BOOL_FALSE = {"off", "false", "no", "0"}


class KnobValidator:
    """Validate and canonicalize requested PostgreSQL knobs."""

    def __init__(self, knob_space_path: str, pg_catalog_path: str | None = None):
        with open(knob_space_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self.knobs = config.get("knobs", {})
        self.catalog = {}
        if pg_catalog_path and Path(pg_catalog_path).exists():
            self.catalog = json.loads(Path(pg_catalog_path).read_text(encoding="utf-8"))

    def validate(self, knobs: dict[str, Any]) -> tuple[dict[str, Any], list[dict], list[dict]]:
        """Return ``(accepted, failed, ignored)`` for a requested knob dict."""
        accepted: dict[str, Any] = {}
        failed: list[dict] = []
        ignored: list[dict] = []

        for name, value in knobs.items():
            spec = self.knobs.get(name)
            if spec is None:
                ignored.append(
                    {
                        "name": name,
                        "value": str(value),
                        "warning": "该参数不在当前任务允许调整范围内，本次未修改",
                    }
                )
                continue

            pg_spec = self.catalog.get(name)
            if self.catalog and pg_spec is None:
                failed.append(
                    {
                        "name": name,
                        "value": str(value),
                        "error": "当前 PostgreSQL 版本不支持该参数",
                    }
                )
                continue

            try:
                accepted[name] = self._canonicalize(name, value, spec, pg_spec or {})
            except ValueError as exc:
                failed.append({"name": name, "value": str(value), "error": str(exc)})

        return accepted, failed, ignored

    def _canonicalize(self, name: str, value: Any, spec: dict, pg_spec: dict) -> Any:
        pg_type = pg_spec.get("vartype")
        knob_type = spec.get("type")

        if pg_type == "bool":
            canonical = self._canonical_bool(value)
            allowed = {str(v).lower() for v in spec.get("values", ["on", "off"])}
            if canonical not in allowed:
                raise ValueError(f"参数值超出当前任务允许范围，可选值为: {', '.join(sorted(allowed))}")
            return canonical

        if pg_type == "enum" or knob_type == "enum":
            return self._canonical_enum(name, value, spec, pg_spec)

        if knob_type == "memory":
            return self._canonical_memory(value, spec, pg_spec)

        if knob_type == "integer" or pg_type == "integer":
            return self._canonical_integer(value, spec, pg_spec)

        if knob_type == "float" or pg_type == "real":
            return self._canonical_float(value, spec, pg_spec)

        return str(value)

    def _canonical_bool(self, value: Any) -> str:
        s = str(value).strip().lower()
        if s in BOOL_TRUE:
            return "on"
        if s in BOOL_FALSE:
            return "off"
        raise ValueError("参数值无效，需要布尔值 on/off")

    def _canonical_enum(self, name: str, value: Any, spec: dict, pg_spec: dict) -> str:
        s = str(value).strip().lower()
        task_values = [str(v).lower() for v in spec.get("values", [])]
        pg_values = [str(v).lower() for v in (pg_spec.get("enumvals") or [])]

        if pg_values and s not in pg_values:
            raise ValueError(f"参数值无效，可选值为: {', '.join(pg_values)}")
        if task_values and s not in task_values:
            raise ValueError(f"参数值超出当前任务允许范围，可选值为: {', '.join(task_values)}")
        if not task_values and not pg_values:
            raise ValueError("参数值无效，缺少可选值定义")
        return s

    def _canonical_memory(self, value: Any, spec: dict, pg_spec: dict) -> str:
        value_kb = self._parse_memory_kb(value)

        ranges: list[tuple[float | None, float | None, str]] = []
        if "min" in spec or "max" in spec:
            ranges.append(
                (
                    self._parse_memory_kb(spec["min"]) if spec.get("min") is not None else None,
                    self._parse_memory_kb(spec["max"]) if spec.get("max") is not None else None,
                    "当前任务允许范围",
                )
            )

        pg_min, pg_max = self._pg_numeric_range(pg_spec)
        pg_unit = pg_spec.get("unit")
        if pg_unit in MEMORY_UNIT_TO_KB:
            mult = MEMORY_UNIT_TO_KB[pg_unit]
            ranges.append(
                (
                    float(pg_min) * mult if pg_min is not None else None,
                    float(pg_max) * mult if pg_max is not None else None,
                    "PostgreSQL 允许范围",
                )
            )

        self._check_ranges(value_kb, ranges)
        return self._format_memory_like_show(value)

    def _canonical_integer(self, value: Any, spec: dict, pg_spec: dict) -> str:
        unit = pg_spec.get("unit") if pg_spec else spec.get("unit")
        if unit in TIME_UNIT_TO_MS:
            numeric = self._parse_time_to_unit(value, unit)
        elif unit in MEMORY_UNIT_TO_KB:
            numeric = self._parse_memory_kb(value) / MEMORY_UNIT_TO_KB[unit]
        else:
            numeric = self._parse_plain_integer(value)

        if not math.isclose(numeric, round(numeric), rel_tol=0, abs_tol=1e-9):
            raise ValueError("参数值无效，需要整数值")
        numeric_int = int(round(numeric))

        ranges = []
        if "min" in spec or "max" in spec:
            ranges.append(
                (
                    self._parse_range_number(spec.get("min"), unit) if spec.get("min") is not None else None,
                    self._parse_range_number(spec.get("max"), unit) if spec.get("max") is not None else None,
                    "当前任务允许范围",
                )
            )

        pg_min, pg_max = self._pg_numeric_range(pg_spec)
        ranges.append((pg_min, pg_max, "PostgreSQL 允许范围"))
        self._check_ranges(numeric_int, ranges)
        if unit in TIME_UNIT_TO_MS:
            return self._format_time_like_show(numeric_int, unit)
        if unit in MEMORY_UNIT_TO_KB:
            return self._format_memory_from_kb(numeric_int * MEMORY_UNIT_TO_KB[unit])
        return str(numeric_int)

    def _canonical_float(self, value: Any, spec: dict, pg_spec: dict) -> str:
        try:
            numeric = float(str(value).strip())
        except (TypeError, ValueError):
            raise ValueError("参数值无效，需要数值")

        ranges = []
        if "min" in spec or "max" in spec:
            ranges.append(
                (
                    float(spec["min"]) if spec.get("min") is not None else None,
                    float(spec["max"]) if spec.get("max") is not None else None,
                    "当前任务允许范围",
                )
            )
        pg_min, pg_max = self._pg_numeric_range(pg_spec)
        ranges.append((pg_min, pg_max, "PostgreSQL 允许范围"))
        self._check_ranges(numeric, ranges)
        return f"{numeric:g}"

    def _parse_memory_kb(self, value: Any) -> float:
        try:
            return float(parse_memory(str(value)))
        except (TypeError, ValueError):
            raise ValueError("参数值无效，需要合法内存值，如 128MB 或 4GB")

    def _parse_plain_integer(self, value: Any) -> float:
        s = str(value).strip()
        if not re.fullmatch(r"[-+]?\d+", s):
            raise ValueError("参数值无效，需要整数值")
        return float(int(s))

    def _parse_time_to_unit(self, value: Any, target_unit: str) -> float:
        s = str(value).strip()
        if re.fullmatch(r"[-+]?\d+(\.\d+)?", s):
            return float(s)

        m = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)\s*(us|ms|s|m|min|h|d)", s, re.IGNORECASE)
        if not m:
            raise ValueError(f"参数值无效，需要整数值或带时间单位的值（目标单位 {target_unit}）")
        number = float(m.group(1))
        source_unit = m.group(2).lower()
        value_ms = number * TIME_UNIT_TO_MS[source_unit]
        return value_ms / TIME_UNIT_TO_MS[target_unit]

    def _parse_range_number(self, value: Any, unit: str | None) -> float:
        if unit in TIME_UNIT_TO_MS:
            return self._parse_time_to_unit(value, unit)
        if unit in MEMORY_UNIT_TO_KB:
            return self._parse_memory_kb(value) / MEMORY_UNIT_TO_KB[unit]
        return float(value)

    def _pg_numeric_range(self, pg_spec: dict) -> tuple[float | None, float | None]:
        def parse_bound(raw):
            if raw is None:
                return None
            try:
                return float(raw)
            except (TypeError, ValueError):
                return None

        return parse_bound(pg_spec.get("min_val")), parse_bound(pg_spec.get("max_val"))

    def _check_ranges(self, value: float, ranges: list[tuple[float | None, float | None, str]]) -> None:
        for lower, upper, label in ranges:
            if lower is not None and value < lower:
                raise ValueError(f"参数值低于{label}下限 {lower:g}")
            if upper is not None and value > upper:
                raise ValueError(f"参数值超过{label}上限 {upper:g}")

    def _format_memory_like_show(self, value: Any) -> str:
        s = str(value).strip()
        m = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)\s*(kB|KB|MB|GB|TB|B)", s, re.IGNORECASE)
        if m:
            number = float(m.group(1))
            unit = m.group(2)
            unit_upper = unit.upper()
            if unit_upper == "GB" and not number.is_integer():
                return f"{int(round(number * 1024))}MB"
            if number.is_integer():
                return f"{int(number)}{unit}"
            return f"{number:g}{unit}"

        value_kb = int(round(self._parse_memory_kb(value)))
        return self._format_memory_from_kb(value_kb)

    def _format_memory_from_kb(self, value_kb: float) -> str:
        value_kb = int(round(value_kb))
        if value_kb >= 1024 * 1024 and value_kb % (1024 * 1024) == 0:
            return f"{value_kb // (1024 * 1024)}GB"
        if value_kb >= 1024 and value_kb % 1024 == 0:
            return f"{value_kb // 1024}MB"
        return f"{value_kb}kB"

    def _format_time_like_show(self, value_in_unit: float, unit: str) -> str:
        value_ms = float(value_in_unit) * TIME_UNIT_TO_MS[unit]
        if value_ms == 0:
            return "0"

        if unit == "s":
            seconds = value_ms / 1000
            if seconds >= 60 and math.isclose(seconds % 60, 0, rel_tol=0, abs_tol=1e-9):
                minutes = seconds / 60
                return f"{minutes:g}min"
            return f"{seconds:g}s"

        if unit == "ms":
            return f"{value_ms:g}ms"

        if unit == "us":
            return f"{value_ms / TIME_UNIT_TO_MS['us']:g}us"

        return f"{value_in_unit:g}{unit}"
