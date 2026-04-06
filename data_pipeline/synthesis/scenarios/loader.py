"""
data_pipeline.synthesis.scenarios.loader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
场景数据加载与清洗的公共逻辑。

因为去重以 knob 配置为 key，属于业务逻辑，放在 scenarios 子包里。
cost_model、environment 等模块统一从这里加载数据，避免散弹式修改。

公共接口
--------
knob_fingerprint(knobs, wl_type) -> str
    计算 (knob配置, workload类型) 的 MD5 指纹，用作去重 key。

dedup_scenarios(items, fname='', logger=None) -> list[dict]
    对 raw JSON dict list 去重：
    同文件内 knob + workload 完全相同，但 TPS 差异 >20% 的条目，
    只保留 TPS 最高的那条。

load_scenario_files(source, logger=None) -> list[dict]
    从单文件或目录（自动 glob collected_*.json）加载 raw dict list，
    并在每个文件内做去重。返回合并后的原始 dict 列表。
"""

import glob
import hashlib
import json
import logging
import os
from collections import defaultdict

_logger = logging.getLogger(__name__)


def knob_fingerprint(knobs: dict, wl_type: str) -> str:
    """(knob配置, workload类型) → MD5 字符串，用作去重 key。"""
    payload = json.dumps(dict(sorted(knobs.items())), sort_keys=True) + "|" + wl_type
    return hashlib.md5(payload.encode()).hexdigest()


def dedup_scenarios(items: list, fname: str = "",
                    logger: logging.Logger = None) -> list:
    """
    去重：同文件内 knob + workload 相同但 TPS 差 >20% 的重复条目，
    保留 TPS 最高的那条。

    Parameters
    ----------
    items : list of dict
        从 JSON 文件直接加载的原始 dict 列表，
        每条包含 'knobs'（dict）和 'workload'（dict）字段。
    fname : str
        文件名，仅用于日志。
    logger : logging.Logger, optional
        日志对象，默认使用模块级 logger。

    Returns
    -------
    list of dict
        去重后的 dict 列表。
    """
    log = logger or _logger
    groups = defaultdict(list)

    for item in items:
        knobs = item.get("knobs", {}) or {}
        wl = item.get("workload", {}) or {}
        wl_type = wl.get("type", "") if isinstance(wl, dict) else str(wl)
        key = knob_fingerprint(knobs, wl_type)
        groups[key].append(item)

    result, n_removed = [], 0
    for entries in groups.values():
        if len(entries) == 1:
            result.extend(entries)
            continue

        def _tps(e):
            wl = e.get("workload", {}) or {}
            return float(wl.get("tps_current", 0) or 0) if isinstance(wl, dict) else 0.0

        tps_vals = [_tps(e) for e in entries]
        pos_vals = [v for v in tps_vals if v > 0]
        tps_max = max(tps_vals)
        tps_min = min(pos_vals) if pos_vals else 0

        if tps_min > 0 and tps_max / tps_min > 1.2:
            best = entries[tps_vals.index(tps_max)]
            result.append(best)
            n_removed += len(entries) - 1
        else:
            result.extend(entries)

    if n_removed > 0:
        tag = os.path.basename(fname) if fname else "data"
        log.info(f"  去重 {tag}: 移除 {n_removed} 条 TPS 矛盾样本，剩余 {len(result)} 条")

    return result


def load_scenario_files(source: str,
                        logger: logging.Logger = None) -> list:
    """
    从单 JSON 文件或目录（自动 glob collected_*.json）加载原始 dict list，
    并在每个文件内做去重。

    Parameters
    ----------
    source : str
        单个 .json 文件路径，或包含 collected_*.json 的目录路径。
    logger : logging.Logger, optional
        日志对象。

    Returns
    -------
    list of dict
        去重后的原始 dict 列表（跨文件顺序合并）。
    """
    log = logger or _logger

    if os.path.isfile(source):
        with open(source, "r", encoding="utf-8") as f:
            items = json.load(f)
        items = dedup_scenarios(items, fname=source, logger=log)
        log.info(f"  加载 {os.path.basename(source)}: {len(items)} 条")
        return items

    elif os.path.isdir(source):
        collected = sorted(glob.glob(os.path.join(source, "collected_*.json")))
        if not collected:
            collected = sorted(
                os.path.join(source, f)
                for f in os.listdir(source) if f.endswith(".json")
            )
        all_items = []
        for fpath in collected:
            with open(fpath, "r", encoding="utf-8") as f:
                items = json.load(f)
            items = dedup_scenarios(items, fname=fpath, logger=log)
            log.info(f"  加载 {os.path.basename(fpath)}: {len(items)} 条")
            all_items.extend(items)
        return all_items

    return []
