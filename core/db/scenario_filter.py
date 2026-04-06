"""
场景过滤工具函数

从 DBToolEnv 移出的过滤逻辑，由调用方（MCTS/sampler）显式控制。
"""

import logging

logger = logging.getLogger(__name__)


def filter_scenarios(scenarios, source_filter=None, tps_min=None, tps_max=None):
    """按 source 和 TPS 范围过滤场景列表。

    Args:
        scenarios: ScenarioState 列表
        source_filter: 只保留指定 source 的场景（如 'llm_generated'），None 表示不过滤
        tps_min: TPS 下限（含），None 表示不限
        tps_max: TPS 上限（含），None 表示不限

    Returns:
        过滤后的场景列表
    """
    before = len(scenarios)
    filtered = []

    for s in scenarios:
        # source 过滤
        if source_filter and getattr(s, 'source', '') != source_filter:
            continue

        # TPS 范围过滤
        if tps_min is not None or tps_max is not None:
            wl = getattr(s, 'workload', {})
            if isinstance(wl, dict):
                tps = float(wl.get('tps_current', 0) or 0)
                if tps_min is not None and tps < tps_min:
                    continue
                if tps_max is not None and tps > tps_max:
                    continue
            # 无 workload 或非 dict（如新生成的 eval set）默认保留

        filtered.append(s)

    after = len(filtered)
    if after < before:
        parts = []
        if source_filter:
            parts.append(f"source={source_filter}")
        if tps_min is not None or tps_max is not None:
            parts.append(f"TPS {tps_min or '*'}~{tps_max or '*'}")
        logger.info(f"  场景过滤: {before} → {after} 条 ({', '.join(parts)})")

    return filtered


def add_filter_args(parser):
    """为 argparse 添加场景过滤参数"""
    group = parser.add_argument_group("场景过滤")
    group.add_argument("--source-filter", default=None,
                       help="只保留指定 source 的场景（如 llm_generated），默认不过滤")
    group.add_argument("--tps-min", type=float, default=None,
                       help="TPS 下限（含），低于此值的场景被过滤")
    group.add_argument("--tps-max", type=float, default=None,
                       help="TPS 上限（含），高于此值的场景被过滤")


def apply_filter_args(scenarios, args):
    """根据命令行参数过滤场景"""
    return filter_scenarios(
        scenarios,
        source_filter=getattr(args, 'source_filter', None),
        tps_min=getattr(args, 'tps_min', None),
        tps_max=getattr(args, 'tps_max', None),
    )
