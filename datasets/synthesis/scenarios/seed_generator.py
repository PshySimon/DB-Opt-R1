"""
Knob 驱动的种子生成器

三层策略：
- Layer 1: 单 knob 极端值（~50 条）
- Layer 2: 预定义关联组合（~30 条）
- Layer 3: 跨类别随机组合（~20 条）

用法:
    from datasets.synthesis.scenarios.seed_generator import generate_all_seeds
    seeds = generate_all_seeds("configs/knob_effects.yaml", "configs/knob_space.yaml")
"""

import yaml
import random
import itertools
import logging

logger = logging.getLogger(__name__)


def load_knob_effects(effects_path: str) -> dict:
    with open(effects_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_knob_space(space_path: str) -> dict:
    with open(space_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("knobs", {})


def _layer1_single_knob(effects: dict, knob_space: dict) -> list:
    """Layer 1: 每个 knob 生成 1-2 条种子（极低/极高）"""
    seeds = []
    knobs_info = effects.get("knobs", {})

    for knob_name, info in knobs_info.items():
        if knob_name not in knob_space:
            continue

        category = info.get("category", "misc")
        space = knob_space[knob_name]

        # 极低方向
        too_low = info.get("too_low") or info.get("off")  # enum 类型用 off
        if too_low:
            seeds.append({
                "name": f"{knob_name}_too_low",
                "difficulty": 1,
                "category": category,
                "knobs": {knob_name: {"direction": "min"}},
                "expected_effects": [too_low],
                "description": f"{knob_name} 设置过低：{too_low}",
            })

        # 极高方向
        too_high = info.get("too_high")
        if too_high:
            seeds.append({
                "name": f"{knob_name}_too_high",
                "difficulty": 1,
                "category": category,
                "knobs": {knob_name: {"direction": "max"}},
                "expected_effects": [too_high],
                "description": f"{knob_name} 设置过高：{too_high}",
            })

    return seeds


def _layer2_combinations(effects: dict) -> list:
    """Layer 2: 预定义的关联 knob 组合"""
    seeds = []
    combos = effects.get("combinations", [])

    for combo in combos:
        knobs = {}
        for knob_name, direction in combo["knobs"].items():
            knobs[knob_name] = {"direction": direction}

        difficulty = min(len(knobs), 3)
        seeds.append({
            "name": combo["name"],
            "difficulty": difficulty,
            "category": combo.get("category", "combo"),
            "knobs": knobs,
            "expected_effects": [combo["description"]],
            "description": combo["description"],
        })

    return seeds


def _layer3_cross_category(effects: dict, knob_space: dict, count: int = 20) -> list:
    """Layer 3: 跨类别随机组合"""
    seeds = []
    knobs_info = effects.get("knobs", {})

    # 按 category 分组，每组取有效果的 knob
    by_category = {}
    for knob_name, info in knobs_info.items():
        if knob_name not in knob_space:
            continue
        cat = info.get("category", "misc")

        directions = []
        if info.get("too_low") or info.get("off"):
            directions.append("min")
        if info.get("too_high"):
            directions.append("max")
        if not directions:
            continue

        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append((knob_name, directions, info))

    # 过滤掉空 category
    categories = [c for c, v in by_category.items() if v]
    used_combos = set()
    attempts = 0

    while len(seeds) < count and attempts < count * 10:
        attempts += 1

        # 随机选 2-3 个不同 category
        n_cats = random.choice([2, 2, 3])  # 偏向 2 个
        if len(categories) < n_cats:
            continue
        selected_cats = random.sample(categories, n_cats)

        # 每个 category 取 1 个 knob
        knobs = {}
        parts = []
        for cat in selected_cats:
            knob_name, directions, info = random.choice(by_category[cat])
            direction = random.choice(directions)
            knobs[knob_name] = {"direction": direction}
            effect = info.get("too_low" if direction == "min" else "too_high", "")
            if not effect:
                effect = info.get("off", "")
            parts.append(f"{knob_name}({direction})")

        # 去重
        combo_key = tuple(sorted(knobs.keys()))
        if combo_key in used_combos:
            continue
        used_combos.add(combo_key)

        name = "cross_" + "_".join(sorted(selected_cats[:2]))
        name += f"_{len(seeds)}"

        seeds.append({
            "name": name,
            "difficulty": min(len(knobs), 3),
            "category": "combo",
            "knobs": knobs,
            "expected_effects": [f"{k}: {v['direction']}" for k, v in knobs.items()],
            "description": "跨类别组合故障：" + "，".join(parts),
        })

    return seeds


def generate_all_seeds(effects_path: str, knob_space_path: str,
                       layer3_count: int = 20) -> list:
    """生成全部种子"""
    effects = load_knob_effects(effects_path)
    knob_space = load_knob_space(knob_space_path)

    l1 = _layer1_single_knob(effects, knob_space)
    l2 = _layer2_combinations(effects)
    l3 = _layer3_cross_category(effects, knob_space, count=layer3_count)

    logger.info(f"Layer 1 (单 knob): {len(l1)} 条")
    logger.info(f"Layer 2 (关联组合): {len(l2)} 条")
    logger.info(f"Layer 3 (跨类组合): {len(l3)} 条")

    all_seeds = l1 + l2 + l3
    logger.info(f"总计: {len(all_seeds)} 条种子")

    return all_seeds


def resolve_knob_value(knob_name: str, direction: str, knob_space: dict,
                       hardware: dict = None, jitter: float = 0.1) -> object:
    """根据 direction 和 knob 定义生成具体值

    Args:
        knob_name: knob 名称
        direction: "min"/"max" 或具体枚举值（如 "off", "on"）
        knob_space: knob 定义
        hardware: 硬件信息
        jitter: 随机扰动比例（0~1），用于 variants 差异化
    """
    spec = knob_space.get(knob_name)
    if not spec:
        return None

    knob_type = spec.get("type")

    # 枚举类型直接返回
    if knob_type == "enum":
        if direction in spec.get("values", []):
            return direction
        if direction == "min":
            return spec["values"][0]
        if direction == "max":
            return spec["values"][-1]
        return spec.get("default")

    # 数值类型
    if knob_type in ("integer", "float"):
        lo = spec["min"]
        hi = spec["max"]
        rng = hi - lo

        if direction == "min":
            base = lo
            val = base + rng * jitter * random.random()
        elif direction == "max":
            base = hi
            val = base - rng * jitter * random.random()
        else:
            val = spec.get("default", lo)

        if knob_type == "integer":
            return max(lo, min(hi, int(round(val))))
        else:
            return max(lo, min(hi, round(val, 4)))

    # memory 类型（如 "32MB"）
    if knob_type == "memory":
        from core.db.knob_space import parse_memory, format_memory
        lo_bytes = parse_memory(str(spec["min"]))
        hi_bytes = parse_memory(str(spec["max"]))
        rng = hi_bytes - lo_bytes

        if direction == "min":
            val = lo_bytes + int(rng * jitter * random.random())
        elif direction == "max":
            val = hi_bytes - int(rng * jitter * random.random())
        else:
            val = parse_memory(str(spec.get("default", spec["min"])))

        # 格式化为人类可读
        return format_memory(max(lo_bytes, min(hi_bytes, val)))

    return None


def seeds_to_knob_configs(seeds: list, knob_space_path: str,
                          hardware: dict, variants: int = 5,
                          difficulty_ratio: str = "2:5:3") -> list:
    """将种子转换为具体的 knob 配置（不需要 LLM）

    Args:
        difficulty_ratio: 难度比例 easy:medium:hard，如 "2:5:3"
            会按比例分配每个难度的 variants 数量
    """
    knob_space = load_knob_space(knob_space_path)

    # 解析难度比例
    ratio_parts = [int(x) for x in difficulty_ratio.split(":")]
    if len(ratio_parts) != 3:
        ratio_parts = [2, 5, 3]
    ratio_sum = sum(ratio_parts)

    # 按难度分组
    by_difficulty = {1: [], 2: [], 3: []}
    for seed in seeds:
        d = min(seed.get("difficulty", 1), 3)
        by_difficulty[d].append(seed)

    # 计算每个难度的 variants 数（按比例分配总预算）
    total_seeds = len(seeds)
    total_budget = total_seeds * variants  # 总配置数预算

    difficulty_variants = {}
    for d, ratio in zip([1, 2, 3], ratio_parts):
        n_seeds = len(by_difficulty[d])
        if n_seeds == 0:
            difficulty_variants[d] = 0
            continue
        # 该难度应占的配置数
        target_count = total_budget * ratio / ratio_sum
        # 每个种子需要的 variants 数
        v = max(1, int(round(target_count / n_seeds)))
        difficulty_variants[d] = v

    logger.info(f"难度比例 {difficulty_ratio}: "
                f"d1({len(by_difficulty[1])}种子×{difficulty_variants.get(1,0)}v), "
                f"d2({len(by_difficulty[2])}种子×{difficulty_variants.get(2,0)}v), "
                f"d3({len(by_difficulty[3])}种子×{difficulty_variants.get(3,0)}v)")

    configs = []
    for seed in seeds:
        d = min(seed.get("difficulty", 1), 3)
        n_variants = difficulty_variants.get(d, variants)

        for v in range(n_variants):
            knobs = {}
            jitter = 0.05 + 0.15 * (v / max(n_variants - 1, 1))

            for knob_name, spec in seed["knobs"].items():
                direction = spec["direction"]
                val = resolve_knob_value(knob_name, direction, knob_space,
                                        hardware=hardware, jitter=jitter)
                if val is not None:
                    knobs[knob_name] = val

            if not knobs:
                continue

            workload = random.choice(["mixed", "read_only", "write_heavy", "high_concurrency"])

            configs.append({
                "name": seed["name"],
                "variant": v,
                "difficulty": seed["difficulty"],
                "description": seed["description"],
                "category": seed.get("category", ""),
                "workload": workload,
                "expected_effects": seed.get("expected_effects", []),
                "knobs": knobs,
                "hardware_hint": hardware,
            })

    return configs


if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser(description="程序化种子生成器")
    parser.add_argument("--effects", default="configs/knob_effects.yaml")
    parser.add_argument("--knob-space", default="configs/knob_space.yaml")
    parser.add_argument("--output", default="datasets/data/scenarios/seeds.json")
    parser.add_argument("--layer3-count", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    seeds = generate_all_seeds(args.effects, args.knob_space, args.layer3_count)

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(seeds, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ 保存到 {args.output}")
