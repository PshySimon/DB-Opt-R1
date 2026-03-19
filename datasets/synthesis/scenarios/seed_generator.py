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
