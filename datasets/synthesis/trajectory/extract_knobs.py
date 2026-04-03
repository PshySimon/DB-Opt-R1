"""
从轨迹文件中提取 LLM 预测的 knob 配置，输出为 knob_configs 格式。

用于增量扩充 cost model 训练数据：
  提取 knob → pipeline collect（真机采集 TPS）→ cost model 训练

用法:
    python3 -m datasets.synthesis.trajectory.extract_knobs \
        --trajectories eval_results/sft_qwen3_4b_cleaned/sft_trajectories.jsonl \
                       datasets/data/train/sft_trajectories_cleaned.jsonl \
        --scenarios datasets/data/scenarios/collected/ \
        --output datasets/data/scenarios/knobs/knob_configs_incremental_v1.json
"""

import argparse
import json
import logging
import re
import os
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_knobs_from_trajectory(messages: list) -> list[dict]:
    """从一条轨迹的 messages 中提取所有 set_knob 调用的 knob 配置。

    返回每次 set_knob 调用后的累积 knob 配置（后面的覆盖前面的）。
    """
    accumulated_knobs = {}

    for msg in messages:
        if msg["role"] != "assistant":
            continue
        content = msg.get("content", "")
        if "set_knob" not in content:
            continue

        match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', content, re.DOTALL)
        if not match:
            continue

        try:
            call = json.loads(match.group(1))
            if call.get("name") != "set_knob":
                continue

            knobs_arg = call.get("arguments", {})
            if isinstance(knobs_arg, str):
                try:
                    knobs_arg = json.loads(knobs_arg)
                except json.JSONDecodeError:
                    continue
            knobs_str = knobs_arg.get("knobs", "{}")
            if isinstance(knobs_str, str):
                knobs = json.loads(knobs_str)
            else:
                knobs = knobs_str

            accumulated_knobs.update(knobs)
        except (json.JSONDecodeError, KeyError, TypeError):
            continue

    return accumulated_knobs


def load_scenario_hardware(scenario_dir: str) -> dict:
    """加载所有 collected_*.json，建立 idx → (hardware, workload) 的映射。"""
    hw_map = {}

    if os.path.isdir(scenario_dir):
        files = sorted(f for f in os.listdir(scenario_dir)
                       if f.startswith("collected") and f.endswith(".json"))
        all_scenarios = []
        for fname in files:
            path = os.path.join(scenario_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                scenarios = json.load(f)
            for s in scenarios:
                all_scenarios.append(s)
        for idx, s in enumerate(all_scenarios):
            hw_map[idx] = {
                "hardware": s.get("hardware", {}),
                "workload": s.get("workload", {}),
                "name": s.get("name", f"env_{idx}"),
                "description": s.get("description", ""),
            }
    else:
        with open(scenario_dir, "r", encoding="utf-8") as f:
            scenarios = json.load(f)
        for idx, s in enumerate(scenarios):
            hw_map[idx] = {
                "hardware": s.get("hardware", {}),
                "workload": s.get("workload", {}),
                "name": s.get("name", f"env_{idx}"),
                "description": s.get("description", ""),
            }

    return hw_map


def main():
    parser = argparse.ArgumentParser(description="从轨迹中提取 LLM 预测的 knob 配置")
    parser.add_argument("--trajectories", nargs="+", required=True,
                        help="轨迹 JSONL 文件（可多个）")
    parser.add_argument("--scenarios", default=None,
                        help="场景目录或文件（用于关联 hardware/workload）")
    parser.add_argument("--output", required=True,
                        help="输出 knob_configs JSON 文件路径")
    parser.add_argument("--dedup", action="store_true", default=True,
                        help="按 knob 配置去重")
    args = parser.parse_args()

    # 加载场景 hardware 映射
    hw_map = {}
    if args.scenarios:
        hw_map = load_scenario_hardware(args.scenarios)
        logger.info(f"加载场景: {len(hw_map)} 个")

    # 从轨迹中提取 knobs
    results = []
    seen_knob_hashes = set()
    total_traj = 0
    skipped_empty = 0

    for traj_file in args.trajectories:
        logger.info(f"处理: {traj_file}")
        with open(traj_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                total_traj += 1
                messages = item.get("messages", [])
                env_idx = item.get("env_sample_idx")

                # 提取累积 knob 配置
                knobs = extract_knobs_from_trajectory(messages)
                if not knobs:
                    skipped_empty += 1
                    continue

                # 去重
                if args.dedup:
                    knob_hash = json.dumps(knobs, sort_keys=True)
                    if knob_hash in seen_knob_hashes:
                        continue
                    seen_knob_hashes.add(knob_hash)

                # 构建 knob_config 记录
                record = {
                    "name": f"llm_predicted_{len(results)}",
                    "source": "llm_trajectory",
                    "knobs": knobs,
                }

                # 关联场景信息
                if env_idx is not None and env_idx in hw_map:
                    scenario = hw_map[env_idx]
                    record["hardware"] = scenario["hardware"]
                    record["workload"] = scenario["workload"].get("type", "mixed") \
                        if isinstance(scenario["workload"], dict) else str(scenario["workload"])
                    record["description"] = scenario.get("description", "")

                results.append(record)

    # 输出
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(
        f"完成: {total_traj} 条轨迹 → {len(results)} 条 knob 配置 "
        f"(跳过: {skipped_empty} 无 set_knob, "
        f"{total_traj - skipped_empty - len(results)} 重复)"
    )
    logger.info(f"输出: {args.output}")


if __name__ == "__main__":
    main()
