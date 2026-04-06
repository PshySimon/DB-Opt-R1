"""
GRPO 数据预处理：将场景数据转换为 verl 所需的 Parquet 格式

GRPO 只需要 prompt（不需要 answer），模型自己 rollout 生成回复。

输入：data_pipeline/data/scenarios/collected.json
输出：datasets/grpo/train.parquet + datasets/grpo/validation.parquet

Usage:
    cd project/db-opt-r1
    python3 -m datasets.preprocess_grpo \
        --input data_pipeline/data/scenarios/collected.json \
        --output_dir datasets/grpo/ \
        --val_ratio 0.1
"""

import json
import argparse
import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# System prompt（与 SFT 保持一致）
SYSTEM_PROMPT = """你是一个专业的数据库性能调优助手。你的任务是通过调整 PostgreSQL 配置参数（knobs）来优化数据库性能。

你可以使用以下工具来获取信息和执行操作：
- get_system_info: 获取系统硬件信息
- get_knob_info: 查询某个 knob 的详细信息
- set_knob: 设置 knob 值
- get_db_status: 获取数据库运行状态
- run_benchmark: 运行性能测试
- analyze_workload: 分析工作负载特征
- get_wait_events: 获取等待事件
- get_slow_queries: 获取慢查询
- get_current_config: 获取当前配置

在每一步操作前，请先用 <think> 标签思考你的调优策略。"""


def build_user_message(scenario: dict) -> str:
    """根据场景数据构建 user prompt"""
    hardware = scenario.get("hardware", {})
    workload = scenario.get("workload", {})
    description = scenario.get("description", "")

    parts = ["请优化这个数据库的性能。\n"]

    if description:
        parts.append(f"**场景描述**：{description}\n")

    if hardware:
        hw_info = []
        if hardware.get("cpu_count"):
            hw_info.append(f"CPU: {hardware['cpu_count']} 核")
        if hardware.get("total_memory_gb"):
            hw_info.append(f"内存: {hardware['total_memory_gb']} GB")
        if hardware.get("disk_type"):
            hw_info.append(f"磁盘: {hardware['disk_type']}")
        if hw_info:
            parts.append(f"**硬件环境**：{', '.join(hw_info)}\n")

    if workload:
        wl_info = []
        if workload.get("type"):
            wl_info.append(f"类型: {workload['type']}")
        if workload.get("tps_current"):
            wl_info.append(f"当前 TPS: {workload['tps_current']:.1f}")
        if workload.get("latency_avg_ms"):
            wl_info.append(f"平均延迟: {workload['latency_avg_ms']:.1f} ms")
        if wl_info:
            parts.append(f"**工作负载**：{', '.join(wl_info)}\n")

    parts.append("请分析当前状况并给出调优建议。")

    return "\n".join(parts)


def process_scenarios(scenarios: list) -> list:
    """将场景列表转换为 GRPO 训练数据"""
    records = []

    for idx, scenario in enumerate(scenarios):
        user_msg = build_user_message(scenario)

        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        # ground_truth 用于 reward 计算
        ground_truth = {
            "scenario_idx": idx,
            "hardware": scenario.get("hardware", {}),
        }

        record = {
            "prompt": prompt,
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth,
            },
            "data_source": "db_tuning",
        }
        records.append(record)

    return records


def main():
    parser = argparse.ArgumentParser(description="GRPO 数据预处理")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="场景数据 JSON 文件路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/grpo/",
        help="输出目录",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="验证集比例",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    args = parser.parse_args()

    # 加载场景数据
    with open(args.input, "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    logger.info(f"加载了 {len(scenarios)} 个场景")

    # 转换为训练数据
    records = process_scenarios(scenarios)

    # 划分训练/验证集
    import random
    random.seed(args.seed)
    random.shuffle(records)

    n_val = max(1, int(len(records) * args.val_ratio))
    val_records = records[:n_val]
    train_records = records[n_val:]

    logger.info(f"训练集: {len(train_records)} 条, 验证集: {len(val_records)} 条")

    # 保存为 Parquet
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split, data in [("train", train_records), ("validation", val_records)]:
        df = pd.DataFrame(data)
        output_path = output_dir / f"{split}.parquet"
        df.to_parquet(output_path, index=False)
        logger.info(f"已保存: {output_path} ({len(data)} 条)")


if __name__ == "__main__":
    main()
