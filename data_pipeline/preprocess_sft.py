"""
将 SFT JSONL 轨迹转换为 verl 0.7.1 所需的多轮 Parquet 格式。

输入：包含 ``messages`` 字段的 JSONL 轨迹。
输出：verl ``sft_trainer`` 所需的 ``train.parquet`` / ``validation.parquet``，
字段以多轮会话列为主，例如 ``messages``、``tools``、``enable_thinking``。
"""

import argparse
import json
import os
import random
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 格式转换
# ============================================================

def convert_record(item: dict, min_turns: int = 2) -> dict | None:
    """将单条 JSONL 轨迹转换为 verl 0.7.1 的多轮 SFT 样本。

    verl 0.7.1 的 ``sft_trainer`` 默认读取 ``messages`` 列，而不是
    ``question/answer`` 或自定义 ``prompt_key/response_key``。
    """
    messages = item.get("messages", [])
    if not messages:
        return None

    assistant_turns = sum(1 for m in messages if m.get("role") == "assistant")
    if assistant_turns < min_turns:
        return None

    converted = {
        "messages": messages,
        "data_source": item.get("data_source", "db_tuning"),
    }

    if "tools" in item:
        converted["tools"] = item["tools"]
    if "enable_thinking" in item:
        converted["enable_thinking"] = item["enable_thinking"]

    return converted


def load_jsonl(path: str) -> list:
    """加载 JSONL 文件"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"  跳过第 {i} 行 (JSON 解析失败): {e}")
    return records


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="将 SFT 轨迹 JSONL 转换为 verl 0.7.1 多轮 Parquet 格式"
    )
    parser.add_argument(
        "--input_files", nargs="+", required=True,
        help="输入的 JSONL 文件路径（可多个，会合并）"
    )
    parser.add_argument(
        "--output_dir", default="datasets/sft/",
        help="Parquet 输出目录"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.95,
        help="训练集比例"
    )
    parser.add_argument(
        "--min_turns", type=int, default=2,
        help="最少交互轮数（过滤过短轨迹）"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子"
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # 1. 加载所有 JSONL
    all_records = []
    for fpath in args.input_files:
        logger.info(f"加载 {fpath}")
        items = load_jsonl(fpath)
        logger.info(f"  读取 {len(items)} 条")
        all_records.extend(items)

    logger.info(f"总计 {len(all_records)} 条轨迹")

    # 2. 转换格式 + 过滤
    converted = []
    skipped = 0
    for item in all_records:
        converted_item = convert_record(item, min_turns=args.min_turns)
        if converted_item is None:
            skipped += 1
            continue
        converted.append(converted_item)

    if skipped:
        logger.info(f"过滤 {skipped} 条（过短或空 answer）")
    logger.info(f"有效样本: {len(converted)}")

    if not converted:
        logger.error("没有有效样本，请检查输入文件")
        return

    # 3. 打乱 + 划分 train/val
    random.shuffle(converted)
    split_idx = int(len(converted) * args.train_ratio)
    train_records = converted[:split_idx]
    val_records = converted[split_idx:]

    # 确保 val 至少 1 条
    if not val_records and len(train_records) > 1:
        val_records = [train_records.pop()]

    # 4. 保存 parquet
    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, "train.parquet")
    val_path = os.path.join(args.output_dir, "validation.parquet")

    pd.DataFrame(train_records).to_parquet(train_path, index=False)
    pd.DataFrame(val_records).to_parquet(val_path, index=False)

    logger.info(f"保存: {len(train_records)} train → {train_path}")
    logger.info(f"保存: {len(val_records)} val   → {val_path}")

    # 5. 打印样例
    if converted:
        sample = converted[0]
        logger.info(f"\n{'='*60}\n样例 messages 条数: {len(sample['messages'])}")
        logger.info(f"首条消息: {sample['messages'][0]}")
        if "tools" in sample:
            logger.info(f"tools 数量: {len(sample['tools'])}")


if __name__ == "__main__":
    main()
