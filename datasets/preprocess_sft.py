"""
将 SFT JSONL 轨迹转换为 verl 所需的 Parquet 格式

输入：MCTS 输出的 trajectories.jsonl 或手工编写的 cold_start.jsonl
输出：verl fsdp_sft_trainer 所需的 train.parquet / validation.parquet

Usage:
    cd project/db-opt-r1
    python -m datasets.preprocess_sft \
        --input_files datasets/sft/cold_start.jsonl \
        --output_dir datasets/sft/ \
        --train_ratio 0.95
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

def messages_to_qa(messages: list) -> dict:
    """将 chat messages 格式转为 verl SFT 所需的 question/answer 格式。

    策略：把多轮工具交互打平成一个长 answer（与 Agent-R1 一致）。
      - question = system + 第一个 user turn
      - answer   = 所有后续 assistant/tool turn
    """
    question_parts = []
    answer_parts = []

    seen_first_user = False
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            question_parts.append(
                f"<|im_start|>system\n{content}<|im_end|>"
            )
        elif role == "user" and not seen_first_user:
            question_parts.append(
                f"<|im_start|>user\n{content}<|im_end|>"
            )
            seen_first_user = True
        elif role == "assistant":
            answer_parts.append(
                f"<|im_start|>assistant\n{content}<|im_end|>"
            )
        elif role in ("tool", "user"):
            # tool response 或后续 user turn 都编码为 user turn
            body = content
            if role == "tool":
                body = f"<tool_response>\n{content}\n</tool_response>"
            answer_parts.append(
                f"<|im_start|>user\n{body}\n<|im_end|>"
            )

    return {
        "question": "\n".join(question_parts),
        "answer": "\n".join(answer_parts),
    }


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
        description="将 SFT 轨迹 JSONL 转换为 verl Parquet 格式"
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
        messages = item.get("messages", [])

        # 过滤过短轨迹
        assistant_turns = sum(1 for m in messages if m["role"] == "assistant")
        if assistant_turns < args.min_turns:
            skipped += 1
            continue

        qa = messages_to_qa(messages)

        # 检查 answer 非空
        if not qa["answer"].strip():
            skipped += 1
            continue

        converted.append({
            "extra_info": qa,
            "data_source": "db_tuning",
        })

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
        q = sample["extra_info"]["question"]
        a = sample["extra_info"]["answer"]
        logger.info(f"\n{'='*60}\n样例 question (前 200 字):\n{q[:200]}...")
        logger.info(f"\n样例 answer (前 300 字):\n{a[:300]}...")


if __name__ == "__main__":
    main()
