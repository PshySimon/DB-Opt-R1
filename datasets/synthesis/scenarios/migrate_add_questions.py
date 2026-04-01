"""
迁移脚本：为存量 ScenarioState 数据补充 question 字段

对 question 为空的条目，基于其真实 DB 状态调 LLM 生成 question，
复用 pipeline.generate_question_for_state。

用法:
    python3 -m datasets.synthesis.scenarios.migrate_add_questions \\
        --input datasets/data/scenarios/ \\
        --model gpt-5 \\
        --api-key sk-xxx \\
        --api-base https://xxx/v1 \\
        --workers 10

断点续跑：已有 question 的条目会跳过，每完成一条立即写回。
"""

import argparse
import glob
import json
import logging
import os
import sys
import threading
import time
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .schema import ScenarioState
from .pipeline import generate_question_for_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _build_llm_fn(args):
    """构建 LLM 生成函数（OpenAI-compatible）"""
    from openai import OpenAI

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
        timeout=120,
    )

    def generate(prompt: str, temperature: float = 0.7) -> str:
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=256,
        )
        return resp.choices[0].message.content

    return generate


def _load_json_file(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json_file(path: str, data: list):
    """原子写入（tmp → replace）"""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def migrate_file(fpath: str, llm_fn, workers: int = 5):
    """为单个 JSON 文件中 question 为空的条目生成 question"""
    raw_items = _load_json_file(fpath)

    # 只对 llm_generated 且 question 为空的条目生成 question
    pending_indices = [
        i for i, item in enumerate(raw_items)
        if not item.get("question", "")
        and item.get("source", "llm_generated") == "llm_generated"
    ]

    if not pending_indices:
        logger.info(f"  {os.path.basename(fpath)}: 全部已有 question，跳过")
        return 0

    logger.info(
        f"  {os.path.basename(fpath)}: {len(raw_items)} 条，"
        f"待生成 {len(pending_indices)} 条"
    )

    lock = threading.Lock()
    done = [0]

    def _process_one(idx):
        item = raw_items[idx]
        try:
            # 复用 ScenarioState 的 from_json 兼容逻辑
            state = ScenarioState(**{
                k: v for k, v in item.items()
                if k in ScenarioState.__dataclass_fields__
            })
            question = generate_question_for_state(state, llm_fn)
            if not question:
                raise ValueError("LLM 返回空 question")

            with lock:
                raw_items[idx]["question"] = question
                done[0] += 1
                if done[0] % 10 == 0 or done[0] == len(pending_indices):
                    logger.info(f"    进度: {done[0]}/{len(pending_indices)}")
                    _save_json_file(fpath, raw_items)

        except Exception as e:
            name = item.get("name", f"idx_{idx}")
            logger.warning(f"    [{name}] question 生成失败: {e}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_process_one, i): i for i in pending_indices}
        for fut in as_completed(futs):
            _ = fut.result()  # 异常已在 _process_one 内捕获

    # 最终写回
    _save_json_file(fpath, raw_items)
    logger.info(f"  ✅ {os.path.basename(fpath)}: 生成 {done[0]} 个 question")
    return done[0]


def main():
    parser = argparse.ArgumentParser(description="为存量 ScenarioState 补充 question 字段")
    parser.add_argument("--input", required=True,
                        help="ScenarioState JSON 文件或目录（自动 glob collected_*.json）")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--workers", type=int, default=5, help="并发线程数")
    args = parser.parse_args()

    llm_fn = _build_llm_fn(args)

    # 收集文件列表
    if os.path.isfile(args.input):
        if not os.path.basename(args.input).startswith("collected_"):
            logger.error(f"只处理 collected_*.json 文件，不支持: {args.input}")
            sys.exit(1)
        files = [args.input]
    elif os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, "collected_*.json")))
        if not files:
            logger.error(f"在 {args.input} 下未找到任何 collected_*.json 文件")
            sys.exit(1)
    else:
        logger.error(f"--input 路径不存在: {args.input}")
        sys.exit(1)

    if not files:
        logger.error("未找到任何 JSON 文件")
        sys.exit(1)

    logger.info(f"共 {len(files)} 个文件，workers={args.workers}")
    total = 0
    for fpath in files:
        total += migrate_file(fpath, llm_fn, workers=args.workers)

    logger.info(f"迁移完成，共生成 {total} 个 question")


if __name__ == "__main__":
    main()
