"""
存量 MCTS 轨迹迁移脚本

为 run_20260330_053333/sft_trajectories.jsonl（旧 MCTS 轨迹）补充规范的 question 字段，
并将 messages[1]["content"] 里的旧技术性 question 替换为无信息泄露的新 question。

迁移逻辑：
    1. 构建反查表：old_question → scenario_name（从 questions_cache.json 反向建索引）
    2. 从 collected_*.json 建立 scenario_name → 场景 dict（source=llm_generated）
    3. 对每条轨迹：
       - 取 messages[1]["content"] = 旧 question
       - 反查 → scenario_name → 场景 dict
       - 调 generate_questions_for_state(state, 1, llm_fn)[0] 生成新 question
       - 写入顶层 "question" 字段 + 替换 messages[1]["content"]
    4. 原子写回（tmp → replace），支持断点续跑

用法:
    python3 -m datasets.synthesis.scenarios.migrate_add_questions \\
        --input datasets/data/mcts/run_20260330_053333/sft_trajectories.jsonl \\
        --cache datasets/data/mcts/questions_cache.json \\
        --scenarios datasets/data/scenarios/ \\
        --model gpt-5 \\
        --api-key sk-xxx \\
        --api-base https://xxx/v1 \\
        --workers 20
"""

import argparse
import glob
import json
import logging
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_llm_fn(args):
    from openai import OpenAI
    client = OpenAI(api_key=args.api_key, base_url=args.api_base, timeout=120)

    def generate(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=1024,
        )
        return resp.choices[0].message.content

    return generate


def load_questions_cache(cache_path: str) -> dict:
    """加载 questions_cache.json，返回 scenario_name → old_question"""
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_scenarios(scenarios_dir: str) -> dict:
    """从 collected_*.json 加载 scenario_name → 场景 dict（source=llm_generated）"""
    name_to_state = {}
    pattern = os.path.join(scenarios_dir, "collected_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        logger.warning(f"未找到 collected_*.json 文件: {pattern}")
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            items = json.load(f)
        for item in items:
            if item.get("source", "llm_generated") != "llm_generated":
                continue
            name = item.get("name", "")
            if name:
                name_to_state[name] = item
    logger.info(f"已加载 {len(name_to_state)} 个 llm_generated 场景")
    return name_to_state


def get_first_user_message(messages: list) -> str:
    """提取 messages 中第一条 role=user 的内容（即旧 question）"""
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def main():
    parser = argparse.ArgumentParser(description="存量 MCTS 轨迹迁移：补充规范 question 字段")
    parser.add_argument("--input", required=True,
                        help="轨迹 JSONL 文件路径（sft_trajectories.jsonl）")
    parser.add_argument("--cache", required=True,
                        help="questions_cache.json 路径")
    parser.add_argument("--scenarios", required=True,
                        help="场景数据目录（包含 collected_*.json）")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    # 1. 加载 cache，构建 old_question → scenario_name 反查表
    cache = load_questions_cache(args.cache)
    reverse_cache = {v: k for k, v in cache.items()}
    logger.info(f"questions_cache: {len(cache)} 条，反查表: {len(reverse_cache)} 条")

    # 2. 加载场景
    name_to_state = load_scenarios(args.scenarios)

    # 3. 读取轨迹文件，找出需要迁移的条目
    if not os.path.exists(args.input):
        logger.error(f"轨迹文件不存在: {args.input}")
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        trajectories = [json.loads(line) for line in f if line.strip()]

    logger.info(f"共 {len(trajectories)} 条轨迹")

    pending = []  # (index, trajectory)
    for i, traj in enumerate(trajectories):
        if traj.get("question", ""):
            continue  # 已完成，断点续跑跳过
        pending.append((i, traj))

    logger.info(f"需要迁移: {len(pending)} 条（已完成: {len(trajectories) - len(pending)} 条）")
    if not pending:
        logger.info("所有轨迹已完成迁移，退出")
        return

    # 4. 构建 LLM 函数
    llm_fn = build_llm_fn(args)
    from datasets.synthesis.scenarios.pipeline import generate_questions_for_state

    # 5. 统计
    done_count = [0]
    skip_count = [0]
    fail_count = [0]
    lock_obj = __import__("threading").Lock()

    def migrate_one(item):
        idx, traj = item
        old_q = get_first_user_message(traj.get("messages", []))
        scenario_name = reverse_cache.get(old_q, "")
        if not scenario_name:
            # 旧 question 不在 cache 里（可能是 cache 不完整），尝试 env_sample_idx 路径
            with lock_obj:
                skip_count[0] += 1
            logger.warning(f"  [{idx}] 未找到对应场景（old_q 不在 cache），跳过")
            return idx, None

        state = name_to_state.get(scenario_name)
        if not state:
            with lock_obj:
                skip_count[0] += 1
            logger.warning(f"  [{idx}] 场景 {scenario_name} 在 collected 文件中不存在，跳过")
            return idx, None

        new_q = generate_questions_for_state(state, 1, llm_fn)[0]
        return idx, new_q

    # 6. 并发生成
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(migrate_one, item): item for item in pending}
        for fut in as_completed(futures):
            try:
                idx, new_q = fut.result()
                if new_q:
                    trajectories[idx]["question"] = new_q
                    # 同时替换 messages[1].content（第一条 user message）
                    msgs = trajectories[idx]["messages"]
                    for msg in msgs:
                        if msg.get("role") == "user":
                            msg["content"] = new_q
                            break
                    with lock_obj:
                        done_count[0] += 1
                        if done_count[0] % 100 == 0:
                            logger.info(f"  进度: {done_count[0] + skip_count[0]}/{len(pending)}")
            except Exception as e:
                with lock_obj:
                    fail_count[0] += 1
                logger.warning(f"  生成失败: {e}")

    # 7. 原子写回
    tmp_path = args.input + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for traj in trajectories:
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")
    os.replace(tmp_path, args.input)

    logger.info(
        f"迁移完成: 成功 {done_count[0]} 条 / 跳过 {skip_count[0]} 条 / 失败 {fail_count[0]} 条"
    )


if __name__ == "__main__":
    main()
