"""
存量 MCTS 轨迹迁移脚本

为 sft_trajectories.jsonl 补充规范的 question 字段，
通过 env_sample_idx 精确定位每条轨迹对应的 ScenarioState（含 knobs），
调用 generate_questions_for_state 生成基于具体 knob 偏差的新 question。

迁移逻辑：
    1. 用和 DBToolEnv 完全一样的加载逻辑，从 --scenarios 构建场景列表
    2. 按 env_sample_idx 分组轨迹
    3. 对每组（同一个 env_sample_idx / 同一个 ScenarioState）：
       - 取出对应的 ScenarioState（含 knobs）
       - 调 generate_questions_for_state(state, N, llm_fn) 批量生成 N 个 question
       - 分配给该组的 N 条轨迹
    4. 原子写回

用法:
    # 方式 1：多中转站轮询（推荐）
    python3 -m data_pipeline.synthesis.scenarios.migrate_add_questions \\
        --input data_pipeline/data/mcts/run_20260330_053333/sft_trajectories.jsonl \\
        --scenarios data_pipeline/data/scenarios/ \\
        --providers-config configs/providers.json \\
        --workers 10

    # 方式 2：单一 API
    python3 -m data_pipeline.synthesis.scenarios.migrate_add_questions \\
        --input data_pipeline/data/mcts/run_20260330_053333/sft_trajectories.jsonl \\
        --scenarios data_pipeline/data/scenarios/ \\
        --model gpt-5 \\
        --api-key sk-xxx \\
        --api-base https://xxx/v1 \\
        --workers 10
"""

import argparse
import json
import logging
import os
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_llm_fn(args):
    """构建 LLM 生成函数：优先使用多中转站轮询，fallback 到单 API"""
    from core.llm.multi_client import MultiProviderLLMClient

    client = MultiProviderLLMClient(
        target_model=args.model,
        providers_config=getattr(args, 'providers_config', None),
        single_api_key=getattr(args, 'api_key', None),
        single_api_base=getattr(args, 'api_base', None),
    )
    return lambda prompt: client.generate(prompt, temperature=0.9)


def load_scenarios_same_as_env(scenarios_path: str) -> list:
    """用和 DBToolEnv._load_scenarios 完全一样的逻辑加载场景列表，
    保证 env_sample_idx 对应关系正确。"""
    from environment.tools import DBToolEnv
    return DBToolEnv._load_scenarios(scenarios_path)


def main():
    parser = argparse.ArgumentParser(description="存量 MCTS 轨迹迁移：基于 knob 生成新 question")
    parser.add_argument("--input", required=True,
                        help="轨迹 JSONL 文件路径（sft_trajectories.jsonl）")
    parser.add_argument("--scenarios", required=True,
                        help="场景数据源（目录或 JSON 文件，和 MCTS 运行时一致）")
    parser.add_argument("--providers-config", default=None,
                        help="多中转站配置文件（providers.json）")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--api-key", default=None,
                        help="单一 API key（不传 --providers-config 时必须）")
    parser.add_argument("--api-base", default=None,
                        help="单一 API base（不传 --providers-config 时必须）")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    if not args.providers_config and not args.api_key:
        parser.error("必须提供 --providers-config 或 --api-key")

    # 1. 加载场景（和 DBToolEnv 完全一样的顺序）
    logger.info("加载场景...")
    scenarios = load_scenarios_same_as_env(args.scenarios)
    logger.info("场景列表长度: {} 条".format(len(scenarios)))

    # 2. 读取轨迹
    if not os.path.exists(args.input):
        logger.error("轨迹文件不存在: {}".format(args.input))
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        trajectories = [json.loads(line) for line in f if line.strip()]
    logger.info("共 {} 条轨迹".format(len(trajectories)))

    # 3. 按 env_sample_idx 分组，找出需要迁移的
    idx_to_traj_positions = defaultdict(list)  # env_sample_idx → [轨迹在 trajectories 中的位置]
    already_done = 0

    for pos, traj in enumerate(trajectories):
        if traj.get("question", ""):
            already_done += 1
            continue
        env_idx = traj.get("env_sample_idx")
        if env_idx is None:
            logger.warning("  轨迹 {} 缺少 env_sample_idx，跳过".format(pos))
            continue
        if env_idx >= len(scenarios):
            logger.warning("  轨迹 {} 的 env_sample_idx={} 超出场景范围 {}，跳过".format(
                pos, env_idx, len(scenarios)))
            continue
        idx_to_traj_positions[env_idx].append(pos)

    total_pending = sum(len(v) for v in idx_to_traj_positions.values())
    unique_envs = len(idx_to_traj_positions)
    logger.info("需要迁移: {} 条轨迹 / {} 个 unique env_sample_idx（已完成: {} 条）".format(
        total_pending, unique_envs, already_done))

    if not idx_to_traj_positions:
        logger.info("无需迁移，退出")
        return

    # 4. 构建 LLM 函数
    llm_fn = build_llm_fn(args)
    from data_pipeline.synthesis.scenarios.pipeline import generate_questions_for_state

    # 5. 并发生成：每个 env_sample_idx 调一次 LLM（批量生成 N 个 question）
    lock = threading.Lock()
    done_envs = [0]
    fail_envs = [0]
    done_trajs = [0]

    # env_sample_idx → list of new questions
    results = {}

    def generate_for_env(env_idx, count):
        """为一个 env_sample_idx 生成 count 个 question"""
        scenario = scenarios[env_idx]
        questions = generate_questions_for_state(scenario, count, llm_fn)
        return env_idx, questions

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(generate_for_env, env_idx, len(positions)): env_idx
            for env_idx, positions in idx_to_traj_positions.items()
        }
        for fut in as_completed(futures):
            env_idx = futures[fut]
            try:
                _, questions = fut.result()
                with lock:
                    results[env_idx] = questions
                    done_envs[0] += 1
                    if done_envs[0] % 50 == 0:
                        logger.info("  场景进度: {}/{}".format(
                            done_envs[0], unique_envs))
            except Exception as e:
                with lock:
                    fail_envs[0] += 1
                logger.warning("  [env_{}] 生成失败: {}".format(env_idx, e))

    # 6. 回填到轨迹
    for env_idx, positions in idx_to_traj_positions.items():
        questions = results.get(env_idx)
        if not questions:
            continue
        for i, pos in enumerate(positions):
            q = questions[i] if i < len(questions) else questions[-1]
            trajectories[pos]["question"] = q
            # 替换 messages 里第一条 user message
            for msg in trajectories[pos]["messages"]:
                if msg.get("role") == "user":
                    msg["content"] = q
                    break
            done_trajs[0] += 1

    # 7. 原子写回
    tmp_path = args.input + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for traj in trajectories:
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")
    os.replace(tmp_path, args.input)

    logger.info(
        "迁移完成: {} 个场景成功 / {} 个失败，共回填 {} 条轨迹".format(
            done_envs[0], fail_envs[0], done_trajs[0])
    )


if __name__ == "__main__":
    main()
