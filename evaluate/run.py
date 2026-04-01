"""
评估脚本入口

用法:
    python -m evaluate.run \
        --scenarios datasets/data/scenarios/ \
        --knob-space configs/knob_space.yaml \
        --cost-model cost_model/checkpoints/v3 \
        --api-key sk-xxx \
        --api-base https://xxx/v1 \
        --model gpt-5 \
        --output eval_results/baseline_gpt5/ \
        --max-turns 10 \
        --parallel 8
"""

import os
import sys
import json
import time
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from evaluate.agent import run_episode
from evaluate.report import compute_metrics, save_report, print_summary

logger = logging.getLogger(__name__)


# 复用 MCTS 的 system prompt
SYSTEM_PROMPT = """你是 PostgreSQL 数据库调优专家。你的目标是通过调整数据库配置参数来最大化性能（TPS）。

## 输出格式

每次回复必须严格遵循以下格式：先用 <think>...</think> 分析推理，再用 <tool_call>...</tool_call> 调用一个工具。

重要规则：
1. 每次只能调用一个工具
2. 必须先 <think> 分析，再 <tool_call> 调用
3. <think> 中要解释你观察到了什么、为什么这样做

## 工作流程

1. 先观察硬件环境（get_hardware_info）
2. 查看关键配置（get_current_config）和运行指标（get_db_metrics）
3. 分析瓶颈，用 set_knob 设置合理的参数
4. 如果修改了 shared_buffers 等 static 参数，调用 restart_pg
5. 用 predict_performance 验证效果

## 调优知识

- shared_buffers：总内存的 25%
- effective_cache_size：总内存的 50%-75%
- work_mem：根据并发连接数分配，一般 64MB-256MB
- random_page_cost：SSD 设 1.1，HDD 设 4.0
- 修改 shared_buffers 等 postmaster 参数后需要 restart_pg"""


def build_llm_fn(args):
    """构建 LLM 生成函数"""
    from openai import OpenAI

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
        timeout=300,  # 5 分钟超时，兼容 vLLM 冷启动
    )

    def generate(messages: list, temperature: float = 0.3) -> str:
        resp = client.chat.completions.create(
            model=args.model,
            messages=messages,
            temperature=temperature,
            max_tokens=2048,
        )
        return resp.choices[0].message.content

    return generate


def load_questions(questions_path: str) -> dict:
    """加载 questions_cache.json"""
    if questions_path and os.path.exists(questions_path):
        with open(questions_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def evaluate_one(sample_idx, env_scenarios, llm_fn, cost_model, questions, args):
    """评估单个场景"""
    from environment.tools import DBToolEnv

    # 每个 worker 独立创建 env，避免竞争
    env = DBToolEnv(
        mode="train",
        cost_model=cost_model,
        max_turns=args.max_turns,
        knob_space_path=args.knob_space,
    )
    env.scenarios = env_scenarios

    env.reset(sample_idx=sample_idx)

    # 获取 user message
    scenario = env_scenarios[sample_idx]
    s_name = getattr(scenario, "name", "")
    _q = questions.get(s_name, f"请优化这个数据库的性能。场景: {s_name}")
    # 兼容新格式（list）和旧格式（str），评估固定取第一个保证可复现
    user_message = _q[0] if isinstance(_q, list) else _q

    result = run_episode(
        env=env,
        llm_fn=llm_fn,
        system_prompt=SYSTEM_PROMPT,
        user_message=user_message,
        max_turns=args.max_turns,
        sample_idx=sample_idx,
        save_trajectory=args.save_trajectories,
    )

    logger.info(
        f"  [env_{sample_idx}] {s_name}: "
        f"steps={result['steps']}, reward={result['total_reward']:.3f}, "
        f"predict={'✓' if result['called_predict'] else '✗'}"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="模型评估")

    # 数据
    parser.add_argument("--scenarios", required=True,
                        help="场景数据源（目录或 JSON 文件）")
    parser.add_argument("--eval-scenarios", default=None,
                        help="评估专用场景文件（如 knob_configs_eval.json），传入则仅评估该文件中的场景")
    parser.add_argument("--questions", default="datasets/data/mcts/questions_cache.json",
                        help="questions_cache.json 路径")
    parser.add_argument("--knob-space", default="configs/knob_space.yaml")
    parser.add_argument("--cost-model", required=True, help="Cost Model checkpoint 目录")

    # LLM
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", default="gpt-5")

    # 评估参数
    parser.add_argument("--output", default="eval_results/", help="输出目录")
    parser.add_argument("--num-envs", type=int, default=0, help="评估场景数（0=全量）")
    parser.add_argument("--max-turns", type=int, default=10, help="每个 episode 最大交互轮数")
    parser.add_argument("--parallel", type=int, default=4, help="并发 worker 数")
    parser.add_argument("--save-trajectories", action="store_true", help="保存完整轨迹")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # 1. 加载 cost model
    logger.info("加载 Cost Model...")
    from cost_model.model import CostModel
    cost_model = CostModel.load(args.cost_model)

    # 2. 加载场景
    logger.info("加载场景...")
    from environment.tools import DBToolEnv
    temp_env = DBToolEnv(
        mode="train",
        scenario_dir=args.eval_scenarios or args.scenarios,
        cost_model=cost_model,
        max_turns=args.max_turns,
        knob_space_path=args.knob_space,
    )
    scenarios = temp_env.scenarios
    logger.info(f"  可用场景: {len(scenarios)}")

    if len(scenarios) == 0:
        logger.error("没有可用的评估场景，退出")
        sys.exit(1)

    # 3. 确定评估范围
    num_envs = args.num_envs if args.num_envs > 0 else len(scenarios)
    num_envs = min(num_envs, len(scenarios))
    eval_indices = list(range(num_envs))

    # 4. 加载 questions
    questions = load_questions(args.questions)
    logger.info(f"  问题缓存: {len(questions)} 条")

    # 5. 构建 LLM 函数
    logger.info(f"LLM: {args.model} @ {args.api_base}")
    llm_fn = build_llm_fn(args)

    # 6. 并发评估
    logger.info(f"开始评估: {num_envs} 个场景, 并发={args.parallel}")
    results = []
    os.makedirs(args.output, exist_ok=True)
    episodes_path = os.path.join(args.output, "episodes.jsonl")

    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(
                evaluate_one, idx, scenarios, llm_fn, cost_model, questions, args
            ): idx
            for idx in eval_indices
        }

        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                result = fut.result()
                results.append(result)

                # 增量写入
                with open(episodes_path, "a", encoding="utf-8") as f:
                    # 不保存 trajectory 到增量文件（太大）
                    slim = {k: v for k, v in result.items() if k != "trajectory"}
                    f.write(json.dumps(slim, ensure_ascii=False) + "\n")

            except Exception as e:
                logger.error(f"  [env_{idx}] 评估失败: {e}")

    elapsed = time.time() - t0
    logger.info(f"评估完成: {len(results)}/{num_envs} 个场景, 耗时 {elapsed:.0f}s")

    # 7. 生成报告
    meta = {
        "model": args.model,
        "api_base": args.api_base,
        "scenarios": args.eval_scenarios or args.scenarios,
        "num_episodes": len(results),
        "max_turns": args.max_turns,
        "elapsed_seconds": round(elapsed, 1),
    }
    report = compute_metrics(results, meta)
    save_report(report, args.output)
    print_summary(report)


if __name__ == "__main__":
    main()
