"""
MCTS 轨迹合成主流程
"""

import json
import argparse
import logging
import os
import sys

# 添加项目根目录到 path（datasets/synthesis/mcts/ → 回到项目根）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from datasets.synthesis.mcts.search import MCTSSearch
from datasets.synthesis.mcts.extract import (
    extract_best_trajectory,
    extract_top_k_trajectories,
    extract_contrastive_pairs,
    format_trajectory_as_messages,
    format_contrastive_as_dpo,
    save_jsonl,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)


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


EXAMPLE_TRAJECTORY = (
    "以下是一个调优示例：\n\n"
    "User: 请优化这个数据库的性能。\n\n"
    "Assistant: <think>\n"
    "首先需要了解硬件环境，才能确定合理的参数配置范围。\n"
    "</think>\n"
    '<tool_call>\n{"name": "get_hardware_info", "arguments": {}}\n</tool_call>\n\n'
    'Observation: {"cpu_count": 8, "total_memory_gb": 32, "disk_type": "SSD"}\n\n'
    "Assistant: <think>\n"
    "8核32G内存，SSD磁盘。先看看当前关键配置是否合理。\n"
    "shared_buffers 建议设为总内存的 25%，即 8GB。\n"
    "</think>\n"
    '<tool_call>\n{"name": "get_current_config", "arguments": {"knob_names": "shared_buffers,work_mem,effective_cache_size,random_page_cost"}}\n</tool_call>\n\n'
    'Observation: {"shared_buffers": "128MB", "work_mem": "4MB", "effective_cache_size": "4GB", "random_page_cost": "4"}\n\n'
    "Assistant: <think>\n"
    "发现几个明显的问题：\n"
    "1. shared_buffers=128MB，远低于建议的 8GB（32G 的 25%）\n"
    "2. work_mem=4MB，太小，会产生大量临时文件\n"
    "3. effective_cache_size=4GB，应设为 16-24GB\n"
    "4. random_page_cost=4，这是 HDD 的默认值，SSD 应设为 1.1\n"
    "先调整这些核心参数。\n"
    "</think>\n"
    '<tool_call>\n{"name": "set_knob", "arguments": {"knobs": "{\\"shared_buffers\\": \\"8GB\\", \\"work_mem\\": \\"256MB\\", \\"effective_cache_size\\": \\"24GB\\", \\"random_page_cost\\": \\"1.1\\"}"}}\n</tool_call>\n\n'
    'Observation: {"success": ["work_mem", "effective_cache_size", "random_page_cost"], "pending_restart": ["shared_buffers"], "failed": []}\n\n'
    "Assistant: <think>\n"
    "shared_buffers 需要重启才能生效，调用 restart_pg。\n"
    "</think>\n"
    '<tool_call>\n{"name": "restart_pg", "arguments": {}}\n</tool_call>\n\n'
    'Observation: {"success": true, "duration_seconds": 2.1}\n\n'
    "Assistant: <think>\n"
    "参数已生效，现在验证性能变化。\n"
    "</think>\n"
    '<tool_call>\n{"name": "predict_performance", "arguments": {}}\n</tool_call>\n\n'
    'Observation: {"predicted_tps": 2875, "baseline_tps": 2500, "improvement_pct": 15.0}\n'
)




def create_llm_client(model: str, api_key: str = None, api_base: str = None):
    """创建 LLM 调用函数"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=api_base)

        def generate(prompt: str, temperature: float = 0.7) -> str:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=512,
            )
            return response.choices[0].message.content

        return generate
    except ImportError:
        logger.error("需要安装 openai: pip install openai")
        sys.exit(1)


def run_mcts(args):
    """主流程"""
    from environment.tools import DBToolEnv

    logger.info(f"加载数据集: {args.dataset}")
    logger.info(f"模型: {args.model}")
    logger.info(f"搜索参数: simulations={args.simulations}, children={args.children}, depth={args.depth}")

    # LLM 客户端
    llm_generate = create_llm_client(
        model=args.model,
        api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
        api_base=args.api_base or os.environ.get("OPENAI_API_BASE"),
    )

    # 搜索配置
    search_config = {
        "num_simulations": args.simulations,
        "max_children": args.children,
        "max_depth": args.depth,
        "ucb_c": args.ucb_c,
        "expand_temperature": args.expand_temp,
        "rollout_temperature": args.rollout_temp,
        "system_prompt": SYSTEM_PROMPT,
    }

    # 加载 cost model
    cost_model = None
    if args.cost_model:
        logger.info(f"加载 Cost Model: {args.cost_model}")
        from cost_model.model import CostModel
        cost_model = CostModel.load(args.cost_model)

    sft_data = []
    contrastive_data = []

    # 对每个环境样本搜索
    import pandas as pd
    dataset = pd.read_csv(args.dataset, on_bad_lines="skip")
    num_samples = min(args.num_envs, len(dataset))

    for i in range(num_samples):
        logger.info(f"\n{'='*50}")
        logger.info(f"环境 {i+1}/{num_samples} (sample_idx={i})")
        logger.info(f"{'='*50}")

        # 创建环境
        env = DBToolEnv(
            mode="train",
            dataset_path=args.dataset,
            cost_model=cost_model,
            max_turns=args.depth,
        )

        # MCTS 搜索
        searcher = MCTSSearch(env=env, llm_generate=llm_generate, config=search_config)
        searcher._example_trajectory = EXAMPLE_TRAJECTORY
        root = searcher.search(sample_idx=i)

        # 提取轨迹
        if root.children:
            # Best-in-tree
            best_traj = extract_best_trajectory(root)
            best_reward = root.best_child_by_reward().avg_reward

            sft_item = format_trajectory_as_messages(
                trajectory=best_traj,
                system_prompt=SYSTEM_PROMPT,
                reward=best_reward,
                sample_idx=i,
            )
            sft_data.append(sft_item)
            logger.info(f"  最优轨迹: {len(best_traj)} 步, reward={best_reward:.3f}")

            # Top-k 轨迹
            top_k = extract_top_k_trajectories(root, k=3)
            for k, traj in enumerate(top_k[1:], 2):  # 跳过第一条（和 best 重复）
                sft_item = format_trajectory_as_messages(
                    trajectory=traj,
                    system_prompt=SYSTEM_PROMPT,
                    sample_idx=i,
                )
                sft_data.append(sft_item)

            # 对比数据
            pairs = extract_contrastive_pairs(root)
            for pair in pairs:
                dpo_item = format_contrastive_as_dpo(pair, SYSTEM_PROMPT)
                contrastive_data.append(dpo_item)
            logger.info(f"  对比数据: {len(pairs)} 对")
        else:
            logger.warning(f"  搜索未展开任何节点，跳过")

    # 保存
    os.makedirs(args.output_dir, exist_ok=True)

    sft_path = os.path.join(args.output_dir, "sft_trajectories.jsonl")
    save_jsonl(sft_data, sft_path)
    logger.info(f"\nSFT 数据: {len(sft_data)} 条 → {sft_path}")

    contrastive_path = os.path.join(args.output_dir, "contrastive_pairs.jsonl")
    save_jsonl(contrastive_data, contrastive_path)
    logger.info(f"对比数据: {len(contrastive_data)} 对 → {contrastive_path}")


def main():
    parser = argparse.ArgumentParser(description="MCTS 轨迹合成")

    # 数据
    parser.add_argument("--dataset", required=True, help="CSV 数据集路径")
    parser.add_argument("--cost-model", default=None, help="Cost Model 路径")
    parser.add_argument("--output-dir", default="datasets/data", help="输出目录（datasets/data/）")
    parser.add_argument("--num-envs", type=int, default=100, help="搜索的环境数")

    # LLM
    parser.add_argument("--model", default="gpt-4", help="LLM 模型名称")
    parser.add_argument("--api-key", default=None, help="API Key")
    parser.add_argument("--api-base", default=None, help="API Base URL")

    # MCTS
    parser.add_argument("--simulations", type=int, default=50, help="每棵树 MCTS 迭代次数")
    parser.add_argument("--children", type=int, default=3, help="最大子节点数")
    parser.add_argument("--depth", type=int, default=10, help="最大深度")
    parser.add_argument("--ucb-c", type=float, default=1.414, help="UCB1 探索系数")
    parser.add_argument("--expand-temp", type=float, default=0.8, help="展开温度")
    parser.add_argument("--rollout-temp", type=float, default=0.3, help="Rollout 温度")

    args = parser.parse_args()
    run_mcts(args)


if __name__ == "__main__":
    main()
