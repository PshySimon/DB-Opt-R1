"""
MCTS 轨迹合成主流程
"""

import json
import argparse
import logging
import os
import sys
import time

# 添加项目根目录到 path（datasets/synthesis/mcts/ → 回到项目根）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from datasets.synthesis.mcts.search import MCTSSearch
from datasets.synthesis.mcts.async_search import AsyncMCTSSearch
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
                max_tokens=2048,
            )
            return response.choices[0].message.content

        return generate
    except ImportError:
        logger.error("需要安装 openai: pip install openai")
        sys.exit(1)


def run_mcts(args):
    """主流程"""
    from environment.tools import DBToolEnv
    t0 = time.time()

    data_source = args.scenarios if args.scenarios else args.dataset
    logger.info(f"数据源: {data_source}")
    logger.info(f"模型: {args.model}")
    logger.info(f"搜索参数: simulations={args.simulations}, children={args.children}, depth={args.depth}, num_workers={getattr(args, 'num_workers', 1)}")

    # LLM 客户端
    llm_generate = create_llm_client(
        model=args.model,
        api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
        api_base=args.api_base or os.environ.get("OPENAI_API_BASE"),
    )

    # 搜索配置
    num_workers = getattr(args, 'num_workers', 1)

    search_config = {
        "num_simulations": args.simulations,
        "max_children": args.children,
        "max_depth": args.depth,
        "ucb_c": args.ucb_c,
        "expand_temperature": args.expand_temp,
        "rollout_temperature": args.rollout_temp,
        "system_prompt": SYSTEM_PROMPT,
        "num_workers": num_workers,
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
    if args.scenarios:
        # 新模式：场景目录
        scenario_files = sorted([
            f for f in os.listdir(args.scenarios)
            if f.endswith(".json")
        ])
        num_samples = min(args.num_envs, len(scenario_files))
    else:
        # 旧模式：CSV
        import pandas as pd
        dataset = pd.read_csv(args.dataset, on_bad_lines="skip")
        num_samples = min(args.num_envs, len(dataset))

    def search_one_env(i):
        """搜索单个环境，返回 (sft_items, contrastive_items)"""
        logger.info(f"\n{'='*50}")
        logger.info(f"环境 {i+1}/{num_samples} (sample_idx={i})")
        logger.info(f"{'='*50}")

        def env_factory():
            return DBToolEnv(
                mode="train",
                dataset_path=args.dataset if not args.scenarios else None,
                scenario_dir=args.scenarios,
                cost_model=cost_model,
                max_turns=args.depth,
                knob_space_path=args.knob_space,
            )

        if num_workers > 1:
            searcher = AsyncMCTSSearch(
                env_factory=env_factory,
                llm_generate=llm_generate,
                config=search_config,
            )
        else:
            env = env_factory()
            searcher = MCTSSearch(env=env, llm_generate=llm_generate, config=search_config)

        root = searcher.search(sample_idx=i)

        # 保存搜索树用于 debug
        tree_dir = os.path.join(args.output_dir, "mcts_trees")
        os.makedirs(tree_dir, exist_ok=True)
        tree_path = os.path.join(tree_dir, f"tree_env_{i}.json")
        with open(tree_path, "w", encoding="utf-8") as f:
            json.dump(root.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"  搜索树已保存: {tree_path}")

        sft_items, contrastive_items = [], []
        if root.children:
            best_traj = extract_best_trajectory(root)
            best_reward = root.best_child_by_reward().avg_reward

            sft_items.append(format_trajectory_as_messages(
                trajectory=best_traj,
                system_prompt=SYSTEM_PROMPT,
                reward=best_reward,
                sample_idx=i,
            ))
            logger.info(f"  最优轨迹: {len(best_traj)} 步, reward={best_reward:.3f}")

            top_k = extract_top_k_trajectories(root, k=3)
            for traj in top_k[1:]:
                sft_items.append(format_trajectory_as_messages(
                    trajectory=traj,
                    system_prompt=SYSTEM_PROMPT,
                    sample_idx=i,
                ))

            pairs = extract_contrastive_pairs(root)
            for pair in pairs:
                contrastive_items.append(format_contrastive_as_dpo(pair, SYSTEM_PROMPT))
            logger.info(f"  对比数据: {len(pairs)} 对")
        else:
            logger.warning(f"  搜索未展开任何节点，跳过")

        return sft_items, contrastive_items

    # 多环境并行 or 串行
    parallel = getattr(args, 'parallel', 1)
    if parallel > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        logger.info(f"多环境并行: {parallel} parallel")
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {pool.submit(search_one_env, i): i for i in range(num_samples)}
            for future in as_completed(futures):
                try:
                    sft_items, contrastive_items = future.result()
                    sft_data.extend(sft_items)
                    contrastive_data.extend(contrastive_items)
                except Exception as e:
                    logger.error(f"环境 {futures[future]} 搜索失败: {e}")
    else:
        for i in range(num_samples):
            sft_items, contrastive_items = search_one_env(i)
            sft_data.extend(sft_items)
            contrastive_data.extend(contrastive_items)

    # 保存
    os.makedirs(args.output_dir, exist_ok=True)

    sft_path = os.path.join(args.output_dir, "sft_trajectories.jsonl")
    save_jsonl(sft_data, sft_path)
    logger.info(f"\nSFT 数据: {len(sft_data)} 条 → {sft_path}")

    contrastive_path = os.path.join(args.output_dir, "contrastive_pairs.jsonl")
    save_jsonl(contrastive_data, contrastive_path)
    logger.info(f"对比数据: {len(contrastive_data)} 对 → {contrastive_path}")

    elapsed = time.time() - t0
    minutes, seconds = divmod(int(elapsed), 60)
    logger.info(f"\n总耗时: {minutes}m {seconds}s")

    # 预览命令提示
    logger.info(f"\n预览数据:")
    logger.info(f"  python3 -m datasets.synthesis.mcts.preview {sft_path}")
    logger.info(f"  python3 -m datasets.synthesis.mcts.preview {contrastive_path}")
    tree_dir = os.path.join(args.output_dir, "mcts_trees")
    if os.path.isdir(tree_dir):
        trees = [f for f in os.listdir(tree_dir) if f.endswith('.json')]
        if trees:
            logger.info(f"  python3 -m datasets.synthesis.mcts.preview {os.path.join(tree_dir, trees[0])}")


def main():
    parser = argparse.ArgumentParser(description="MCTS 轨迹合成")

    # 数据
    parser.add_argument("--dataset", default=None, help="CSV 数据集路径（旧模式）")
    parser.add_argument("--scenarios", default=None, help="YAML 场景目录（新模式，优先于 --dataset）")
    parser.add_argument("--cost-model", default=None, help="Cost Model 路径")
    parser.add_argument("--knob-space", default="configs/knob_space.yaml", help="knob_space.yaml 路径")
    parser.add_argument("--output-dir", default="datasets/data", help="输出目录（datasets/data/）")
    parser.add_argument("--num-envs", type=int, default=100, help="搜索的环境数")
    parser.add_argument("--parallel", type=int, default=1, help="多环境并行数（1=串行）")
    parser.add_argument("--num-workers", type=int, default=1, help="单棵树内并发 simulation 线程数（1=串行）")

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
