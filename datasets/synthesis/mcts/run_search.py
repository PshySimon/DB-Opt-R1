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

import re as _re


def _build_question_prompt(scenario) -> str:
    """根据场景信息生成 LLM 提示，让 LLM 以用户视角提出自然的调优问题。"""
    hw = getattr(scenario, 'hardware', {}) or {}
    wl = getattr(scenario, 'workload', {}) or {}
    desc = getattr(scenario, 'description', '') or ''
    return (
        "你是一位真实的 PostgreSQL 使用者，正在向技术支持提问。"
        "根据以下场景信息，生成一个自然的中文用户提问。\n\n"
        "规则：\n"
        "1. 从用户视角出发，用户不知道根因，只知道自己遇到了问题\n"
        "2. 提问风格可以多样：可以描述观察到的现象（'某个操作变慢了'、'磁盘 IO 异常'），"
        "也可以描述遇到的困扰（'报表跑不动'、'某个作业一直没跑完'）\n"
        "3. 不要在提问中提及任何具体指标数字\n"
        "4. 语言自然、口语化，像真实用户在技术论坛或工单里提问\n"
        f"5. 输出严格的 JSON，格式为：{{\"question\": \"...\"}}\n\n"
        f"场景背景（仅供参考，不要照搬）：\n{desc}\n"
        f"硬件环境：{hw.get('total_memory_gb')}GB 内存，{hw.get('disk_type')} 磁盘\n"
        f"负载类型：{wl.get('type')}\n\n"
        "JSON:"
    )


def _extract_question(raw: str) -> str:
    """从模型返回的 JSON 中提取 question 字段，失败则返回默认值。"""
    try:
        m = _re.search(r'\{.*?"question"\s*:\s*"(.+?)"\s*\}', raw, _re.DOTALL)
        if m:
            return m.group(1).strip()
        return json.loads(raw)["question"]
    except Exception:
        return "请优化这个数据库的性能。"


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
        client = OpenAI(
            api_key=api_key, 
            base_url=api_base,
            default_headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"}
        )

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

    # 断点续跑：复用已有 run 目录；否则创建新目录
    if getattr(args, 'resume_dir', None):
        run_dir = args.resume_dir
        if not os.path.isdir(run_dir):
            logger.error(f"--resume-dir 目录不存在: {run_dir}")
            sys.exit(1)
        logger.info(f"断点续跑，复用目录: {run_dir}")
    else:
        run_id = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(args.output_dir, f"run_{run_id}")
        os.makedirs(run_dir, exist_ok=True)
        logger.info(f"输出目录: {run_dir}")

    data_source = args.scenarios if args.scenarios else args.dataset
    logger.info(f"数据源: {data_source}")
    logger.info(f"模型: {args.model}")
    logger.info(f"搜索参数: simulations={args.simulations}, children={args.children}, depth={args.depth}, num_workers={getattr(args, 'num_workers', 1)}")

    # LLM 客户端（支持多中转站轮询与自动回退）
    from core.llm.multi_client import MultiProviderLLMClient
    llm_client = MultiProviderLLMClient(
        target_model=args.model,
        providers_config=args.providers_config,
        single_api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
        single_api_base=args.api_base or os.environ.get("OPENAI_API_BASE"),
    )
    llm_generate = llm_client.generate

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

    # 断点续跑：恢复已有结果
    sft_path = os.path.join(run_dir, "sft_trajectories.jsonl")
    contrastive_path = os.path.join(run_dir, "contrastive_pairs.jsonl")
    sft_data, contrastive_data = [], []
    if getattr(args, 'resume_dir', None):
        for path, store in [(sft_path, sft_data), (contrastive_path, contrastive_data)]:
            if os.path.isfile(path):
                with open(path, encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            store.append(json.loads(line))
                logger.info(f"  恢复已有数据: {len(store)} 条 ← {path}")

    # 对每个环境样本搜索：预加载场景（只读一次磁盘）
    preloaded_scenarios = None
    if args.scenarios:
        env_tmp = DBToolEnv(
            mode="train", scenario_dir=args.scenarios,
            max_turns=args.depth, knob_space_path=args.knob_space,
        )
        total = env_tmp.num_samples
        preloaded_scenarios = env_tmp.scenarios  # 保留预加载的场景列表
        del env_tmp
    else:
        # 旧模式：CSV
        import pandas as pd
        dataset = pd.read_csv(args.dataset, on_bad_lines="skip")
        total = len(dataset)

    num_samples = min(args.num_envs, total) if args.num_envs > 0 else total
    logger.info(f"场景总数: {total}，本次搜索: {num_samples}")

    # 已完成的环境：扫描 tree 文件并做完整性校验
    # 完整 = 文件 >1KB + 合法 JSON + 所有非根节点 visit_count > 0
    tree_dir = os.path.join(run_dir, "mcts_trees")
    done_envs = set()

    def _is_tree_complete(path: str) -> bool:
        """返回 True 表示该树完整可用，False 表示需要重新搜索。"""
        try:
            d = json.load(open(path, encoding="utf-8"))
        except Exception:
            return False
        # 空树：根节点没有 children
        if not d.get("children"):
            return False
        # 递归检查：非根节点 visit_count 必须 > 0
        def _check(node, depth):
            if depth > 0 and (node.get("visit_count") or 0) == 0:
                return False
            for child in node.get("children") or []:
                if not _check(child, depth + 1):
                    return False
            return True
        return _check(d, 0)

    if os.path.isdir(tree_dir):
        for fname in os.listdir(tree_dir):
            if not (fname.startswith("tree_env_") and fname.endswith(".json")):
                continue
            try:
                env_idx = int(fname[len("tree_env_"):-len(".json")])
            except ValueError:
                continue
            path = os.path.join(tree_dir, fname)
            if _is_tree_complete(path):
                done_envs.add(env_idx)
            else:
                # 不完整：删除后重新搜索，避免漏数据
                os.remove(path)
                logger.warning(f"  树不完整，已删除将重新搜索: {fname}")
    if done_envs:
        logger.info(f"已完成环境: {len(done_envs)} 个，跳过")

    # 加载已生成的 questions 缓存 (跨 run_dir 复用)
    questions_cache_path = os.path.join(args.output_dir, "questions_cache.json")
    questions = {}
    if os.path.exists(questions_cache_path):
        try:
            with open(questions_cache_path, "r", encoding="utf-8") as f:
                questions = json.load(f)
            logger.info(f"已加载 {len(questions)} 个预生成的场景问题缓存")
        except Exception as e:
            logger.warning(f"读取 questions.json 缓存失败: {e}")
            questions = {}

    import threading
    q_lock = threading.Lock()

    def _save_questions_cache():
        with q_lock:
            with open(questions_cache_path, "w", encoding="utf-8") as f:
                json.dump(questions, f, ensure_ascii=False, indent=2)
    def search_one_env(i):
        """搜索单个环境，返回 (sft_items, contrastive_items)"""
        logger.info(f"\n{'='*50}")
        logger.info(f"环境 {i+1}/{num_samples} (sample_idx={i})")
        logger.info(f"{'='*50}")

        # 取场景专属提问（从缓存），没有即跳过
        scenario_name = preloaded_scenarios[i].name if preloaded_scenarios else f"env_{i}"
        q = questions.get(scenario_name)
        if not q:
            logger.error(f"  场景 {scenario_name} 无可用专属问题，跳过 MCTS 搜索")
            return [], []

        # 每个环境独立 copy，避免并行竞争
        scenario_config = {**search_config, "user_message": q}

        def env_factory():
            env = DBToolEnv(
                mode="train",
                dataset_path=args.dataset if not args.scenarios else None,
                cost_model=cost_model,
                max_turns=args.depth,
                knob_space_path=args.knob_space,
            )
            # 复用预加载的场景，不再重新读磁盘
            if preloaded_scenarios is not None:
                env.scenarios = preloaded_scenarios
            return env

        if num_workers > 1:
            searcher = AsyncMCTSSearch(
                env_factory=env_factory,
                llm_generate=llm_generate,
                config=scenario_config,
            )
        else:
            env = env_factory()
            searcher = MCTSSearch(env=env, llm_generate=llm_generate, config=scenario_config)

        root = searcher.search(sample_idx=i)

        sft_items, contrastive_items = [], []
        if root.children:
            # 保存搜索树用于 debug（空树不保存，避免干扰断点续跑）
            os.makedirs(tree_dir, exist_ok=True)
            tree_path = os.path.join(tree_dir, f"tree_env_{i}.json")
            with open(tree_path, "w", encoding="utf-8") as f:
                json.dump(root.to_dict(), f, ensure_ascii=False, indent=2)
            logger.info(f"  搜索树已保存: {tree_path}")

            best_traj = extract_best_trajectory(root)
            best_reward = root.best_child_by_reward().avg_reward

            sft_items.append(format_trajectory_as_messages(
                trajectory=best_traj,
                system_prompt=SYSTEM_PROMPT,
                reward=best_reward,
                sample_idx=i,
                user_message=q,
            ))
            logger.info(f"  最优轨迹: {len(best_traj)} 步, reward={best_reward:.3f}")

            top_k = extract_top_k_trajectories(root, k=3)
            for traj in top_k[1:]:
                sft_items.append(format_trajectory_as_messages(
                    trajectory=traj,
                    system_prompt=SYSTEM_PROMPT,
                    sample_idx=i,
                    user_message=q,
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
    pending = [i for i in range(num_samples) if i not in done_envs]
    logger.info(f"待搜索: {len(pending)} 个环境")

    # 并发预生成场景问题（与 MCTS 搜索串行，无竞争）
    pending_q_idx = []
    if preloaded_scenarios is not None and pending:
        for i in pending:
            s_name = preloaded_scenarios[i].name
            if s_name not in questions:
                pending_q_idx.append(i)

    def _generate_with_retry(prompt, max_retries=3):
        for attempt in range(max_retries):
            try:
                # LLM 生成并提取
                return _extract_question(llm_generate(prompt, 0.7))
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(1)

    if pending_q_idx:
        logger.info(f"预生成场景问题（{len(pending_q_idx)} 个，并发={parallel}）...")
        from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            fut_map = {
                pool.submit(_generate_with_retry, _build_question_prompt(preloaded_scenarios[i])): i
                for i in pending_q_idx
            }
            completed_q = 0
            total_q = len(pending_q_idx)
            for fut in _as_completed(fut_map):
                idx = fut_map[fut]
                s_name = preloaded_scenarios[idx].name
                try:
                    q_text = fut.result()
                    with q_lock:
                        questions[s_name] = q_text
                    _save_questions_cache()
                except Exception as e:
                    logger.error(f"场景 {idx} ({s_name}) question 生成失败(已重试3次): {e}")
                
                completed_q += 1
                if completed_q % 10 == 0 or completed_q == total_q:
                    logger.info(f"  > 问题生成进度: {completed_q}/{total_q}")

        logger.info("问题预生成阶段结束")

    def _append_incremental(sft_items, contrastive_items):
        """每完成一个环境立即追加写入，避免崩溃丢失数据"""
        if sft_items:
            with open(sft_path, 'a', encoding='utf-8') as f:
                for item in sft_items:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        if contrastive_items:
            with open(contrastive_path, 'a', encoding='utf-8') as f:
                for item in contrastive_items:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

    if parallel > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        logger.info(f"多环境并行: {parallel} parallel")
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {pool.submit(search_one_env, i): i for i in pending}
            for future in as_completed(futures):
                try:
                    sft_items, contrastive_items = future.result()
                    sft_data.extend(sft_items)
                    contrastive_data.extend(contrastive_items)
                    _append_incremental(sft_items, contrastive_items)
                except Exception as e:
                    logger.error(f"环境 {futures[future]} 搜索失败: {e}")
    else:
        for i in pending:
            sft_items, contrastive_items = search_one_env(i)
            sft_data.extend(sft_items)
            contrastive_data.extend(contrastive_items)
            _append_incremental(sft_items, contrastive_items)

    # 汇总（数据已增量写入，这里只打印统计）
    logger.info(f"\nSFT 数据: {len(sft_data)} 条 → {sft_path}")
    logger.info(f"对比数据: {len(contrastive_data)} 对 → {contrastive_path}")

    elapsed = time.time() - t0
    minutes, seconds = divmod(int(elapsed), 60)
    logger.info(f"\n总耗时: {minutes}m {seconds}s")

    # 预览命令提示
    logger.info(f"\n预览数据:")
    logger.info(f"  python3 -m datasets.synthesis.mcts.preview {sft_path}")
    logger.info(f"  python3 -m datasets.synthesis.mcts.preview {contrastive_path}")
    tree_dir = os.path.join(run_dir, "mcts_trees")
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
    parser.add_argument("--num-envs", type=int, default=0, help="搜索的环境数（0=全量）")
    parser.add_argument("--parallel", type=int, default=1, help="多环境并行数（1=串行）")
    parser.add_argument("--num-workers", type=int, default=1, help="单棵树内并发 simulation 线程数（1=串行）")
    parser.add_argument("--resume-dir", default=None, help="断点续跑：指定上次的 run_* 目录，自动跳过已完成环境")

    # LLM
    parser.add_argument("--model", default="gpt-5", help="LLM 模型名称")
    parser.add_argument("--api-key", default=None, help="单节点 API Key（如果不传 providers config）")
    parser.add_argument("--api-base", default=None, help="单节点 API Base URL")
    parser.add_argument("--providers-config", default=None, help="多中转站 JSON 配置文件路径，传入后忽略单节点配置")

    # MCTS
    parser.add_argument("--simulations", type=int, default=5, help="每棵树 MCTS 迭代次数")
    parser.add_argument("--children", type=int, default=3, help="最大子节点数")
    parser.add_argument("--depth", type=int, default=10, help="最大深度")
    parser.add_argument("--ucb-c", type=float, default=1.414, help="UCB1 探索系数")
    parser.add_argument("--expand-temp", type=float, default=0.8, help="展开温度")
    parser.add_argument("--rollout-temp", type=float, default=0.3, help="Rollout 温度")

    args = parser.parse_args()
    run_mcts(args)


if __name__ == "__main__":
    main()
