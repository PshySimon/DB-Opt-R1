"""
纯轨迹采样器

对每个场景并行跑 N 次独立 episode，
保留 improvement_pct > 3% 的轨迹作为 SFT 正样本。

支持 --questions-only 模式：仅为场景生成 question，不跑 rollout，不需要 cost model。

用法:
    # 训练数据采样（需要 cost model）
    python3 -m datasets.synthesis.trajectory.sampler \\
        --scenarios datasets/data/scenarios/ \\
        --cost-model cost_model/checkpoints/v3 \\
        --output-dir datasets/data/trajectory/ \\
        --num-rollouts 8 \\
        --providers-config configs/providers.json \\
        --parallel 8

    # 评估集只生成 question（不需要 cost model）
    python3 -m datasets.synthesis.trajectory.sampler \\
        --scenarios datasets/data/scenarios/knob_configs_eval.json \\
        --questions-only \\
        --providers-config configs/providers.json \\
        --parallel 10
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────── system prompt ───────────────────────────
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

PREDICT_CALL = '<tool_call>\n{"name": "predict_performance", "arguments": {}}\n</tool_call>'


# ───────────────────────────────── 核心函数 ──────────────────────────────────

def _get_improvement_pct(env) -> float:
    """调用 predict_performance 获取最终 TPS 提升比例（%）。"""
    try:
        obs, _, _, _ = env.step(PREDICT_CALL)
        if isinstance(obs, dict):
            return float(obs.get("improvement_pct", 0.0))
        parsed = json.loads(obs)
        return float(parsed.get("improvement_pct", 0.0))
    except Exception as e:
        logger.debug(f"predict_performance 解析失败: {e}")
        return 0.0


def _messages_to_sft(messages: list, improvement_pct: float,
                     sample_idx: int) -> dict:
    """将 rollout 的 message 列表转换为 SFT 格式，统一 role 命名。"""
    import re
    sft_messages = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user" and "<tool_response>" in content:
            # 将 rollout 的 tool_response 格式转为 role=tool
            m = re.search(r"<tool_response>\n?(.*?)\n?</tool_response>",
                          content, re.DOTALL)
            obs = m.group(1) if m else content
            sft_messages.append({"role": "tool", "content": obs})
        else:
            sft_messages.append({"role": role, "content": content})

    # 确保以 assistant 结尾
    if sft_messages and sft_messages[-1]["role"] == "tool":
        sft_messages.append({
            "role": "assistant",
            "content": "<think>调优流程已完成，以上为全部操作步骤。</think>",
        })

    return {
        "messages": sft_messages,
        "reward": round(improvement_pct / 100.0, 4),
        "improvement_pct": round(improvement_pct, 2),
        "env_sample_idx": sample_idx,
    }


def sample_one_scenario(
    sample_idx: int,
    scenarios,
    cost_model,
    llm_fn,
    knob_space_path: str,
    max_turns: int,
    num_rollouts: int,
    good_threshold: float,
    temperature: float,
) -> list:
    """对单个场景跑 num_rollouts 次 episode，返回通过阈值的 SFT 样本列表。"""
    from environment.tools import DBToolEnv
    from core.agent import rollout
    scenario = scenarios[sample_idx]
    s_name = getattr(scenario, "name", f"env_{sample_idx}")

    # 直接使用预先给定的 eval_scenarios 里面的问题
    preset_q = getattr(scenario, "question", "")
    if preset_q:
        questions = [preset_q] * num_rollouts
    else:
        # 如果既没有预设也没有问题库，就报错，因为 eval 必须对齐 SFT 数据集
        logger.error(f"  [{s_name}] 缺少预设 question (必须配合 --sft-questions)")
        return []

    good_samples = []
    threshold_pct = good_threshold * 100  # 转换为百分比

    for rollout_idx in range(num_rollouts):
        try:
            env = DBToolEnv(
                mode="train",
                cost_model=cost_model,
                max_turns=max_turns,
                knob_space_path=knob_space_path,
            )
            env.scenarios = scenarios
            env.reset(sample_idx=sample_idx)

            messages, _ = rollout(
                env=env,
                llm_fn=llm_fn,
                system_prompt=SYSTEM_PROMPT,
                user_message=questions[rollout_idx],
                max_turns=max_turns,
                temperature=temperature,
            )

            improvement_pct = _get_improvement_pct(env)

            logger.debug(
                f"  [{s_name}] rollout {rollout_idx + 1}/{num_rollouts}: "
                f"improvement_pct={improvement_pct:.2f}%"
            )

            if improvement_pct > threshold_pct:
                good_samples.append(
                    _messages_to_sft(messages, improvement_pct, sample_idx)
                )

        except Exception as e:
            logger.warning(
                f"  [{s_name}] rollout {rollout_idx + 1} 失败: {e}"
            )

    return good_samples


# ───────────────────────────────── main ──────────────────────────────────────

def build_llm_fn(args):
    """构建 LLM 生成函数：优先多中转站轮询，fallback 到单 API"""
    from core.llm.multi_client import MultiProviderLLMClient

    client = MultiProviderLLMClient(
        target_model=args.model,
        providers_config=getattr(args, 'providers_config', None),
        single_api_key=getattr(args, 'api_key', None),
        single_api_base=getattr(args, 'api_base', None),
        api_max_concurrent=getattr(args, 'api_max_concurrent', 5),
    )

    def generate(messages_or_prompt, temperature: float = 0.7) -> str:
        if isinstance(messages_or_prompt, str):
            return client.generate(messages_or_prompt, temperature=temperature)
        else:
            # list[dict] 格式，拼成纯文本
            text = "\n".join(m.get("content", "") for m in messages_or_prompt)
            return client.generate(text, temperature=temperature)

    return generate


def main():
    parser = argparse.ArgumentParser(description="轨迹采样 / 场景 question 生成")

    # 场景
    parser.add_argument("--scenarios", required=True,
                        help="ScenarioState 场景文件或目录")
    parser.add_argument("--sft-questions", default=None,
                        help="外部加载问题的 SFT JSONL 文件（常配合 evaluation 使用）")
    parser.add_argument("--knob-space", default="configs/knob_space.yaml")
    parser.add_argument("--cost-model", default=None,
                        help="Cost Model checkpoint 目录（--questions-only 时不需要）")

    # 模式
    parser.add_argument("--questions-only", action="store_true",
                        help="仅生成 question 写回场景文件，不跑 rollout")

    # LLM
    parser.add_argument("--providers-config", default=None,
                        help="多中转站配置文件（providers.json）")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-max-concurrent", type=int, default=5,
                        help="限制对单个 API 节点的底层并发数（用来保护外网 API 额度，跑本地 vLLM 时可通过参数手动调大）")

    # 采样参数
    parser.add_argument("--num-rollouts", type=int, default=8,
                        help="每个场景的 rollout 次数")
    parser.add_argument("--good-threshold", type=float, default=0.03,
                        help="SFT 正样本阈值（improvement_pct / 100），默认 0.03 即 3%%")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="LLM rollout 采样温度")
    parser.add_argument("--max-turns", type=int, default=10,
                        help="每个 episode 最大交互轮数")
    parser.add_argument("--num-scenarios", type=int, default=0,
                        help="处理场景数（0=全量）")

    # 输出
    parser.add_argument("--output-dir", default="datasets/data/trajectory/",
                        help="输出目录")
    parser.add_argument("--output-file", default="eval_trajectories.jsonl",
                        help="输出的文件名，默认为 eval_trajectories.jsonl")
    parser.add_argument("--parallel", type=int, default=4,
                        help="并发 worker 数")

    args = parser.parse_args()

    if not args.questions_only and not args.cost_model:
        parser.error("非 --questions-only 模式必须提供 --cost-model")
    if not args.providers_config and not args.api_key:
        parser.error("必须提供 --providers-config 或 --api-key")

    # 构建 LLM 函数
    llm_fn = build_llm_fn(args)

    if args.questions_only:
        _run_questions_only(args, llm_fn)
    else:
        _run_sampling(args, llm_fn)


def _run_questions_only(args, llm_fn):
    """仅为场景生成 question，写回场景文件"""
    from datasets.synthesis.scenarios.pipeline import generate_questions_for_state
    import threading

    # 加载场景 JSON
    scenario_path = args.scenarios
    with open(scenario_path, "r", encoding="utf-8") as f:
        scenarios = json.load(f)
    logger.info(f"共 {len(scenarios)} 个场景")

    pending = [(i, s) for i, s in enumerate(scenarios) if not s.get("question", "")]
    logger.info(f"需生成: {len(pending)} 个（已有: {len(scenarios) - len(pending)}）")

    if not pending:
        logger.info("无需生成")
        return

    lock = threading.Lock()
    done = [0]
    fail = [0]

    def gen(idx, sc):
        qs = generate_questions_for_state(sc, 1, llm_fn)
        return idx, qs[0]

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(gen, i, s): i for i, s in pending}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                _, q = fut.result()
                with lock:
                    scenarios[idx]["question"] = q
                    done[0] += 1
                    if done[0] % 50 == 0:
                        logger.info(f"  进度: {done[0]}/{len(pending)}")
            except Exception as e:
                with lock:
                    fail[0] += 1
                logger.warning(f"  [{idx}] 失败: {e}")

    # 输出 SFT 格式 JSONL
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "eval_scenarios.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(scenarios):
            q = s.get("question", "")
            if not q:
                continue
            item = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                ],
                "question": q,
                "env_sample_idx": i,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"完成: {done[0]}/{len(pending)} 成功 → {out_path}")


def _run_sampling(args, llm_fn):
    """完整轨迹采样（需要 cost model）"""
    # 1. 加载 cost model
    logger.info("加载 Cost Model...")
    from cost_model.model import CostModel
    cost_model = CostModel.load(args.cost_model)

    # 2. 加载场景
    logger.info("加载场景...")
    from environment.tools import DBToolEnv
    temp_env = DBToolEnv(
        mode="train",
        scenario_dir=args.scenarios,
        cost_model=cost_model,
        max_turns=args.max_turns,
        knob_space_path=args.knob_space,
    )
    scenarios = temp_env.scenarios
    logger.info(f"  可用场景: {len(scenarios)}")

    # 注入 SFT JSONL 文件里的 preset questions
    if args.sft_questions:
        logger.info(f"加载预设 SFT 问题: {args.sft_questions}")
        with open(args.sft_questions, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                idx = item.get("env_sample_idx")
                q = item.get("question")
                if idx is not None and q and idx < len(scenarios):
                    scenarios[idx].question = q

    if not scenarios:
        logger.error("没有可用场景，退出")
        sys.exit(1)

    # 3. 确定处理范围
    n = args.num_scenarios if args.num_scenarios > 0 else len(scenarios)
    n = min(n, len(scenarios))
    indices = list(range(n))

    # 4. 输出文件
    os.makedirs(args.output_dir, exist_ok=True)
    sft_path = os.path.join(args.output_dir, args.output_file)
    logger.info(
        f"开始采样: {n} 个场景, "
        f"每场景 {args.num_rollouts} 次 rollout, "
        f"threshold={args.good_threshold * 100:.0f}%, "
        f"并发={args.parallel}"
    )

    t0 = time.time()
    total_good = 0

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futs = {
            pool.submit(
                sample_one_scenario,
                idx, scenarios, cost_model, llm_fn,
                args.knob_space, args.max_turns,
                args.num_rollouts, args.good_threshold, args.temperature,
            ): idx
            for idx in indices
        }

        done = 0
        for fut in as_completed(futs):
            idx = futs[fut]
            s_name = getattr(scenarios[idx], "name", f"env_{idx}")
            try:
                good_samples = fut.result()
                if good_samples:
                    with open(sft_path, "a", encoding="utf-8") as f:
                        for item in good_samples:
                            f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    total_good += len(good_samples)

                done += 1
                improvement_pcts = [s["improvement_pct"] for s in good_samples]
                logger.info(
                    f"  [{done}/{n}] {s_name}: "
                    f"好样本 {len(good_samples)}/{args.num_rollouts} 条"
                    + (f", improvements: {[f'{p:.1f}%' for p in improvement_pcts]}"
                       if improvement_pcts else "")
                )

            except Exception as e:
                done += 1
                logger.error(f"  [{done}/{n}] {s_name}: 采样失败: {e}")

    elapsed = time.time() - t0
    logger.info(
        f"采样完成: {total_good} 条正样本，"
        f"耗时 {elapsed:.0f}s → {sft_path}"
    )


if __name__ == "__main__":
    main()
