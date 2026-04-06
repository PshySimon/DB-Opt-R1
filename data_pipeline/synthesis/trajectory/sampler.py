"""
轨迹采样器（sampler）

sampler 是数据合成的核心脚本，职责是生成 SFT 训练格式的数据。

## 两种模式

### 1. generate 模式 — 生成数据集

  --mode generate --split train   生成训练集：动态生成 question + 跑 rollout + 按 threshold 过滤
  --mode generate --split eval    生成评估集：动态生成 question，不跑 rollout，不需要 cost model

### 2. eval 模式 — 评估模型

  --mode eval --eval-questions <path>   读取固定 question，跑 rollout，保留所有轨迹

## 用法

    # 生成训练数据
    python3 -m data_pipeline.synthesis.trajectory.sampler \\
        --mode generate --split train \\
        --scenarios data_pipeline/data/scenarios/collected/ \\
        --cost-model cost_model/checkpoints/v8_lgbm \\
        --output-dir data_pipeline/data/train/ \\
        --num-rollouts 8 \\
        --providers-config configs/providers.json \\
        --parallel 8

    # 生成 eval question（不需要 cost model）
    python3 -m data_pipeline.synthesis.trajectory.sampler \\
        --mode generate --split eval \\
        --scenarios data_pipeline/data/scenarios/collected/collected_eval.json \\
        --output-dir data_pipeline/data/eval/ \\
        --providers-config configs/providers.json \\
        --parallel 10

    # 评估模型
    python3 -m data_pipeline.synthesis.trajectory.sampler \\
        --mode eval \\
        --eval-questions data_pipeline/data/eval/eval_trajectories.jsonl \\
        --scenarios data_pipeline/data/scenarios/collected/collected_eval.json \\
        --cost-model cost_model/checkpoints/v8_lgbm \\
        --output-dir eval_results/sft_qwen3_4b/ \\
        --model qwen3-4b-sft \\
        --api-base http://localhost:8000/v1 \\
        --api-key dummy
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

# 默认关闭 HTTP 请求日志噪音
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

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
                     sample_idx: int, question: str = "") -> dict:
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

    result = {
        "messages": sft_messages,
        "reward": round(improvement_pct / 100.0, 4),
        "improvement_pct": round(improvement_pct, 2),
        "env_sample_idx": sample_idx,
    }
    if question:
        result["question"] = question
    return result


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
    generate_question: bool = True,
    on_rollout_done=None,
) -> list:
    """对单个场景跑 num_rollouts 次 episode，返回通过阈值的 SFT 样本列表。

    Args:
        generate_question: True 时动态生成 question（generate/train 模式），
                          False 时使用 scenario.question 预设值（eval 模式）。
        on_rollout_done: 每个 rollout 完成后的回调 (is_good: bool) -> None
    """
    from environment.tools import DBToolEnv
    from core.agent import rollout
    scenario = scenarios[sample_idx]
    s_name = getattr(scenario, "name", f"env_{sample_idx}")

    # ---- 获取 question ----
    preset_q = getattr(scenario, "question", "")

    if preset_q:
        # 有预设 question（eval 模式注入的，或场景自带的）
        questions = [preset_q] * num_rollouts
    elif generate_question:
        # 没有预设，动态生成（generate/train 模式）
        try:
            from data_pipeline.synthesis.scenarios.pipeline import generate_questions_for_state
            questions = generate_questions_for_state(scenario, num_rollouts, llm_fn)
        except Exception as e:
            logger.warning(f"  [{s_name}] question 生成失败: {e}，使用默认问题")
            questions = ["请帮我优化一下数据库的性能配置。"] * num_rollouts
    else:
        # eval 模式但没有预设 question，报错
        logger.error(f"  [{s_name}] eval 模式缺少预设 question")
        return []

    all_samples = []
    threshold_pct = good_threshold * 100  # 仅用于进度条统计

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
            is_good = improvement_pct > threshold_pct

            all_samples.append(
                _messages_to_sft(messages, improvement_pct, sample_idx,
                                 question=questions[rollout_idx])
            )

        except Exception as e:
            is_good = False
            logger.warning(
                f"  [{s_name}] rollout {rollout_idx + 1} 失败: {e}"
            )

        if on_rollout_done:
            on_rollout_done(is_good)

    return all_samples


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
    parser = argparse.ArgumentParser(description="轨迹采样器（生成数据集 / 评估模型）")

    # ---- 模式 ----
    parser.add_argument("--mode", required=True, choices=["generate", "eval"],
                        help="generate: 生成数据集; eval: 评估模型")
    parser.add_argument("--split", default="train", choices=["train", "eval"],
                        help="generate 模式下的子模式: train=生成训练集, eval=仅生成 question")
    parser.add_argument("--eval-questions", default=None,
                        help="eval 模式必需: 预生成的 question 文件（JSONL, 含 question + env_sample_idx）")

    # ---- 场景 ----
    parser.add_argument("--scenarios", required=True, nargs='+',
                        help="ScenarioState 场景文件或目录（支持多个）")
    parser.add_argument("--knob-space", default="configs/knob_space.yaml")
    parser.add_argument("--cost-model", default=None,
                        help="Cost Model checkpoint 目录（generate/eval 生成 question 时不需要）")

    # ---- LLM ----
    parser.add_argument("--providers-config", default=None,
                        help="多中转站配置文件（providers.json）")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-max-concurrent", type=int, default=5,
                        help="限制对单个 API 节点的底层并发数")

    # ---- 采样参数 ----
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

    # ---- 场景过滤 ----
    from core.db.scenario_filter import add_filter_args
    add_filter_args(parser)

    # ---- 输出 ----
    parser.add_argument("--output-dir", default="data_pipeline/data/trajectory/",
                        help="输出目录")
    parser.add_argument("--output-file", default=None,
                        help="输出文件名（默认: generate/train→sft_trajectories.jsonl, "
                             "generate/eval→eval_trajectories.jsonl, eval→sft_trajectories.jsonl）")
    parser.add_argument("--parallel", type=int, default=4,
                        help="并发 worker 数")
    parser.add_argument("--verbose-http", action="store_true",
                        help="显示 HTTP 请求日志（默认关闭）")

    # ---- 向后兼容（已废弃，解析但忽略） ----
    parser.add_argument("--questions-only", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--sft-questions", default=None,
                        help=argparse.SUPPRESS)

    args = parser.parse_args()

    # ---- HTTP 日志开关 ----
    if args.verbose_http:
        logging.getLogger("httpx").setLevel(logging.INFO)
        logging.getLogger("openai").setLevel(logging.INFO)

    # ---- 向后兼容转换 ----
    if args.questions_only:
        logger.warning("--questions-only 已废弃，自动转为 --mode generate --split eval")
        args.mode = "generate"
        args.split = "eval"
    if args.sft_questions and not args.eval_questions:
        logger.warning("--sft-questions 已废弃，自动转为 --eval-questions")
        args.eval_questions = args.sft_questions

    # ---- 参数校验 ----
    if args.mode == "eval" and not args.eval_questions:
        parser.error("eval 模式必须提供 --eval-questions")
    if args.mode == "eval" and not args.cost_model:
        parser.error("eval 模式必须提供 --cost-model")
    if args.mode == "generate" and args.split == "train" and not args.cost_model:
        parser.error("generate/train 模式必须提供 --cost-model")
    if not args.providers_config and not args.api_key:
        parser.error("必须提供 --providers-config 或 --api-key")

    # ---- 默认输出文件名 ----
    if args.output_file is None:
        if args.mode == "generate" and args.split == "eval":
            args.output_file = "eval_trajectories.jsonl"
        else:
            args.output_file = "sft_trajectories.jsonl"

    # ---- 构建 LLM 函数 ----
    llm_fn = build_llm_fn(args)

    # ---- 分派 ----
    if args.mode == "generate" and args.split == "eval":
        _run_generate_eval(args, llm_fn)
    elif args.mode == "generate" and args.split == "train":
        _run_generate_train(args, llm_fn)
    elif args.mode == "eval":
        _run_eval(args, llm_fn)


# ─────────────── generate/eval: 仅生成 question（不跑 rollout）─────────────

def _run_generate_eval(args, llm_fn):
    """generate --split eval: 为场景生成 question，不跑 rollout"""
    from data_pipeline.synthesis.scenarios.pipeline import generate_questions_for_state
    import threading

    # 加载场景 JSON（generate/eval 模式只用单文件）
    scenario_path = args.scenarios[0] if isinstance(args.scenarios, list) else args.scenarios
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

    # 输出 JSONL
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, args.output_file)
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


# ─────────────── generate/train: 生成训练集（动态 question + rollout）────────

def _run_generate_train(args, llm_fn):
    """generate --split train: 动态生成 question + 跑 rollout + 过滤"""
    _run_rollout(args, llm_fn, generate_question=True)


# ─────────────── eval: 评估模型（固定 question + rollout）────────────────────

def _run_eval(args, llm_fn):
    """eval 模式: 用固定 question 跑 rollout，保存原始轨迹（不计算指标、不过滤）。

    指标在 evaluate/run.py 阶段由 Cost Model 重新计算。
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # 1. 加载 Cost Model
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

    # 场景过滤
    from core.db.scenario_filter import apply_filter_args
    scenarios = apply_filter_args(scenarios, args)
    logger.info(f"  可用场景: {len(scenarios)}")

    # 3. 注入预生成的 question
    if args.eval_questions:
        logger.info(f"加载 eval question: {args.eval_questions}")
        with open(args.eval_questions, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                idx = item.get("env_sample_idx")
                q = item.get("question")
                if idx is not None and q and idx < len(scenarios):
                    scenarios[idx].question = q

    if not scenarios:
        logger.error("没有可用场景，退出")
        sys.exit(1)

    # 4. 确定处理范围
    n = args.num_scenarios if args.num_scenarios > 0 else len(scenarios)
    n = min(n, len(scenarios))
    indices = list(range(n))

    # 5. 输出文件
    os.makedirs(args.output_dir, exist_ok=True)
    sft_path = os.path.join(args.output_dir, args.output_file)

    logger.info(f"开始 eval: {n} 个场景, 并发={args.parallel}")

    def eval_one(idx):
        """跑单个场景的 eval rollout，返回原始轨迹 dict。"""
        from environment.tools import DBToolEnv
        from core.agent import rollout
        import re

        scenario = scenarios[idx]
        question = getattr(scenario, "question", "")
        if not question:
            return None

        env = DBToolEnv(
            mode="train",
            cost_model=cost_model,
            max_turns=args.max_turns,
            knob_space_path=args.knob_space,
        )
        env.scenarios = scenarios
        env.reset(sample_idx=idx)

        messages, _ = rollout(
            env=env,
            llm_fn=llm_fn,
            system_prompt=SYSTEM_PROMPT,
            user_message=question,
            max_turns=args.max_turns,
            temperature=0.3,  # eval 用低温度确保稳定
        )

        # 格式化 messages（tool_response → role=tool）
        sft_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user" and "<tool_response>" in content:
                m = re.search(r"<tool_response>\n?(.*?)\n?</tool_response>",
                              content, re.DOTALL)
                obs = m.group(1) if m else content
                sft_messages.append({"role": "tool", "content": obs})
            else:
                sft_messages.append({"role": role, "content": content})

        return {
            "messages": sft_messages,
            "question": question,
            "env_sample_idx": idx,
        }

    t0 = time.time()
    total_ok = 0

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futs = {pool.submit(eval_one, idx): idx for idx in indices}

        done = 0
        for fut in as_completed(futs):
            idx = futs[fut]
            s_name = getattr(scenarios[idx], "name", f"env_{idx}")
            try:
                result = fut.result()
                if result:
                    with open(sft_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    total_ok += 1

                done += 1
                n_steps = sum(1 for m in result["messages"] if m["role"] == "assistant") if result else 0
                logger.info(f"  [{done}/{n}] {s_name}: {n_steps} 步")
            except Exception as e:
                done += 1
                logger.error(f"  [{done}/{n}] {s_name}: 失败: {e}")

    elapsed = time.time() - t0
    logger.info(f"eval 完成: {total_ok}/{n} 个场景, 耗时 {elapsed:.0f}s → {sft_path}")


# ─────────────── 通用 rollout 执行 ───────────────────────────────────────────

def _run_rollout(args, llm_fn, generate_question: bool):
    """通用 rollout 逻辑，generate/train 和 eval 共用"""
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

    # 调用方显式过滤场景
    from core.db.scenario_filter import apply_filter_args
    scenarios = apply_filter_args(scenarios, args)
    logger.info(f"  可用场景: {len(scenarios)}")

    # 3. eval 模式：注入预生成的 question
    if not generate_question and args.eval_questions:
        logger.info(f"加载 eval question: {args.eval_questions}")
        with open(args.eval_questions, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                idx = item.get("env_sample_idx")
                q = item.get("question")
                if idx is not None and q and idx < len(scenarios):
                    scenarios[idx].question = q

    if not scenarios:
        logger.error("没有可用场景，退出")
        sys.exit(1)

    # 4. 确定处理范围
    n = args.num_scenarios if args.num_scenarios > 0 else len(scenarios)
    n = min(n, len(scenarios))
    indices = list(range(n))

    # 5. 输出文件
    os.makedirs(args.output_dir, exist_ok=True)
    sft_path = os.path.join(args.output_dir, args.output_file)

    # 6. 断点续跑：读取已有输出，跳过已完成的场景
    done_counts = {}  # env_sample_idx -> 已完成 rollout 数
    if os.path.isfile(sft_path):
        with open(sft_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    idx = item.get("env_sample_idx")
                    if idx is not None:
                        done_counts[idx] = done_counts.get(idx, 0) + 1
                except json.JSONDecodeError:
                    pass

    skipped = [i for i in indices if done_counts.get(i, 0) >= args.num_rollouts]
    indices = [i for i in indices if done_counts.get(i, 0) < args.num_rollouts]

    if skipped:
        logger.info(f"  断点续跑: 跳过 {len(skipped)} 个已完成场景，剩余 {len(indices)} 个")

    if not indices:
        logger.info("所有场景已完成，无需重跑")
        return

    mode_label = "generate/train" if generate_question else "eval"
    logger.info(
        f"开始采样 [{mode_label}]: {len(indices)} 个场景, "
        f"每场景 {args.num_rollouts} 次 rollout, "
        f"threshold={args.good_threshold * 100:.0f}%, "
        f"并发={args.parallel}"
    )

    t0 = time.time()
    total_good = 0
    total_rollouts = len(indices) * args.num_rollouts
    import threading
    from tqdm import tqdm
    pbar = tqdm(total=total_rollouts, desc="采样进度", unit="rollout",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] 好样本:{postfix}")
    pbar.set_postfix_str("0")
    _pbar_lock = threading.Lock()

    def _on_rollout_done(is_good: bool):
        nonlocal total_good
        with _pbar_lock:
            if is_good:
                total_good += 1
            pbar.update(1)
            pbar.set_postfix_str(str(total_good))

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futs = {
            pool.submit(
                sample_one_scenario,
                idx, scenarios, cost_model, llm_fn,
                args.knob_space, args.max_turns,
                args.num_rollouts, args.good_threshold, args.temperature,
                generate_question,
                on_rollout_done=_on_rollout_done,
            ): idx
            for idx in indices
        }

        for fut in as_completed(futs):
            idx = futs[fut]
            try:
                good_samples = fut.result()
                if good_samples:
                    with open(sft_path, "a", encoding="utf-8") as f:
                        for item in good_samples:
                            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            except Exception as e:
                s_name = getattr(scenarios[idx], "name", f"env_{idx}")
                logger.error(f"  {s_name}: 采样失败: {e}")

    pbar.close()

    elapsed = time.time() - t0
    logger.info(
        f"采样完成: {total_good} 条正样本，"
        f"耗时 {elapsed:.0f}s → {sft_path}"
    )


if __name__ == "__main__":
    main()
