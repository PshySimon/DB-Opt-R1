#!/bin/bash
# 纯 rollout 基准：抽取 16 条 prompt，测多轮工具交互耗时
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/root/private_data/DB-Opt-R1/model_save/sft_qwen3_4b_cleaned_merged}"
TRAIN_DATA="${TRAIN_DATA:-./data_pipeline/data/train/sft_trajectories.jsonl}"
SCENARIO_FILES="${SCENARIO_FILES:-data_pipeline/data/scenarios/collected/collected_server1.json data_pipeline/data/scenarios/collected/collected_server2.json data_pipeline/data/scenarios/collected/collected_server3.json}"
COST_MODEL="${COST_MODEL:-./cost_model/checkpoints/v10_lgbm}"

NUM_SAMPLES="${NUM_SAMPLES:-16}"
NUM_GEN="${NUM_GEN:-4}"
MAX_TURNS="${MAX_TURNS:-10}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-}"
VLLM_SERVER_HOST="${VLLM_SERVER_HOST:-127.0.0.1}"
VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-8000}"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-qwen3-4b-sft}"
VLLM_TIMEOUT="${VLLM_TIMEOUT:-300}"
VLLM_MAX_TOKENS="${VLLM_MAX_TOKENS:-1024}"
ROLLOUT_LOG_INTERVAL="${ROLLOUT_LOG_INTERVAL:-1}"

echo "============================================"
echo "  GRPO 16 样本纯 Rollout 基准"
echo "============================================"
echo "模型(仅 tokenizer): $MODEL_PATH"
echo "训练数据:           $TRAIN_DATA"
echo "样本数:             $NUM_SAMPLES"
echo "每样本轨迹数:       $NUM_GEN"
echo "最大轮数:           $MAX_TURNS"
echo "Rollout 并发:       ${ROLLOUT_BATCH_SIZE:-全部活跃轨迹}"
echo "vLLM tokens:        $VLLM_MAX_TOKENS"
echo "vLLM 服务:          http://${VLLM_SERVER_HOST}:${VLLM_SERVER_PORT}/v1"
echo "============================================"

PYTHONPATH=. python - \
  "$MODEL_PATH" \
  "$TRAIN_DATA" \
  "$SCENARIO_FILES" \
  "$COST_MODEL" \
  "$NUM_SAMPLES" \
  "$NUM_GEN" \
  "$MAX_TURNS" \
  "$ROLLOUT_BATCH_SIZE" \
  "$VLLM_SERVER_HOST" \
  "$VLLM_SERVER_PORT" \
  "$VLLM_MODEL_NAME" \
  "$VLLM_TIMEOUT" \
  "$VLLM_MAX_TOKENS" \
  "$ROLLOUT_LOG_INTERVAL" <<'PY'
import json
import logging
import time
import sys

from openai import OpenAI
from transformers import AutoTokenizer

from environment.tools import DBToolEnv
from training.data_utils import SYSTEM_PROMPT


logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger("rollout_bench")


def format_rollout_turn_log(turn_idx, active_count, batch_size, prompt_token_lengths, elapsed_s):
    avg_prompt_tokens = int(sum(prompt_token_lengths) / max(len(prompt_token_lengths), 1))
    max_prompt_tokens = max(prompt_token_lengths) if prompt_token_lengths else 0
    min_prompt_tokens = min(prompt_token_lengths) if prompt_token_lengths else 0
    return (
        f"[rollout-only] turn {turn_idx + 1} "
        f"active={active_count}/{batch_size} "
        f"prompt_tokens(avg={avg_prompt_tokens}, min={min_prompt_tokens}, max={max_prompt_tokens}) "
        f"elapsed={elapsed_s:.2f}s"
    )


(
    model_path,
    train_data,
    scenario_files_raw,
    cost_model_path,
    num_samples,
    num_gen,
    max_turns,
    rollout_batch_size,
    vllm_host,
    vllm_port,
    vllm_model_name,
    vllm_timeout,
    vllm_max_tokens,
    rollout_log_interval,
) = sys.argv[1:15]

scenario_files = scenario_files_raw.split()
num_samples = int(num_samples)
num_gen = int(num_gen)
max_turns = int(max_turns)
rollout_batch_size = int(rollout_batch_size) if rollout_batch_size else None
vllm_port = int(vllm_port)
vllm_timeout = int(vllm_timeout)
vllm_max_tokens = int(vllm_max_tokens)
rollout_log_interval = int(rollout_log_interval)

base_url = vllm_host.strip()
if "://" not in base_url:
    base_url = f"http://{base_url}"
base_url = f"{base_url.rstrip('/')}:{vllm_port}/v1" if base_url.count(":") == 1 and base_url.startswith("http") else base_url
if not base_url.endswith("/v1"):
    base_url = f"{base_url.rstrip('/')}/v1"

logger.info("加载 tokenizer: %s", model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

logger.info("连接 vLLM server: %s (model=%s)", base_url, vllm_model_name)
client = OpenAI(
    api_key="EMPTY",
    base_url=base_url,
    timeout=vllm_timeout,
    max_retries=0,
    default_headers={"User-Agent": "Mozilla/5.0"},
)

logger.info("加载 Cost Model: %s", cost_model_path)
from cost_model.model import CostModel
cost_model = CostModel.load(cost_model_path)

logger.info("加载场景")
all_scenarios = DBToolEnv._load_scenarios(scenario_files)

def env_factory():
    env = DBToolEnv(
        mode="train",
        cost_model=cost_model,
        max_turns=max_turns,
        knob_space_path="configs/knob_space.yaml",
    )
    env.scenarios = all_scenarios
    return env

selected = []
seen = set()
with open(train_data, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        question = row.get("question", "")
        env_sample_idx = row.get("env_sample_idx")
        if not question or question in seen:
            continue
        seen.add(question)
        selected.append((question, env_sample_idx))
        if len(selected) >= num_samples:
            break

if len(selected) < num_samples:
    raise SystemExit(f"only found {len(selected)} unique questions, expected at least {num_samples}")

logger.info("抽样完成: %s 条 prompt", len(selected))

expanded = []
for question, env_sample_idx in selected:
    for _ in range(num_gen):
        expanded.append((question, env_sample_idx))

logger.info("展开后 rollout 轨迹数: %s", len(expanded))

envs = []
messages_list = []
improvements = [0.0] * len(expanded)
active = list(range(len(expanded)))

for question, env_sample_idx in expanded:
    env = env_factory()
    env.reset(sample_idx=env_sample_idx) if env_sample_idx is not None else env.reset()
    envs.append(env)

    tools_desc = env.tools_format_func()
    full_system = f"{SYSTEM_PROMPT}\n\n{tools_desc}"
    messages_list.append([
        {"role": "system", "content": full_system},
        {"role": "user", "content": question},
    ])

total_start = time.perf_counter()
total_generated_turns = 0
total_vllm_calls = 0

for turn in range(max_turns):
    if not active:
        break

    turn_start = time.perf_counter()
    prompt_token_lengths = []
    still_active = []
    chunk_size = rollout_batch_size or len(active)

    for offset in range(0, len(active), chunk_size):
        chunk_indices = active[offset:offset + chunk_size]
        batch_texts = []
        for idx in chunk_indices:
            text = tokenizer.apply_chat_template(
                messages_list[idx], tokenize=False, add_generation_prompt=True
            )
            batch_texts.append(text)

        prompt_token_lengths.extend(
            len(tokenizer.encode(text, add_special_tokens=False))
            for text in batch_texts
        )

        try:
            response = client.completions.create(
                model=vllm_model_name,
                prompt=batch_texts,
                temperature=1.0,
                top_p=1.0,
                max_tokens=vllm_max_tokens,
                stop=["</tool_call>"],
            )
            outputs = [""] * len(batch_texts)
            for choice in response.choices:
                outputs[int(choice.index)] = choice.text or ""
        except Exception as exc:
            logger.warning("批量请求失败，降级逐条请求: %s", exc)
            outputs = []
            for text in batch_texts:
                response = client.completions.create(
                    model=vllm_model_name,
                    prompt=text,
                    temperature=1.0,
                    top_p=1.0,
                    max_tokens=vllm_max_tokens,
                    stop=["</tool_call>"],
                )
                outputs.append(response.choices[0].text or "")

        total_vllm_calls += 1
        for pos, idx in enumerate(chunk_indices):
            gen_text = outputs[pos] or ""
            if "<tool_call>" in gen_text and "</tool_call>" not in gen_text:
                gen_text += "</tool_call>"

            messages_list[idx].append({"role": "assistant", "content": gen_text})

            if "</tool_call>" in gen_text:
                obs, _, done, _ = envs[idx].step(gen_text)
                messages_list[idx].append({
                    "role": "user",
                    "content": f"<tool_response>\n{obs}\n</tool_response>"
                })
                if not done:
                    still_active.append(idx)

    total_generated_turns += len(active)
    active = still_active

    if rollout_log_interval > 0 and (
        turn == 0 or
        (turn + 1) % rollout_log_interval == 0 or
        not active
    ):
        logger.info(
            format_rollout_turn_log(
                turn_idx=turn,
                active_count=len(active),
                batch_size=len(expanded),
                prompt_token_lengths=prompt_token_lengths,
                elapsed_s=time.perf_counter() - turn_start,
            )
        )

for i, env in enumerate(envs):
    imp = getattr(env, "improvement_pct", 0.0)
    if imp == 0.0:
        try:
            obs, _, _, _ = env.step(
                '<tool_call>\n{"name": "predict_performance", "arguments": {}}\n</tool_call>'
            )
            imp = float(json.loads(obs).get("improvement_pct", 0.0))
        except Exception:
            pass
    improvements[i] = imp

elapsed = time.perf_counter() - total_start
avg_turns = total_generated_turns / max(len(expanded), 1)
avg_improvement = sum(improvements) / max(len(improvements), 1)

logger.info("============================================")
logger.info("纯 rollout 基准完成")
logger.info("prompt 数: %s", num_samples)
logger.info("rollout 轨迹数: %s", len(expanded))
logger.info("总生成轮次: %s", total_generated_turns)
logger.info("vLLM 请求次数: %s", total_vllm_calls)
logger.info("平均每轨迹轮次: %.2f", avg_turns)
logger.info("平均 improvement: %.4f", avg_improvement)
logger.info("总耗时: %.2fs", elapsed)
logger.info("约合: %dm %ds", int(elapsed // 60), int(elapsed % 60))
logger.info("============================================")
PY
