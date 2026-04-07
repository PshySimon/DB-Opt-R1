"""
trl GRPO 多轮工具调用训练入口

与 verl 路径完全对齐：
- 普通 chat 格式，不走 function calling
- 工具描述写在 system prompt
- 模型生成 <tool_call>...</tool_call> 文本
- 代码用正则提取、调用 ToolEnv.step()
- tool response 用 <|im_start|>user\n<tool_response>...\n</tool_response><|im_end|> 拼回去

使用 TRL 的 rollout_func 手动控制多轮生成。

Usage:
    PYTHONPATH=. python -m training.trl.grpo \\
        --model_path model_save/sft/checkpoint-xxx \\
        --train_data data_pipeline/data/train/sft_trajectories_v2.jsonl \\
        --scenario_files \\
            data_pipeline/data/scenarios/collected/collected_server1.json \\
            data_pipeline/data/scenarios/collected/collected_server2.json \\
            data_pipeline/data/scenarios/collected/collected_server3.json \\
        --cost_model cost_model/checkpoints/v9_lgbm \\
        --output_dir model_save/grpo/
"""

import re
import json
import math
import argparse
import logging
import random
from copy import deepcopy

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

from training.data_utils import SYSTEM_PROMPT
from training.reward_score import compute_score_format

logger = logging.getLogger(__name__)

# 与 verl grpo_trainer.yaml 中 tool_custom_response_template 完全一致
TOOL_RESPONSE_TEMPLATE = """
<|im_start|>user
<tool_response>
{tool_response}
</tool_response><|im_end|>
<|im_start|>assistant
<think>
"""


# ============================================================
# 多轮 Rollout
# ============================================================

def multi_turn_rollout(prompts, trainer):
    """
    自定义 rollout_func，与 verl 的 ToolGenerationManager.run_llm_loop 对齐。

    流程（每个 prompt）：
    1. tokenize prompt（含 system prompt + 工具描述）
    2. model.generate() 生成文本
    3. 检测 <tool_call>...</tool_call>
    4. 用 ToolEnv.step() 执行工具
    5. 拼接 tool_response 到上下文，继续生成
    6. 重复直到没有 tool_call 或达到 max_turns
    7. 用完整轨迹做一次 forward pass 得到 logprobs

    Args:
        prompts: 当前进程分配到的 prompt 列表
        trainer: GRPOTrainer 实例

    Returns:
        dict with prompt_ids, completion_ids, logprobs, + reward 用的额外字段
    """
    model = trainer.model
    tokenizer = trainer.processing_class
    device = model.device

    num_generations = trainer.args.num_generations
    max_completion_length = trainer.args.max_completion_length

    # 从 trainer 的自定义属性读取配置
    max_turns = getattr(trainer, '_max_turns', 10)
    envs_factory = getattr(trainer, '_envs_factory', None)

    all_prompt_ids = []
    all_completion_ids = []
    all_logprobs = []
    all_improvements = []  # 传给 reward 函数

    for prompt in prompts:
        # prompt 是 dict: {"role": ..., "content": ...} 列表
        # dataset 额外列 env_sample_idx, question 会由 TRL 传入
        # 但 rollout_func 只收到 prompts，所以我们把 env_sample_idx 编码在 prompt 里
        # prompt format: [{"role":"system",...}, {"role":"user","content":"..."}]
        # env_sample_idx 从 trainer._prompt_env_map 查询

        # 每个 prompt 生成 num_generations 个 rollout
        for _ in range(num_generations):
            # 创建独立的环境
            env = envs_factory()

            # 从 prompt 中提取 user question，查找对应的 env_sample_idx
            prompt_env_map = getattr(trainer, '_prompt_env_map', {})
            if isinstance(prompt, list):
                user_msg = next(
                    (m["content"] for m in prompt if m["role"] == "user"), ""
                )
            else:
                user_msg = str(prompt)
            sample_idx = prompt_env_map.get(user_msg)

            # reset 到对应场景
            if sample_idx is not None:
                env.reset(sample_idx=sample_idx)
            else:
                env.reset()  # 随机场景

            # 构建 messages（与 core/agent.py 的 rollout 完全一致）
            tools_desc = env.tools_format_func()
            full_system = f"{SYSTEM_PROMPT}\n\n{tools_desc}"

            if isinstance(prompt, list):
                # conversational format
                messages = list(prompt)  # copy
                # 确保 system prompt 包含 tool 描述
                if messages and messages[0]["role"] == "system":
                    messages[0]["content"] = full_system
                else:
                    messages.insert(0, {"role": "system", "content": full_system})
            else:
                messages = [
                    {"role": "system", "content": full_system},
                    {"role": "user", "content": str(prompt)},
                ]

            # tokenize prompt
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tokenizer.encode(
                prompt_text, return_tensors="pt", add_special_tokens=False
            ).to(device)

            # 多轮生成
            full_completion_ids = torch.tensor([], dtype=torch.long, device=device)
            generation_masks = []  # True = 模型生成, False = tool response

            for turn in range(max_turns):
                current_input = torch.cat(
                    [prompt_ids, full_completion_ids.unsqueeze(0)] if full_completion_ids.numel() > 0
                    else [prompt_ids],
                    dim=1
                )

                # 检查长度
                remaining = max_completion_length - full_completion_ids.numel()
                if remaining <= 0:
                    break

                # 生成
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=current_input,
                        max_new_tokens=min(remaining, 1024),
                        do_sample=True,
                        temperature=1.0,
                        top_p=1.0,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                new_ids = outputs[0, current_input.shape[1]:]
                new_text = tokenizer.decode(new_ids, skip_special_tokens=False)

                # 检测 </tool_call>
                if "</tool_call>" in new_text:
                    # 截断到第一个 </tool_call>
                    cut_text = new_text.split("</tool_call>")[0] + "</tool_call>"
                    cut_ids = tokenizer.encode(
                        cut_text, return_tensors="pt", add_special_tokens=False
                    ).to(device)[0]

                    full_completion_ids = torch.cat([full_completion_ids, cut_ids])
                    generation_masks.extend([True] * len(cut_ids))

                    # 用 env.step() 执行工具（与 verl 一致）
                    obs, reward, done, info = env.step(cut_text)

                    # 拼接 tool response（与 verl tool_custom_response_template 一致）
                    tool_resp_text = TOOL_RESPONSE_TEMPLATE.format(tool_response=obs)
                    tool_resp_ids = tokenizer.encode(
                        tool_resp_text, return_tensors="pt", add_special_tokens=False
                    ).to(device)[0]

                    full_completion_ids = torch.cat([full_completion_ids, tool_resp_ids])
                    generation_masks.extend([False] * len(tool_resp_ids))

                    if done:
                        break
                else:
                    # 没有 tool_call，模型正常结束
                    full_completion_ids = torch.cat([full_completion_ids, new_ids])
                    generation_masks.extend([True] * len(new_ids))
                    break

            # 截断到 max_completion_length
            if full_completion_ids.numel() > max_completion_length:
                full_completion_ids = full_completion_ids[:max_completion_length]
                generation_masks = generation_masks[:max_completion_length]

            # 计算 logprobs：forward pass 整个轨迹
            full_input = torch.cat([prompt_ids[0], full_completion_ids]).unsqueeze(0)
            with torch.no_grad():
                model_output = model(input_ids=full_input)
                # logits 对应 completion 部分
                logits = model_output.logits[0, prompt_ids.shape[1] - 1:-1, :]
                log_probs = torch.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(
                    -1, full_completion_ids.unsqueeze(-1)
                ).squeeze(-1)

            # 获取 improvement（用于 reward 计算）
            improvement = getattr(env, 'improvement_pct', 0.0)
            if improvement == 0.0:
                # 尝试最后调一次 predict_performance
                try:
                    predict_text = '<tool_call>\n{"name": "predict_performance", "arguments": {}}\n</tool_call>'
                    obs, _, _, _ = env.step(predict_text)
                    parsed = json.loads(obs) if isinstance(obs, str) else obs
                    improvement = float(parsed.get("improvement_pct", 0.0))
                except Exception:
                    pass

            all_prompt_ids.append(prompt_ids[0].cpu())
            all_completion_ids.append(full_completion_ids.cpu())
            all_logprobs.append(token_log_probs.cpu())
            all_improvements.append(improvement)

    return {
        "prompt_ids": all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs": all_logprobs,
        "improvement_pct": all_improvements,
    }


# ============================================================
# Reward 函数
# ============================================================

def reward_fn(completions, improvement_pct=None, **kwargs):
    """
    Reward 函数，与 verl 的 DBRewardManager 对齐。

    总分 = format_score + answer_score
    - format_score: compute_score_format（0~1.5）
    - answer_score: log(1 + improvement)，improvement 从 rollout 传入
    """
    rewards = []
    improvements = improvement_pct or [0.0] * len(completions)

    for i, completion in enumerate(completions):
        # 提取文本
        if isinstance(completion, list):
            content = " ".join(
                m.get("content", "") for m in completion
                if m.get("role") == "assistant"
            )
        else:
            content = str(completion)

        # 格式分
        format_score = compute_score_format(content)

        # 任务分
        imp = improvements[i] / 100.0 if i < len(improvements) else 0.0
        imp = min(2.0, max(0.0, imp))
        answer_score = math.log(1 + imp) if imp > 0 else 0.0

        rewards.append(format_score + answer_score)

    return rewards


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="trl GRPO 多轮工具调用训练")

    # 模型
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", default="./model_save/grpo/")

    # 数据
    parser.add_argument("--train_data", required=True,
                        help="SFT 轨迹文件（JSONL），从中提取 prompt")
    parser.add_argument("--scenario_files", nargs="+", required=True,
                        help="场景 JSON 文件（与 SFT 数据的 env_sample_idx 对应）")

    # Cost Model
    parser.add_argument("--cost_model", default=None)

    # 训练超参
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_completion_length", type=int, default=4096)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--max_turns", type=int, default=10)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--knob_space", default="configs/knob_space.yaml")
    parser.add_argument("--flash_attn", action="store_true", default=False)
    args = parser.parse_args()

    # 加载 Cost Model
    cost_model = None
    if args.cost_model:
        try:
            from cost_model.model import CostModel
            cost_model = CostModel.load(args.cost_model)
            print(f"✅ 加载 Cost Model: {args.cost_model}")
        except Exception as e:
            print(f"[WARNING] Cost Model 加载失败: {e}")

    # ---- 从 SFT 轨迹提取 prompt ----
    # 每条轨迹有 question + env_sample_idx
    # question 作为 user message，env_sample_idx 用于 rollout 时 reset 到对应场景
    prompt_records = []  # [{"question": str, "env_sample_idx": int}]
    prompt_env_map = {}  # question -> env_sample_idx
    seen_questions = set()

    with open(args.train_data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            question = data.get("question", "")
            idx = data.get("env_sample_idx", None)
            if not question or question in seen_questions:
                continue
            seen_questions.add(question)
            prompt_records.append({"question": question, "env_sample_idx": idx})
            if idx is not None:
                prompt_env_map[question] = idx

    # 构建 TRL dataset（conversational format）
    dataset = Dataset.from_list([
        {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": r["question"]},
            ]
        }
        for r in prompt_records
    ])
    print(f"📊 数据集: {len(dataset)} 条独立 prompt（从 {args.train_data} 提取）")

    # 加载模型
    print(f"🔄 加载模型: {args.model_path}")
    attn_impl = "flash_attention_2" if args.flash_attn else "eager"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # LoRA
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank // 2,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

    # GRPOConfig
    training_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        bf16=args.bf16,
        fp16=not args.bf16,
        gradient_checkpointing=True,
        logging_steps=1,
        save_steps=50,
        save_total_limit=3,
        report_to="none",
        max_steps=args.max_steps,
        seed=42,
    )

    # 环境工厂（与 verl 的 DBToolEnv 使用完全相同）
    from environment.tools import DBToolEnv

    # 预加载场景（只加载一次，所有 env 共享）
    all_scenarios = DBToolEnv._load_scenarios(args.scenario_files)
    print(f"📦 加载 {len(all_scenarios)} 个场景")

    def env_factory():
        env = DBToolEnv(
            mode="train",
            cost_model=cost_model,
            max_turns=args.max_turns,
            knob_space_path=args.knob_space,
        )
        env.scenarios = all_scenarios
        return env

    # Trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_config,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        train_dataset=dataset,
        peft_config=peft_config,
        rollout_func=multi_turn_rollout,
    )

    # 把环境工厂、配置、prompt→场景映射挂到 trainer 上
    trainer._max_turns = args.max_turns
    trainer._envs_factory = env_factory
    trainer._prompt_env_map = prompt_env_map

    # 显存统计
    if torch.cuda.is_available():
        total_p = sum(p.numel() for p in trainer.model.parameters())
        train_p = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        print(f"\n{'='*60}")
        print(f"  总参数: {total_p/1e6:.1f}M | 可训练: {train_p/1e6:.1f}M ({train_p/total_p:.2%})")
        print(f"{'='*60}")
        torch.cuda.reset_peak_memory_stats()

    print("\n🚀 开始 GRPO 多轮工具调用训练...")
    trainer.train()

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\n📊 峰值显存: {peak:.2f} GB")

    trainer.save_model(args.output_dir)
    print(f"✅ 模型保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
