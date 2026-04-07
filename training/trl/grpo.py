"""
trl GRPO 多轮工具调用训练入口

子类化 TRL 0.16.1 的 GRPOTrainer，覆写 _generate_and_score_completions
实现多轮工具交互（与 verl 路径完全对齐）。

Usage:
    PYTHONPATH=. python -m training.trl.grpo \\
        --model_path model_save/sft/checkpoint-xxx \\
        --train_data data_pipeline/data/train/sft_trajectories.jsonl \\
        --scenario_files \\
            data_pipeline/data/scenarios/collected/collected_server1.json \\
            data_pipeline/data/scenarios/collected/collected_server2.json \\
            data_pipeline/data/scenarios/collected/collected_server3.json \\
        --cost_model cost_model/checkpoints/v9_lgbm \\
        --output_dir model_save/grpo/
"""

import json
import math
import argparse
import logging
from collections import defaultdict
from typing import Any, Union

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from trl.trainer.utils import pad, selective_log_softmax
from trl.data_utils import maybe_apply_chat_template, is_conversational, apply_chat_template
from peft import LoraConfig

from accelerate.utils import gather, gather_object, broadcast_object_list
from trl.models import unwrap_model_for_generation

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


class MultiTurnGRPOTrainer(GRPOTrainer):
    """
    覆写 _generate_and_score_completions，实现多轮工具交互。

    核心改动：
    - 生成阶段：逐条 prompt 循环生成，检测 <tool_call>，调用 ToolEnv.step()，
      拼回 tool_response 继续生成，直到没有 tool_call 或达到 max_turns
    - 评分阶段：用自定义 reward_fn（format_score + answer_score）
    - 返回格式：与原版 _generate_and_score_completions 完全一致
    """

    def __init__(self, *args, env_factory=None, prompt_env_map=None,
                 max_turns=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_factory = env_factory
        self.prompt_env_map = prompt_env_map or {}
        self.max_turns = max_turns

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """
        多轮工具交互版本的生成与评分。

        与原版保持相同的返回格式：
        {
            "prompt_ids", "prompt_mask",
            "completion_ids", "completion_mask",
            "old_per_token_logps", "ref_per_token_logps",
            "advantages"
        }
        """
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]

        # tokenize prompts（与原版一致）
        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True,
            padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super(GRPOTrainer, self)._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        # ==================== 多轮生成 ====================
        # inputs 里同一个 prompt 会重复 num_generations 次（TRL sampler 机制）
        # 我们需要对每条都独立做 rollout

        all_completion_ids = []
        all_improvements = []

        with unwrap_model_for_generation(
            self.model_wrapped, self.accelerator,
            gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:

            for batch_idx in range(len(prompts)):
                prompt = prompts[batch_idx]
                prompt_text = prompts_text[batch_idx]
                single_prompt_ids = prompt_ids[batch_idx:batch_idx+1]

                # 提取 user question 找对应 env_sample_idx
                if isinstance(prompt, list):
                    user_msg = next(
                        (m["content"] for m in prompt if m["role"] == "user"), ""
                    )
                else:
                    user_msg = str(prompt)
                sample_idx = self.prompt_env_map.get(user_msg)

                # 创建并 reset 环境
                env = self.env_factory()
                if sample_idx is not None:
                    env.reset(sample_idx=sample_idx)
                else:
                    env.reset()

                # 多轮生成循环
                current_ids = single_prompt_ids  # (1, prompt_len)
                completion_tokens = []  # 收集所有 completion token ids

                for turn in range(self.max_turns):
                    remaining = self.max_completion_length - len(completion_tokens)
                    if remaining <= 0:
                        break

                    with torch.no_grad():
                        outputs = unwrapped_model.generate(
                            input_ids=current_ids,
                            attention_mask=torch.ones_like(current_ids),
                            generation_config=self.generation_config,
                            max_new_tokens=min(remaining, self.max_completion_length),
                        )

                    new_ids = outputs[0, current_ids.shape[1]:]  # 生成的新 token
                    new_text = self.processing_class.decode(
                        new_ids, skip_special_tokens=False
                    )

                    if "</tool_call>" in new_text:
                        # 截断到第一个 </tool_call>
                        cut_text = new_text.split("</tool_call>")[0] + "</tool_call>"
                        cut_ids = self.processing_class.encode(
                            cut_text, add_special_tokens=False
                        )
                        completion_tokens.extend(cut_ids)

                        # 执行工具
                        obs, reward, done, info = env.step(cut_text)

                        # 拼接 tool response
                        tool_resp_text = TOOL_RESPONSE_TEMPLATE.format(
                            tool_response=obs
                        )
                        tool_resp_ids = self.processing_class.encode(
                            tool_resp_text, add_special_tokens=False
                        )
                        completion_tokens.extend(tool_resp_ids)

                        # 更新 current_ids 用于下一轮
                        all_ids = list(single_prompt_ids[0].cpu().numpy()) + completion_tokens
                        current_ids = torch.tensor(
                            [all_ids], dtype=torch.long, device=device
                        )

                        if done:
                            break
                    else:
                        # 无 tool_call，正常结束
                        completion_tokens.extend(new_ids.tolist())
                        break

                # 截断到 max_completion_length
                if len(completion_tokens) > self.max_completion_length:
                    completion_tokens = completion_tokens[:self.max_completion_length]

                all_completion_ids.append(
                    torch.tensor(completion_tokens, dtype=torch.long, device=device)
                )

                # 获取 improvement
                improvement = getattr(env, 'improvement_pct', 0.0)
                if improvement == 0.0:
                    try:
                        predict_text = '<tool_call>\n{"name": "predict_performance", "arguments": {}}\n</tool_call>'
                        obs, _, _, _ = env.step(predict_text)
                        parsed = json.loads(obs) if isinstance(obs, str) else obs
                        improvement = float(parsed.get("improvement_pct", 0.0))
                    except Exception:
                        pass
                all_improvements.append(improvement)

        # ==================== Pad completions ====================
        completion_ids = pad(
            all_completion_ids,
            padding_value=self.processing_class.pad_token_id
        )

        # Mask: 非 pad 部分
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(
            is_eos.size(1), device=device
        ).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        # 也要 mask 掉 pad
        completion_mask = completion_mask * (
            completion_ids != self.processing_class.pad_token_id
        ).int()

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # ==================== 计算 logprobs ====================
        logits_to_keep = completion_ids.size(1)

        with torch.no_grad():
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # ==================== 评分 ====================
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        # 用自定义 reward 计算
        rewards_list = []
        for i, text in enumerate(completions_text):
            format_score = compute_score_format(text)
            imp = all_improvements[i] / 100.0 if i < len(all_improvements) else 0.0
            imp = min(2.0, max(0.0, imp))
            answer_score = math.log(1 + imp) if imp > 0 else 0.0
            rewards_list.append(format_score + answer_score)

        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)

        # 需要 gather 用于 group normalize
        rewards = gather(rewards)

        # Grouped rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        advantages = rewards - mean_grouped_rewards
        if self.args.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # 切片回当前进程
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # ==================== Metrics ====================
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            self._total_train_tokens += (
                self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
            )
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]
        completion_length = (
            self.accelerator.gather_for_metrics(completion_mask.sum(1))
            .float().mean().item()
        )
        self._metrics[mode]["completion_length"].append(completion_length)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        # improvement 统计
        avg_imp = sum(all_improvements) / max(len(all_improvements), 1)
        self._metrics[mode]["avg_improvement_pct"].append(avg_imp)

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }


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
    prompt_records = []
    prompt_env_map = {}
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
    # 注意：per_device_train_batch_size 必须能被 num_generations 整除
    # batch_size=4, num_generations=4 → 每个 step 1 个独立 prompt × 4 generations
    effective_batch = args.batch_size * args.num_generations
    training_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=effective_batch,
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
        # GRPO 特有
        beta=0.0,  # 不需要 ref model（用 LoRA disable_adapter 替代）
        scale_rewards=True,
    )

    # 环境工厂
    from environment.tools import DBToolEnv

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

    # 用自定义的 reward_fn 作为占位（实际评分在 _generate_and_score_completions 里做）
    def dummy_reward_fn(prompts, completions, **kwargs):
        return [0.0] * len(completions)

    # Trainer
    trainer = MultiTurnGRPOTrainer(
        model=model,
        reward_funcs=dummy_reward_fn,
        args=training_config,
        processing_class=tokenizer,
        train_dataset=dataset,
        peft_config=peft_config,
        env_factory=env_factory,
        prompt_env_map=prompt_env_map,
        max_turns=args.max_turns,
    )

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
