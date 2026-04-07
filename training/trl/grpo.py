"""
trl GRPO 多轮工具调用训练（vLLM 进程内加速）

架构：
- 生成：vllm.LLM 进程内批量生成（快，无需启动 server）
- 训练：HF 模型 forward + backward（LoRA）
- 两个模型共享 GPU 显存（vLLM 占 30%，训练占 70%）

Usage:
    PYTHONPATH=. python -m training.trl.grpo \\
        --model_path model_save/sft_qwen3_4b_cleaned_merged \\
        --train_data data_pipeline/data/train/sft_trajectories.jsonl \\
        --scenario_files ... \\
        --cost_model cost_model/checkpoints/v9_lgbm \\
        --output_dir model_save/grpo/
"""

import json
import math
import time
import argparse
import logging
from collections import defaultdict
from typing import Any, Union

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from trl.trainer.utils import pad, selective_log_softmax
from trl.data_utils import maybe_apply_chat_template, is_conversational
from peft import LoraConfig
from accelerate.utils import gather

from training.data_utils import SYSTEM_PROMPT
from training.reward_score import compute_score_format

logger = logging.getLogger(__name__)


class MultiTurnGRPOTrainer(GRPOTrainer):
    """vLLM 进程内加速的多轮工具交互 GRPO Trainer"""

    def __init__(self, *args, env_factory=None, prompt_env_map=None,
                 max_turns=10, vllm_llm=None, vllm_tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_factory = env_factory
        self.prompt_env_map = prompt_env_map or {}
        self.max_turns = max_turns
        self.vllm_llm = vllm_llm
        self.vllm_tokenizer = vllm_tokenizer

    def _do_rollouts_batched(self, prompts):
        """
        批量多轮 rollout：每轮把所有活跃 rollout 一起交给 vLLM 批量生成。
        比逐条串行快 N 倍（N = num_generations）。
        """
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=1.0, top_p=1.0, max_tokens=1024,
            stop=["</tool_call>"], include_stop_str_in_output=True,
        )

        n = len(prompts)
        # 每条 rollout 的状态
        envs = []
        prompt_msgs_list = []  # 用于 tokenize 的 prompt
        messages_list = []     # 完整对话
        improvements = [0.0] * n
        active = list(range(n))  # 当前活跃的 rollout 索引

        for i in range(n):
            prompt = prompts[i]
            if isinstance(prompt, list):
                user_msg = next((m["content"] for m in prompt if m["role"] == "user"), "")
            else:
                user_msg = str(prompt)

            sample_idx = self.prompt_env_map.get(user_msg)
            env = self.env_factory()
            env.reset(sample_idx=sample_idx) if sample_idx is not None else env.reset()
            envs.append(env)

            tools_desc = env.tools_format_func()
            full_system = f"{SYSTEM_PROMPT}\n\n{tools_desc}"
            pm = [{"role": "system", "content": full_system},
                  {"role": "user", "content": user_msg}]
            prompt_msgs_list.append(pm)
            messages_list.append(list(pm))

        # 多轮循环
        for turn in range(self.max_turns):
            if not active:
                break

            # 把活跃 rollout 的 messages 格式化成 prompt 文本
            batch_texts = []
            for i in active:
                text = self.vllm_tokenizer.apply_chat_template(
                    messages_list[i], tokenize=False, add_generation_prompt=True
                )
                batch_texts.append(text)

            # vLLM 批量生成
            outputs = self.vllm_llm.generate(batch_texts, sampling_params)

            # 处理每条输出
            still_active = []
            for j, i in enumerate(active):
                gen_text = outputs[j].outputs[0].text or ""

                # 补全 stop token（include_stop_str_in_output 有时不可靠）
                if "<tool_call>" in gen_text and "</tool_call>" not in gen_text:
                    gen_text += "</tool_call>"

                messages_list[i].append({"role": "assistant", "content": gen_text})

                if "</tool_call>" in gen_text:
                    obs, _, done, _ = envs[i].step(gen_text)
                    messages_list[i].append({
                        "role": "user",
                        "content": f"<tool_response>\n{obs}\n</tool_response>"
                    })
                    if not done:
                        still_active.append(i)
                    # done 了就不再继续
                # else: 没有 tool_call，结束

            active = still_active

        # 获取 improvement
        for i in range(n):
            imp = getattr(envs[i], 'improvement_pct', 0.0)
            if imp == 0.0:
                try:
                    obs, _, _, _ = envs[i].step(
                        '<tool_call>\n{"name": "predict_performance", "arguments": {}}\n</tool_call>'
                    )
                    imp = float(json.loads(obs).get("improvement_pct", 0.0))
                except Exception:
                    pass
            improvements[i] = imp

        return prompt_msgs_list, messages_list, improvements

    def _generate_and_score_completions(self, inputs):
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]

        # ==================== 批量多轮 rollout ====================
        prompt_msgs_list, messages_list, improvements = self._do_rollouts_batched(prompts)

        # ==================== Tokenize ====================
        all_prompt_texts = []
        all_completion_ids = []

        for i in range(len(prompts)):
            prompt_text = self.vllm_tokenizer.apply_chat_template(
                prompt_msgs_list[i], tokenize=False, add_generation_prompt=True
            )
            full_text = self.vllm_tokenizer.apply_chat_template(
                messages_list[i], tokenize=False, add_generation_prompt=False
            )
            all_prompt_texts.append(prompt_text)

            # 提取 completion token ids
            prompt_len = len(self.vllm_tokenizer.encode(prompt_text, add_special_tokens=False))
            full_ids = self.vllm_tokenizer.encode(full_text, add_special_tokens=False)
            comp_ids = full_ids[prompt_len:]

            if len(comp_ids) > self.max_completion_length:
                comp_ids = comp_ids[:self.max_completion_length]
            if len(comp_ids) == 0:
                comp_ids = [self.vllm_tokenizer.eos_token_id]

            all_completion_ids.append(torch.tensor(comp_ids, dtype=torch.long, device=device))

        # Prompt（left pad）
        prompt_inputs = self.processing_class(
            text=all_prompt_texts, return_tensors="pt", padding=True,
            padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super(GRPOTrainer, self)._prepare_inputs(prompt_inputs)
        prompt_ids = prompt_inputs["input_ids"]
        prompt_mask = prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        # Pad completions
        completion_ids = pad(all_completion_ids, padding_value=self.processing_class.pad_token_id)

        # Completion mask
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        if is_eos.any(dim=1).any():
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        seq_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = ((seq_indices <= eos_idx.unsqueeze(1)) &
                           (completion_ids != self.processing_class.pad_token_id)).int()

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # ==================== Logprobs（HF 模型）====================
        logits_to_keep = completion_ids.size(1)
        with torch.no_grad():
            old_per_token_logps = (
                self._get_per_token_logps(self.model, prompt_completion_ids, attention_mask, logits_to_keep)
                if self.num_iterations > 1 else None
            )
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

        # ==================== Reward ====================
        rewards_list = []
        for i in range(len(prompts)):
            text = self.processing_class.decode(all_completion_ids[i], skip_special_tokens=True)
            fmt = compute_score_format(text)
            imp = min(2.0, max(0.0, improvements[i] / 100.0))
            ans = math.log(1 + imp) if imp > 0 else 0.0
            rewards_list.append(fmt + ans)

        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)
        rewards = gather(rewards)

        mean_r = rewards.view(-1, self.num_generations).mean(dim=1)
        std_r = rewards.view(-1, self.num_generations).std(dim=1)
        mean_r = mean_r.repeat_interleave(self.num_generations, dim=0)
        std_r = std_r.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_r
        if self.args.scale_rewards:
            advantages = advantages / (std_r + 1e-4)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Metrics
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            self._total_train_tokens += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]
        self._metrics[mode]["completion_length"].append(
            self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        )
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_r.mean().item())
        self._metrics[mode]["avg_improvement_pct"].append(
            sum(improvements) / max(len(improvements), 1)
        )

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", default="./model_save/grpo/")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--scenario_files", nargs="+", required=True)
    parser.add_argument("--cost_model", default=None)
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
    parser.add_argument("--vllm_gpu_util", type=float, default=0.30,
                        help="vLLM 显存占比（剩余给训练）")
    args = parser.parse_args()

    # Cost Model
    cost_model = None
    if args.cost_model:
        try:
            from cost_model.model import CostModel
            cost_model = CostModel.load(args.cost_model)
            print(f"✅ Cost Model: {args.cost_model}")
        except Exception as e:
            print(f"[WARNING] Cost Model 加载失败: {e}")

    # ---- 先启动 vLLM（在 HF 模型之前，优先分配显存）----
    print(f"🔄 启动 vLLM（gpu_util={args.vllm_gpu_util}）: {args.model_path}")
    from vllm import LLM
    vllm_llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        gpu_memory_utilization=args.vllm_gpu_util,
        trust_remote_code=True,
        max_model_len=4096,
    )
    vllm_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    print("✅ vLLM 就绪")

    # 从 SFT 轨迹提取 prompt
    prompt_records, prompt_env_map, seen = [], {}, set()
    with open(args.train_data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            data = json.loads(line)
            q = data.get("question", "")
            idx = data.get("env_sample_idx")
            if not q or q in seen: continue
            seen.add(q)
            prompt_records.append({"question": q, "env_sample_idx": idx})
            if idx is not None:
                prompt_env_map[q] = idx

    dataset = Dataset.from_list([
        {"prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": r["question"]}]}
        for r in prompt_records
    ])
    print(f"📊 {len(dataset)} 条 prompt")

    # HF 模型（训练用，加载到剩余显存）
    print(f"🔄 加载 HF 模型（训练用）: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if args.flash_attn else "eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    peft_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank // 2, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    effective_batch = args.batch_size * args.num_generations
    training_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=effective_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, warmup_ratio=0.1,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        bf16=args.bf16, fp16=not args.bf16,
        gradient_checkpointing=True,
        logging_steps=1, save_steps=50, save_total_limit=3,
        report_to="none", max_steps=args.max_steps, seed=42,
        beta=0.0, scale_rewards=True,
    )

    # 环境工厂
    from environment.tools import DBToolEnv
    all_scenarios = DBToolEnv._load_scenarios(args.scenario_files)
    print(f"📦 {len(all_scenarios)} 个场景")

    def env_factory():
        env = DBToolEnv(mode="train", cost_model=cost_model,
                        max_turns=args.max_turns, knob_space_path=args.knob_space)
        env.scenarios = all_scenarios
        return env

    def dummy_reward(prompts, completions, **kw):
        return [0.0] * len(completions)

    trainer = MultiTurnGRPOTrainer(
        model=model, reward_funcs=dummy_reward, args=training_config,
        processing_class=tokenizer, train_dataset=dataset, peft_config=peft_config,
        env_factory=env_factory, prompt_env_map=prompt_env_map,
        max_turns=args.max_turns,
        vllm_llm=vllm_llm, vllm_tokenizer=vllm_tokenizer,
    )

    if torch.cuda.is_available():
        tp = sum(p.numel() for p in trainer.model.parameters())
        trp = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        print(f"\n  总参数: {tp/1e6:.1f}M | 可训练: {trp/1e6:.1f}M ({trp/tp:.2%})")

    print("\n🚀 开始 GRPO 训练（vLLM 进程内加速）...")
    trainer.train()

    if torch.cuda.is_available():
        print(f"📊 峰值显存: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

    trainer.save_model(args.output_dir)
    print(f"✅ 保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
