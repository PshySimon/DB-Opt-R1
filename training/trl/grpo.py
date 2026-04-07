"""
trl GRPO 多轮工具调用训练（vLLM 加速版）

架构：
- 生成：通过 vLLM OpenAI API 并发多轮 rollout（快）
- 训练：HF 模型 forward + backward（LoRA）
- 生成策略固定为 SFT 模型，训练策略实时更新

Usage:
    # 1. 启动 vLLM（限制显存给训练留空间）
    GPU_UTIL=0.25 bash scripts/serve_vllm.sh

    # 2. 启动训练
    PYTHONPATH=. python -m training.trl.grpo \\
        --model_path model_save/sft_qwen3_4b_cleaned_merged \\
        --train_data data_pipeline/data/train/sft_trajectories.jsonl \\
        --scenario_files \\
            data_pipeline/data/scenarios/collected/collected_server1.json \\
            data_pipeline/data/scenarios/collected/collected_server2.json \\
            data_pipeline/data/scenarios/collected/collected_server3.json \\
        --cost_model cost_model/checkpoints/v9_lgbm \\
        --vllm_base_url http://localhost:8000/v1 \\
        --vllm_model qwen3-4b-sft \\
        --output_dir model_save/grpo/
"""

import json
import math
import argparse
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Union

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from trl.trainer.utils import pad, selective_log_softmax
from trl.data_utils import maybe_apply_chat_template, is_conversational
from peft import LoraConfig
from accelerate.utils import gather
from trl.models import unwrap_model_for_generation

from training.data_utils import SYSTEM_PROMPT
from training.reward_score import compute_score_format

logger = logging.getLogger(__name__)


class MultiTurnGRPOTrainer(GRPOTrainer):
    """vLLM 加速的多轮工具交互 GRPO Trainer"""

    def __init__(self, *args, env_factory=None, prompt_env_map=None,
                 max_turns=10, vllm_base_url=None, vllm_model_name=None,
                 rollout_concurrency=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_factory = env_factory
        self.prompt_env_map = prompt_env_map or {}
        self.max_turns = max_turns
        self.vllm_model_name = vllm_model_name
        self.rollout_concurrency = rollout_concurrency

        from openai import OpenAI
        self.vllm_api = OpenAI(api_key="dummy", base_url=vllm_base_url)
        logger.info(f"vLLM API: {vllm_base_url}, model: {vllm_model_name}")

    def _do_single_rollout(self, prompt):
        """单条多轮 rollout（通过 vLLM API）"""
        if isinstance(prompt, list):
            user_msg = next((m["content"] for m in prompt if m["role"] == "user"), "")
        else:
            user_msg = str(prompt)

        sample_idx = self.prompt_env_map.get(user_msg)
        env = self.env_factory()
        env.reset(sample_idx=sample_idx) if sample_idx is not None else env.reset()

        tools_desc = env.tools_format_func()
        full_system = f"{SYSTEM_PROMPT}\n\n{tools_desc}"
        prompt_msgs = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": user_msg},
        ]
        messages = list(prompt_msgs)

        for turn in range(self.max_turns):
            text = ""
            for retry in range(3):
                try:
                    resp = self.vllm_api.chat.completions.create(
                        model=self.vllm_model_name,
                        messages=messages,
                        temperature=1.0,
                        max_tokens=1024,
                    )
                    text = resp.choices[0].message.content or ""
                    if not text:
                        text = getattr(resp.choices[0].message, 'reasoning_content', None) or ""
                    break
                except Exception as e:
                    logger.warning(f"vLLM 生成失败 (retry {retry+1}/3): {e}")
                    import time; time.sleep(5)
            if not text:
                break

            if "</tool_call>" in text:
                text = text.split("</tool_call>")[0] + "</tool_call>"

            messages.append({"role": "assistant", "content": text})

            if "</tool_call>" in text:
                obs, _, done, _ = env.step(text)
                messages.append({"role": "user", "content": f"<tool_response>\n{obs}\n</tool_response>"})
                if done:
                    break
            else:
                break

        # 获取 improvement
        improvement = getattr(env, 'improvement_pct', 0.0)
        if improvement == 0.0:
            try:
                obs, _, _, _ = env.step('<tool_call>\n{"name": "predict_performance", "arguments": {}}\n</tool_call>')
                improvement = float(json.loads(obs).get("improvement_pct", 0.0))
            except Exception:
                pass

        return prompt_msgs, messages, improvement

    def _generate_and_score_completions(self, inputs):
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]

        # ==================== 并发 rollout（vLLM） ====================
        results = [None] * len(prompts)
        workers = min(len(prompts), self.rollout_concurrency)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(self._do_single_rollout, p): i for i, p in enumerate(prompts)}
            for fut in as_completed(futs):
                idx = futs[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    logger.error(f"Rollout {idx} 异常: {e}")
                    dummy_msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                                  {"role": "user", "content": ""},
                                  {"role": "assistant", "content": ""}]
                    results[idx] = (dummy_msgs[:2], dummy_msgs, 0.0)

        # ==================== Tokenize ====================
        all_prompt_texts = []
        all_completion_ids = []
        all_improvements = []

        for prompt_msgs, full_msgs, improvement in results:
            prompt_text = self.processing_class.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True
            )
            full_text = self.processing_class.apply_chat_template(
                full_msgs, tokenize=False, add_generation_prompt=False
            )
            all_prompt_texts.append(prompt_text)
            all_improvements.append(improvement)

            # 提取 completion ids
            prompt_token_len = len(self.processing_class.encode(prompt_text, add_special_tokens=False))
            full_ids = self.processing_class.encode(full_text, add_special_tokens=False)
            comp_ids = full_ids[prompt_token_len:]

            if len(comp_ids) > self.max_completion_length:
                comp_ids = comp_ids[:self.max_completion_length]
            # 兜底：至少有一个 eos token，防止空 tensor 导致 argmax 崩溃
            if len(comp_ids) == 0:
                comp_ids = [self.processing_class.eos_token_id]

            all_completion_ids.append(torch.tensor(comp_ids, dtype=torch.long, device=device))

        # Prompt tokenize（left pad）
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
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        seq_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = ((seq_indices <= eos_idx.unsqueeze(1)) &
                           (completion_ids != self.processing_class.pad_token_id)).int()

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # ==================== Logprobs（HF 模型） ====================
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
            imp = min(2.0, max(0.0, all_improvements[i] / 100.0))
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

        # ==================== Metrics ====================
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
            sum(all_improvements) / max(len(all_improvements), 1)
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
    parser.add_argument("--vllm_base_url", default="http://localhost:8000/v1")
    parser.add_argument("--vllm_model", default="qwen3-4b-sft")
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
    parser.add_argument("--rollout_concurrency", type=int, default=8)
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

    # 模型
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
        vllm_base_url=args.vllm_base_url, vllm_model_name=args.vllm_model,
        rollout_concurrency=args.rollout_concurrency,
    )

    if torch.cuda.is_available():
        tp = sum(p.numel() for p in trainer.model.parameters())
        trp = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        print(f"\n  总参数: {tp/1e6:.1f}M | 可训练: {trp/1e6:.1f}M ({trp/tp:.2%})")
        torch.cuda.reset_peak_memory_stats()

    print("\n🚀 开始 GRPO 训练（vLLM 加速）...")
    trainer.train()

    if torch.cuda.is_available():
        print(f"📊 峰值显存: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

    trainer.save_model(args.output_dir)
    print(f"✅ 保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
