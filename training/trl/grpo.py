"""
trl GRPO 训练入口

单轮生成模式：模型一次性生成完整调优轨迹，
reward 函数从生成文本中提取 knob，通过 Cost Model 评分。

Usage:
    python -m training.trl.grpo \
        --model_path model_save/sft/checkpoint-xxx \
        --scenario_dir data_pipeline/data/scenarios/ \
        --cost_model cost_model/checkpoints/v1 \
        --output_dir model_save/grpo/
"""

import argparse
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

from training.data_utils import load_grpo_prompts
from training.reward_score import (
    compute_score_format,
    compute_score_answer,
)


def make_reward_fn(cost_model=None):
    """创建 reward 函数（闭包捕获 cost_model）"""

    def reward_fn(completions, ground_truth, **kwargs):
        rewards = []
        for completion, gt in zip(completions, ground_truth):
            content = (
                completion[0]["content"]
                if isinstance(completion, list)
                else completion
            )
            gt_dict = gt if isinstance(gt, dict) else {}

            format_score = compute_score_format(content)
            answer_score = compute_score_answer(
                content, gt_dict, cost_model=cost_model
            )
            rewards.append(format_score + answer_score)

        return rewards

    return reward_fn


def main():
    parser = argparse.ArgumentParser(description="trl GRPO 训练")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--scenario_dir", required=True)
    parser.add_argument("--cost_model", default=None)
    parser.add_argument("--output_dir", default="./model_save/grpo/")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--max_completion_length", type=int, default=2048)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--bf16", action="store_true", default=True)
    args = parser.parse_args()

    # 加载 Cost Model
    cost_model = None
    if args.cost_model:
        try:
            from cost_model.model import CostModel
            cost_model = CostModel.load(args.cost_model)
            print(f"加载 Cost Model: {args.cost_model}")
        except Exception as e:
            print(f"[WARNING] Cost Model 加载失败: {e}，answer_score 将为 0")

    # 加载 prompt 数据
    prompt_data = load_grpo_prompts(args.scenario_dir)
    dataset = Dataset.from_list(prompt_data)

    # 加载模型
    print(f"加载模型: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    # 训练配置
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
        logging_steps=5,
        save_steps=50,
        save_total_limit=3,
        report_to="none",
        max_steps=args.max_steps,
        seed=42,
    )

    # Trainer
    trainer = GRPOTrainer(
        args=training_config,
        model=model,
        processing_class=tokenizer,
        reward_funcs=make_reward_fn(cost_model),
        train_dataset=dataset,
        peft_config=peft_config,
    )

    print("开始 GRPO 训练...")
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"✅ 训练完成，模型保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
