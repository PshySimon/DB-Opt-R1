"""
trl SFT 训练入口

直接加载 JSONL messages 格式，使用 trl.SFTTrainer + LoRA 训练。

Usage:
    python -m training.trl.sft \
        --model_path ~/models/Qwen2.5-3B-Instruct \
        --data_files datasets/sft/cold_start.jsonl \
        --output_dir model_save/sft/
"""

import argparse
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

from training.data_utils import load_sft_data


def main():
    parser = argparse.ArgumentParser(description="trl SFT 训练")
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--data_files", nargs="+", required=True)
    parser.add_argument("--output_dir", default="./model_save/sft/")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)  # 改这里或通过命令行传入
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--flash_attn", action="store_true", default=False)
    args = parser.parse_args()

    # 加载数据
    records = load_sft_data(args.data_files)
    dataset = Dataset.from_list(records)

    # 加载模型
    print(f"加载模型: {args.model_path}")
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
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        max_seq_length=args.max_length,
        bf16=args.bf16,
        fp16=not args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=5,
        save_steps=50,
        save_total_limit=3,
        report_to="none",
        max_steps=args.max_steps,
        seed=42,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("开始 SFT 训练...")
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"✅ 训练完成，模型保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
