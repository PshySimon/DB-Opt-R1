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
import json
import os
from pathlib import Path
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

from training.data_utils import load_sft_data


class _SetTrainMode(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        use_lora = self.const == "lora"
        setattr(namespace, "use_lora", use_lora)
        setattr(namespace, "full_finetune", not use_lora)


def build_parser():
    parser = argparse.ArgumentParser(description="trl SFT 训练")
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--data_files", nargs="+", required=True)
    parser.add_argument("--output_dir", default="./model_save/sft/")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--save_config_path", default=None)

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--use_lora", nargs=0, action=_SetTrainMode, const="lora")
    mode_group.add_argument("--full_finetune", nargs=0, action=_SetTrainMode, const="full")
    parser.set_defaults(use_lora=True, full_finetune=False)
    return parser


def is_rank_zero() -> bool:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def save_training_config(args, extra: dict) -> None:
    if not args.save_config_path or not is_rank_zero():
        return

    payload = vars(args).copy()
    payload.update(extra)
    path = Path(args.save_config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def maybe_configure_torch_device_for_distributed():
    """Bind each torchrun worker to its local device before model init.

    Flash Attention 2 availability checks are device-sensitive on ROCm/HIP.
    Under torchrun, leaving the current device on the default rank-0 device can
    cause some workers to incorrectly fail the dispatch check even though the
    visible device set is valid.
    """
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None or not torch.cuda.is_available():
        return None

    device_index = int(local_rank)
    if device_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={device_index} 超出可见 GPU 范围 (device_count={torch.cuda.device_count()})"
        )

    torch.cuda.set_device(device_index)
    return device_index


def main():
    parser = build_parser()
    args = parser.parse_args()

    maybe_configure_torch_device_for_distributed()

    # 先加载 tokenizer（用于筛选超长轨迹）
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.flash_attn:
        tokenizer.padding_side = "left"

    # 加载数据
    records = load_sft_data(args.data_files)

    # 按 token 数过滤超长轨迹（截断的轨迹不完整，直接丢弃）
    before = len(records)
    filtered = []
    for r in records:
        msgs = r.get("messages", [])
        text = tokenizer.apply_chat_template(msgs, tokenize=False)
        n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        if n_tokens <= args.max_length:
            filtered.append(r)
    records = filtered
    if len(records) < before:
        print(f"过滤超长轨迹: {before} → {len(records)} 条 (max_length={args.max_length})")

    save_training_config(
        args,
        {
            "records_before_filter": before,
            "records_after_filter": len(records),
            "train_mode": "lora" if args.use_lora else "full",
            "world_size": int(torch.distributed.get_world_size())
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 1,
        },
    )

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


    peft_config = None
    if args.use_lora:
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
        ddp_find_unused_parameters=False if args.use_lora else None,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # 显存 profiling
    def print_memory_stats(tag: str):
        if not torch.cuda.is_available():
            return
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\n{'='*60}")
        print(f"  显存统计 [{tag}]")
        print(f"{'='*60}")
        print(f"  Allocated:  {alloc:.2f} GB")
        print(f"  Reserved:   {reserved:.2f} GB")
        print(f"  Peak:       {peak:.2f} GB")
        print(f"{'='*60}")

    def print_param_stats(model):
        total, trainable = 0, 0
        frozen_bytes, trainable_bytes = 0, 0
        for p in model.parameters():
            n = p.numel()
            total += n
            b = n * p.element_size()
            if p.requires_grad:
                trainable += n
                trainable_bytes += b
            else:
                frozen_bytes += b

        # 可训练参数的优化器开销: master(fp32) + m(fp32) + v(fp32) = 12 bytes/param
        optim_bytes = trainable * 12
        grad_bytes = trainable * 2  # bf16 梯度

        print(f"\n{'='*60}")
        print(f"  参数 & 显存拆解（理论值）")
        print(f"{'='*60}")
        print(f"  总参数:           {total/1e6:>10.1f} M")
        print(f"  可训练参数:       {trainable/1e6:>10.1f} M  ({trainable/total:.2%})")
        print(f"{'─'*60}")
        print(f"  冻结模型 (bf16):  {frozen_bytes/1024**3:>10.2f} GB")
        print(f"  可训练参数 (bf16):{trainable_bytes/1024**3:>10.2f} GB")
        print(f"  梯度 (bf16):      {grad_bytes/1024**3:>10.2f} GB")
        print(f"  优化器状态 (fp32):{optim_bytes/1024**3:>10.2f} GB")
        print(f"{'─'*60}")
        subtotal = frozen_bytes + trainable_bytes + grad_bytes + optim_bytes
        print(f"  小计 (不含激活):  {subtotal/1024**3:>10.2f} GB")
        print(f"{'='*60}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        print_param_stats(trainer.model)
        print_memory_stats("训练前")

    mode_name = "LoRA" if args.use_lora else "全量"
    print(f"\n开始 SFT 训练... 模式: {mode_name}")
    trainer.train()

    if torch.cuda.is_available():
        print_memory_stats("训练后")
        pre_alloc = sum(p.numel() * p.element_size() for p in trainer.model.parameters()) / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\n  → 激活值 + 其他开销 ≈ {peak - pre_alloc:.2f} GB (峰值 - 参数)")
        print(f"\n{torch.cuda.memory_summary()}")

    trainer.save_model(args.output_dir)
    print(f"✅ 训练完成，模型保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
