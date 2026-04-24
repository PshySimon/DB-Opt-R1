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
import random
import torch
from datasets import Dataset
from packaging.version import Version
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import trl
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

from training.data_utils import load_sft_data


MIN_TRANSFORMERS_FOR_ASSISTANT_MASKS = Version("4.56.2")
TRL_CHAT_TEMPLATES_DIR = (
    Path(trl.__file__).resolve().parent / "chat_templates"
    if getattr(trl, "__file__", None)
    else Path("chat_templates")
)


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
    parser.add_argument("--train_ratio", type=float, default=0.95)
    parser.add_argument("--eval_data_files", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--save_config_path", default=None)
    parser.add_argument("--chat_template_path", default=None)
    parser.add_argument("--deepspeed", default=None, help="DeepSpeed 配置 JSON 路径或 JSON 字符串")
    parser.add_argument("--fsdp", default=None, help="Transformers TrainingArguments 的 FSDP 策略字符串")
    parser.add_argument("--fsdp_config", default=None, help="FSDP 配置 JSON 路径或 JSON 字符串")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--use_lora", nargs=0, action=_SetTrainMode, const="lora")
    mode_group.add_argument("--full_finetune", nargs=0, action=_SetTrainMode, const="full")
    parser.set_defaults(use_lora=True, full_finetune=False)
    return parser


def validate_distributed_backend_args(args) -> None:
    deepspeed = getattr(args, "deepspeed", None)
    fsdp = getattr(args, "fsdp", None)
    fsdp_config = getattr(args, "fsdp_config", None)

    if deepspeed and (fsdp or fsdp_config):
        raise ValueError("不能同时启用 DeepSpeed 和 FSDP，请只设置其中一种分布式优化后端。")
    if fsdp_config and not fsdp:
        raise ValueError("--fsdp_config 需要同时设置 --fsdp。")


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


def validate_transformers_version_for_assistant_masks():
    current = Version(transformers.__version__)
    if current < MIN_TRANSFORMERS_FOR_ASSISTANT_MASKS:
        raise RuntimeError(
            "当前 TRL SFT 的 assistant-only loss 依赖 "
            f"transformers>={MIN_TRANSFORMERS_FOR_ASSISTANT_MASKS}，"
            f"但当前环境是 {transformers.__version__}。"
        )


def resolve_training_chat_template_path(model_path: str, explicit_path: str | None = None) -> str | None:
    if explicit_path:
        return explicit_path

    lowered = str(model_path).lower()
    candidate_names = []
    if "qwen3" in lowered:
        candidate_names.append("qwen3_training.jinja")
    elif any(tag in lowered for tag in ("qwen2.5", "qwen2_5", "qwen2-5")):
        candidate_names.append("qwen2_5_training.jinja")

    for candidate_name in candidate_names:
        candidate_path = TRL_CHAT_TEMPLATES_DIR / candidate_name
        if candidate_path.exists():
            return str(candidate_path)
    return None


def maybe_apply_training_chat_template(args, tokenizer) -> str | None:
    chat_template_path = resolve_training_chat_template_path(
        model_path=args.model_path,
        explicit_path=args.chat_template_path,
    )
    if chat_template_path:
        tokenizer.chat_template = Path(chat_template_path).read_text(encoding="utf-8")
        args.chat_template_path = chat_template_path
    return chat_template_path


def extract_chat_template_kwargs(record: dict) -> dict:
    kwargs = dict(record.get("chat_template_kwargs") or {})
    enable_thinking = record.get("enable_thinking")
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return kwargs


def extract_tools(record: dict):
    tools = record.get("tools")
    return json.loads(tools) if isinstance(tools, str) else tools


def validate_assistant_mask_support(records, tokenizer):
    sample = next(
        (
            record
            for record in records
            if any(message.get("role") == "assistant" for message in record.get("messages", []))
        ),
        None,
    )
    if sample is None:
        return

    processed = tokenizer.apply_chat_template(
        sample["messages"],
        tools=extract_tools(sample),
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        **extract_chat_template_kwargs(sample),
    )
    assistant_masks = processed.get("assistant_masks")
    mask_values = assistant_masks.tolist() if hasattr(assistant_masks, "tolist") else assistant_masks
    if not mask_values or 1 not in mask_values:
        raise RuntimeError(
            "TRL assistant-only loss 的 assistant mask 预检查失败：当前 tokenizer/chat template "
            "没有为 assistant token 生成有效 mask。请确认 transformers 版本满足要求，并使用带 "
            "`{% generation %}` 的训练模板。"
        )


def filter_records_by_max_length(records, tokenizer, max_length: int):
    filtered = []
    for record in records:
        msgs = record.get("messages", [])
        text = tokenizer.apply_chat_template(msgs, tokenize=False)
        n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        if n_tokens <= max_length:
            filtered.append(record)
    return filtered


def split_train_eval_records(records, train_ratio: float, seed: int):
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio 必须在 (0, 1) 区间内，当前为 {train_ratio}")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    train_records = shuffled[:split_idx]
    eval_records = shuffled[split_idx:]

    if not eval_records and len(train_records) > 1:
        eval_records = [train_records.pop()]

    return train_records, eval_records


def build_sft_config_kwargs(args, has_eval: bool):
    kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "warmup_ratio": 0.1,
        "max_length": args.max_length,
        "bf16": args.bf16,
        "fp16": not args.bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "logging_steps": 5,
        "save_steps": 50,
        "save_strategy": "steps",
        "save_total_limit": 3,
        "report_to": "none",
        "max_steps": args.max_steps,
        "seed": args.seed,
        "assistant_only_loss": True,
        "ddp_find_unused_parameters": False if args.use_lora else None,
    }
    chat_template_path = getattr(args, "chat_template_path", None)
    if chat_template_path:
        kwargs["chat_template_path"] = chat_template_path
    deepspeed = getattr(args, "deepspeed", None)
    if deepspeed:
        kwargs["deepspeed"] = deepspeed
    fsdp = getattr(args, "fsdp", None)
    if fsdp:
        kwargs["fsdp"] = fsdp
    fsdp_config = getattr(args, "fsdp_config", None)
    if fsdp_config:
        kwargs["fsdp_config"] = fsdp_config
    if has_eval:
        kwargs.update(
            {
                "per_device_eval_batch_size": args.batch_size,
                "eval_strategy": "steps",
                "eval_steps": 50,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
            }
        )
    return kwargs


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_distributed_backend_args(args)

    maybe_configure_torch_device_for_distributed()
    validate_transformers_version_for_assistant_masks()

    # 先加载 tokenizer（用于筛选超长轨迹）
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    chat_template_path = maybe_apply_training_chat_template(args, tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.flash_attn:
        tokenizer.padding_side = "left"

    # 加载数据
    records = load_sft_data(args.data_files)
    train_before_filter = len(records)
    records = filter_records_by_max_length(records, tokenizer, args.max_length)
    if len(records) < train_before_filter:
        print(f"过滤超长轨迹: {train_before_filter} → {len(records)} 条 (max_length={args.max_length})")

    if args.eval_data_files:
        eval_records = load_sft_data(args.eval_data_files)
        eval_before_filter = len(eval_records)
        eval_records = filter_records_by_max_length(eval_records, tokenizer, args.max_length)
        if len(eval_records) < eval_before_filter:
            print(f"过滤超长验证轨迹: {eval_before_filter} → {len(eval_records)} 条 (max_length={args.max_length})")
        train_records = records
    else:
        eval_before_filter = 0
        train_records, eval_records = split_train_eval_records(records, args.train_ratio, args.seed)

    validate_assistant_mask_support(train_records, tokenizer)

    save_training_config(
        args,
        {
            "records_before_filter": train_before_filter,
            "records_after_filter": len(records),
            "train_records": len(train_records),
            "eval_records": len(eval_records),
            "eval_records_before_filter": eval_before_filter,
            "train_mode": "lora" if args.use_lora else "full",
            "chat_template_path": chat_template_path,
            "world_size": int(torch.distributed.get_world_size())
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 1,
        },
    )

    dataset = Dataset.from_list(train_records)
    eval_dataset = Dataset.from_list(eval_records) if eval_records else None
    training_args = SFTConfig(**build_sft_config_kwargs(args, has_eval=eval_dataset is not None))

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

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
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
