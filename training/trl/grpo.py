"""
trl GRPO 多轮工具调用训练入口

通过 TRL 原生 environment_factory 实现多轮 agent ↔ 环境交互训练。
模型在训练时真正与工具环境交互，reward 基于 predict_performance 的 TPS 提升。

Usage:
    python -m training.trl.grpo \
        --model_path model_save/sft/checkpoint-xxx \
        --scenario_dir data_pipeline/data/scenarios/collected/ \
        --cost_model cost_model/checkpoints/v9_lgbm \
        --output_dir model_save/grpo/

    # 指定场景文件
    python -m training.trl.grpo \
        --model_path model_save/sft/checkpoint-xxx \
        --scenario_files \
            data_pipeline/data/scenarios/collected/collected_server1.json \
            data_pipeline/data/scenarios/collected/collected_server2.json \
        --cost_model cost_model/checkpoints/v9_lgbm \
        --output_dir model_save/grpo/
"""

import argparse
import math
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

from training.data_utils import load_grpo_prompts, SYSTEM_PROMPT
from training.reward_score import compute_score_format


# ============================================================
# Reward 函数
# ============================================================

def reward_fn(completions, environments=None, **kwargs):
    """
    多轮交互 reward 函数。

    通过 environment_factory 传入的 env 实例获取 improvement_pct。

    总分 = format_score + answer_score
    - format_score: 检查 response 格式（think/tool_call 标签）
    - answer_score: env.improvement_pct 经 log 压缩

    Args:
        completions: 模型生成的 completion 列表
        environments: TRL 传入的 env 实例列表

    Returns:
        list[float]: 每个 completion 的 reward
    """
    rewards = []
    for i, completion in enumerate(completions):
        # 提取 completion 文本
        if isinstance(completion, list):
            # conversational format: list of dicts
            content = " ".join(
                m.get("content", "") for m in completion
                if m.get("role") == "assistant"
            )
        else:
            content = str(completion)

        # 格式分 (0 ~ 1.5)
        format_score = compute_score_format(content)

        # 任务分: 从 env 读取 improvement_pct
        answer_score = 0.0
        if environments and i < len(environments):
            env = environments[i]
            improvement = getattr(env, "improvement_pct", 0.0) / 100.0
            improvement = min(2.0, max(0.0, improvement))
            answer_score = math.log(1 + improvement) if improvement > 0 else 0.0

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

    # 场景数据
    parser.add_argument("--scenario_dir", default=None,
                        help="场景数据目录")
    parser.add_argument("--scenario_files", nargs="+", default=None,
                        help="场景 JSON 文件列表（优先于 scenario_dir）")

    # Cost Model
    parser.add_argument("--cost_model", default=None,
                        help="Cost Model checkpoint 目录")

    # 训练超参
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--num_generations", type=int, default=4,
                        help="GRPO 每个 prompt 生成的 completion 数")
    parser.add_argument("--max_completion_length", type=int, default=4096,
                        help="每轮 completion 最大 token 数")
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--max_turns", type=int, default=10,
                        help="agent 最大交互轮数")
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
            print(f"[WARNING] Cost Model 加载失败: {e}，answer_score 将为 0")

    # 构建 prompt 数据集
    scenario_source = args.scenario_files or args.scenario_dir
    if not scenario_source:
        raise ValueError("必须指定 --scenario_dir 或 --scenario_files")

    prompt_data = load_grpo_prompts(
        scenario_source if isinstance(scenario_source, str) else scenario_source[0]
    )

    # TRL 要求 conversational format
    dataset = Dataset.from_list([
        {"prompt": item["prompt"]}
        for item in prompt_data
    ])
    print(f"📊 数据集大小: {len(dataset)} 条 prompt")

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
        logging_steps=1,
        save_steps=50,
        save_total_limit=3,
        report_to="none",
        max_steps=args.max_steps,
        seed=42,
        # 禁用 thinking 模式
        chat_template_kwargs={"enable_thinking": False},
    )

    # environment_factory: TRL 每个 rollout 创建一个 env 实例
    from training.trl.db_env import DBTuningTRLEnv

    def env_factory():
        return DBTuningTRLEnv(
            cost_model=cost_model,
            scenario_dir=args.scenario_dir,
            scenario_files=args.scenario_files,
            knob_space_path=args.knob_space,
            max_turns=args.max_turns,
        )

    # Trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_config,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        train_dataset=dataset,
        peft_config=peft_config,
        environment_factory=env_factory,
    )

    # 显存统计
    if torch.cuda.is_available():
        total_params = sum(p.numel() for p in trainer.model.parameters())
        trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        print(f"\n{'='*60}")
        print(f"  参数统计")
        print(f"{'='*60}")
        print(f"  总参数:     {total_params/1e6:.1f}M")
        print(f"  可训练参数: {trainable_params/1e6:.1f}M ({trainable_params/total_params:.2%})")
        print(f"{'='*60}")
        torch.cuda.reset_peak_memory_stats()

    print("\n🚀 开始 GRPO 多轮工具调用训练...")
    trainer.train()

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\n📊 峰值显存: {peak:.2f} GB")

    trainer.save_model(args.output_dir)
    print(f"✅ 训练完成，模型保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
