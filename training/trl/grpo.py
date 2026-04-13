"""
trl GRPO 多轮工具调用训练（外置 vLLM server）

架构：
- 生成：调用外置 vLLM OpenAI-compatible server
- 训练：HF 模型 forward + backward（LoRA）
- 推理和训练分别占用不同 GPU，避免显存争用

Usage:
    PYTHONPATH=. python -m training.trl.grpo \\
        --model_path model_save/sft_qwen3_4b_cleaned_merged \\
        --train_data data_pipeline/data/train/sft_trajectories.jsonl \\
        --scenario_files ... \\
        --cost_model cost_model/checkpoints/v9_lgbm \\
        --vllm_server_host 127.0.0.1 \\
        --vllm_server_port 8000 \\
        --vllm_model_name qwen3-4b-sft \\
        --output_dir model_save/grpo/
"""

import json
import math
import argparse
import inspect
import logging
import time
from urllib.parse import urlparse, urlunparse

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


class _SetTrainMode(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        use_lora = self.const == "lora"
        setattr(namespace, "use_lora", use_lora)
        setattr(namespace, "full_finetune", not use_lora)


def build_vllm_base_url(host: str, port: int) -> str:
    """把 host/port 规范化成 OpenAI client 需要的 /v1 base_url。"""
    raw = host.strip()
    if "://" not in raw:
        raw = f"http://{raw}"
    parsed = urlparse(raw)

    hostname = parsed.hostname or "127.0.0.1"
    netloc = parsed.netloc or hostname
    if parsed.port is None:
        netloc = f"{hostname}:{port}"

    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        path = f"{path}/v1" if path else "/v1"

    return urlunparse((parsed.scheme or "http", netloc, path, "", "", ""))


def format_rollout_turn_log(rollout_id, turn_idx, active_count, batch_size, prompt_token_lengths, elapsed_s):
    avg_prompt_tokens = int(sum(prompt_token_lengths) / max(len(prompt_token_lengths), 1))
    max_prompt_tokens = max(prompt_token_lengths) if prompt_token_lengths else 0
    min_prompt_tokens = min(prompt_token_lengths) if prompt_token_lengths else 0
    return (
        f"[rollout#{rollout_id}] turn {turn_idx + 1} "
        f"active={active_count}/{batch_size} "
        f"prompt_tokens(avg={avg_prompt_tokens}, min={min_prompt_tokens}, max={max_prompt_tokens}) "
        f"elapsed={elapsed_s:.2f}s"
    )


def expand_inputs_for_generations(inputs, prompt_batch_size, num_generations):
    if num_generations <= 1 or len(inputs) != prompt_batch_size:
        return inputs

    expanded = []
    for item in inputs:
        expanded.extend([item] * num_generations)
    return expanded


class VLLMServerBackend:
    """通过 OpenAI-compatible API 访问外置 vLLM server。"""

    def __init__(self, model_name, base_url, api_key="EMPTY", timeout=300, client=None):
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout
        self.client = client or self._create_client(api_key=api_key)

    def _create_client(self, api_key):
        from openai import OpenAI

        return OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            max_retries=0,
            timeout=self.timeout,
            default_headers={"User-Agent": "Mozilla/5.0"},
        )

    def _extract_batch_texts(self, response, batch_size):
        outputs = [""] * batch_size
        for choice in response.choices:
            outputs[int(choice.index)] = choice.text or ""
        return outputs

    def _generate_one(self, prompt, temperature, top_p, max_tokens, stop):
        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
        )
        return response.choices[0].text or ""

    def generate_texts(self, prompts, temperature, top_p, max_tokens, stop):
        if not prompts:
            return []

        try:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompts,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop,
            )
            return self._extract_batch_texts(response, batch_size=len(prompts))
        except Exception as exc:
            logger.warning("批量请求 vLLM 失败，降级为逐条请求: %s", exc)
            return [
                self._generate_one(prompt, temperature, top_p, max_tokens, stop)
                for prompt in prompts
            ]


class MultiTurnGRPOTrainer(GRPOTrainer):
    """基于外置 vLLM server 的多轮工具交互 GRPO Trainer"""

    def __init__(self, *args, env_factory=None, prompt_env_map=None,
                 max_turns=10, generation_backend=None,
                 rollout_tokenizer=None, generation_max_tokens=1024,
                 rollout_log_interval=1, prompt_batch_size=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_factory = env_factory
        self.prompt_env_map = prompt_env_map or {}
        self.max_turns = max_turns
        self.generation_backend = generation_backend
        self.rollout_tokenizer = rollout_tokenizer
        self.generation_max_tokens = generation_max_tokens
        self.rollout_log_interval = rollout_log_interval
        self.prompt_batch_size = prompt_batch_size
        self._rollout_call_idx = 0

    def _do_rollouts_batched(self, prompts):
        """
        批量多轮 rollout：每轮把所有活跃 rollout 一起交给 vLLM server 批量生成。
        比逐条串行快 N 倍（N = num_generations）。
        """
        n = len(prompts)
        self._rollout_call_idx += 1
        rollout_id = self._rollout_call_idx
        rollout_start = time.perf_counter()

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

            turn_start = time.perf_counter()
            # 把活跃 rollout 的 messages 格式化成 prompt 文本
            batch_texts = []
            for i in active:
                text = self.rollout_tokenizer.apply_chat_template(
                    messages_list[i], tokenize=False, add_generation_prompt=True
                )
                # turn > 0: 追加 <think> 引导，与 VERL tool_custom_response_template 一致
                if turn > 0:
                    text += "<think>\n"
                batch_texts.append(text)

            prompt_token_lengths = [
                len(self.rollout_tokenizer.encode(text, add_special_tokens=False))
                for text in batch_texts
            ]
            outputs = self.generation_backend.generate_texts(
                batch_texts,
                temperature=1.0,
                top_p=1.0,
                max_tokens=self.generation_max_tokens,
                stop=["</tool_call>"],
            )

            # 处理每条输出
            still_active = []
            for j, i in enumerate(active):
                gen_text = outputs[j] or ""

                # 补全 stop token（include_stop_str_in_output 有时不可靠）
                if "<tool_call>" in gen_text and "</tool_call>" not in gen_text:
                    gen_text += "</tool_call>"

                # turn > 0: <think> 已作为 prompt 的一部分发送，需补回到 message 内容中
                if turn > 0:
                    gen_text = "<think>\n" + gen_text

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
            if self.rollout_log_interval > 0 and (
                turn == 0 or
                (turn + 1) % self.rollout_log_interval == 0 or
                not active
            ):
                logger.info(
                    format_rollout_turn_log(
                        rollout_id=rollout_id,
                        turn_idx=turn,
                        active_count=len(active) if active else len(still_active),
                        batch_size=n,
                        prompt_token_lengths=prompt_token_lengths,
                        elapsed_s=time.perf_counter() - turn_start,
                    )
                )

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

        logger.info(
            "[rollout#%s] completed batch=%s total_elapsed=%.2fs avg_improvement=%.4f",
            rollout_id,
            n,
            time.perf_counter() - rollout_start,
            sum(improvements) / max(len(improvements), 1),
        )

        return prompt_msgs_list, messages_list, improvements

    def _generate_and_score_completions(self, inputs):
        device = self.accelerator.device
        expanded_inputs = expand_inputs_for_generations(
            inputs=inputs,
            prompt_batch_size=self.prompt_batch_size,
            num_generations=self.num_generations,
        )
        prompts = [x["prompt"] for x in expanded_inputs]

        # ==================== 批量多轮 rollout ====================
        prompt_msgs_list, messages_list, improvements = self._do_rollouts_batched(prompts)

        # ==================== Tokenize ====================
        all_prompt_texts = []
        all_completion_ids = []

        for i in range(len(prompts)):
            prompt_text = self.rollout_tokenizer.apply_chat_template(
                prompt_msgs_list[i], tokenize=False, add_generation_prompt=True
            )
            full_text = self.rollout_tokenizer.apply_chat_template(
                messages_list[i], tokenize=False, add_generation_prompt=False
            )
            all_prompt_texts.append(prompt_text)

            # 提取 completion token ids
            prompt_len = len(self.rollout_tokenizer.encode(prompt_text, add_special_tokens=False))
            full_ids = self.rollout_tokenizer.encode(full_text, add_special_tokens=False)
            comp_ids = full_ids[prompt_len:]

            if len(comp_ids) > self.max_completion_length:
                comp_ids = comp_ids[:self.max_completion_length]
            if len(comp_ids) == 0:
                comp_ids = [self.rollout_tokenizer.eos_token_id]

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

def build_parser():
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
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--use_lora", nargs=0, action=_SetTrainMode, const="lora")
    mode_group.add_argument("--full_finetune", nargs=0, action=_SetTrainMode, const="full")
    parser.add_argument("--attn_impl", default="sdpa",
                        choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--vllm_server_host", default="127.0.0.1")
    parser.add_argument("--vllm_server_port", type=int, default=8000)
    parser.add_argument("--vllm_model_name", default="qwen3-4b-sft")
    parser.add_argument("--vllm_api_key", default="EMPTY")
    parser.add_argument("--vllm_timeout", type=int, default=300)
    parser.add_argument("--vllm_max_tokens", type=int, default=1024)
    parser.add_argument("--rollout_log_interval", type=int, default=1)
    parser.add_argument("--rollout_batch_size", type=int, default=None,
                        help="单独控制 rollout/generation batch；不填则使用 TRL 默认调度")
    parser.set_defaults(use_lora=True, full_finetune=False)
    return parser


def build_grpo_config_kwargs(args):
    kwargs = dict(
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
        beta=0.0,
        scale_rewards=True,
    )
    if args.rollout_batch_size is not None:
        kwargs["generation_batch_size"] = args.rollout_batch_size
    return kwargs


def make_grpo_config(args):
    kwargs = build_grpo_config_kwargs(args)
    try:
        signature = inspect.signature(GRPOConfig.__init__)
        supported = set(signature.parameters.keys())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported}
        dropped = sorted(set(kwargs.keys()) - set(filtered_kwargs.keys()))
        if dropped:
            logger.warning("当前 TRL 版本不支持这些 GRPOConfig 参数，已忽略: %s", ", ".join(dropped))
        return GRPOConfig(**filtered_kwargs)
    except (TypeError, ValueError):
        return GRPOConfig(**kwargs)


def main():
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )

    # Cost Model
    cost_model = None
    if args.cost_model:
        try:
            from cost_model.model import CostModel
            cost_model = CostModel.load(args.cost_model)
            print(f"✅ Cost Model: {args.cost_model}")
        except Exception as e:
            print(f"[WARNING] Cost Model 加载失败: {e}")

    # ---- 连接外置 vLLM server ----
    vllm_base_url = build_vllm_base_url(args.vllm_server_host, args.vllm_server_port)
    print(
        f"🔄 连接 vLLM server: {vllm_base_url} "
        f"(model={args.vllm_model_name})"
    )
    generation_backend = VLLMServerBackend(
        model_name=args.vllm_model_name,
        base_url=vllm_base_url,
        api_key=args.vllm_api_key,
        timeout=args.vllm_timeout,
    )
    rollout_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    print("✅ vLLM server 就绪")

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
        attn_implementation=args.attn_impl,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    peft_config = None
    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_rank // 2, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )

    training_config = make_grpo_config(args)

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
        generation_backend=generation_backend,
        rollout_tokenizer=rollout_tokenizer,
        generation_max_tokens=args.vllm_max_tokens,
        rollout_log_interval=args.rollout_log_interval,
        prompt_batch_size=args.batch_size,
    )

    if torch.cuda.is_available():
        tp = sum(p.numel() for p in trainer.model.parameters())
        trp = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        print(f"\n  总参数: {tp/1e6:.1f}M | 可训练: {trp/1e6:.1f}M ({trp/tp:.2%})")

    mode_name = "LoRA" if args.use_lora else "全量"
    print(f"\n🚀 开始 GRPO 训练（外置 vLLM server，模式: {mode_name}）...")
    trainer.train()

    if torch.cuda.is_available():
        print(f"📊 峰值显存: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

    trainer.save_model(args.output_dir)
    print(f"✅ 保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
