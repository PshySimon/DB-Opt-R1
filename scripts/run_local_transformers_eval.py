#!/usr/bin/env python3
"""
本地 transformers 推理版 SFT eval runner。

复用 sampler 的 eval 逻辑与 evaluate.run 报表逻辑，只替换 LLM 调用后端：
- 不依赖 vLLM / OpenAI API
- 直接加载本地 HF 模型
- 适合单机离线评估
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


_DTYPE_MAP = {
    "auto": "auto",
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class LocalTransformersLLM:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_new_tokens: int = 512,
        attn_implementation: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._lock = threading.Lock()

        logger.info("加载本地 tokenizer: %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "trust_remote_code": True,
        }
        if dtype not in _DTYPE_MAP:
            raise ValueError(f"不支持的 dtype: {dtype}")
        if _DTYPE_MAP[dtype] != "auto":
            model_kwargs["torch_dtype"] = _DTYPE_MAP[dtype]
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        logger.info("加载本地模型: %s", model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        self.model.to(device)
        self.model.eval()

    def generate(self, messages_or_prompt: List[Dict] | str, temperature: float = 0.3) -> str:
        if isinstance(messages_or_prompt, str):
            prompt = messages_or_prompt
        else:
            prompt = self.tokenizer.apply_chat_template(
                messages_or_prompt,
                tokenize=False,
                add_generation_prompt=True,
            )

        with self._lock:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            prompt_len = inputs["input_ids"].shape[1]

            generate_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if temperature > 0:
                generate_kwargs["do_sample"] = True
                generate_kwargs["temperature"] = temperature
            else:
                generate_kwargs["do_sample"] = False

            with torch.inference_mode():
                outputs = self.model.generate(**inputs, **generate_kwargs)

            completion_ids = outputs[0][prompt_len:]
            text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
            return text.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地 transformers 推理版 SFT eval")
    parser.add_argument("--model-path", required=True, help="本地 HF 模型目录")
    parser.add_argument("--eval-questions", required=True, help="eval 问题 JSONL")
    parser.add_argument("--scenarios", required=True, help="评估场景 JSON")
    parser.add_argument("--cost-model", required=True, help="cost model checkpoint")
    parser.add_argument("--knob-space", required=True, help="knob_space.yaml")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--output-file", default="sft_trajectories.jsonl", help="输出文件名")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--parallel", type=int, default=1, help="沿用 sampler 参数；本地模型建议 1")
    parser.add_argument("--num-scenarios", type=int, default=-1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(_DTYPE_MAP.keys()))
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--attn-implementation", default=None,
                        help="可选 attention backend，如 sdpa/eager")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(f"未找到模型目录: {args.model_path}")

    llm = LocalTransformersLLM(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        attn_implementation=args.attn_implementation,
    )

    from data_pipeline.synthesis.trajectory.sampler import _run_eval

    logger.info("开始本地 transformers eval")
    logger.info("  model_path=%s", args.model_path)
    logger.info("  device=%s", args.device)
    logger.info("  dtype=%s", args.dtype)
    logger.info("  max_new_tokens=%s", args.max_new_tokens)
    logger.info("  attn_implementation=%s", args.attn_implementation or "default")
    logger.info("  parallel=%s", args.parallel)

    _run_eval(args, llm.generate)


if __name__ == "__main__":
    main()
