#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into its base HuggingFace model."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="Base HuggingFace model path")
    parser.add_argument("--adapter", required=True, help="LoRA adapter path")
    parser.add_argument("--output-dir", required=True, help="Merged model output directory")
    parser.add_argument("--device-map", default="auto", help="transformers device_map, default: auto")
    parser.add_argument("--overwrite", action="store_true", help="Remove output dir if it already exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_model = Path(args.base_model)
    adapter = Path(args.adapter)
    output_dir = Path(args.output_dir)

    if not base_model.exists():
        raise FileNotFoundError(f"base model not found: {base_model}")
    if not adapter.exists():
        raise FileNotFoundError(f"LoRA adapter not found: {adapter}")
    if output_dir.exists():
        if not args.overwrite:
            print(f"merged model already exists: {output_dir}")
            return
        shutil.rmtree(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(str(base_model), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=args.device_map,
    )
    model = PeftModel.from_pretrained(model, str(adapter))
    model = model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(output_dir))
    model.save_pretrained(str(output_dir), safe_serialization=True)
    print(f"merged saved to: {output_dir}")


if __name__ == "__main__":
    main()
