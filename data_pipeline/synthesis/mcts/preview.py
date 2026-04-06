"""
预览合成数据集（SFT / Contrastive Pairs / MCTS 树）

用法:
  python3 -m data_pipeline.synthesis.mcts.preview data_pipeline/data/sft_trajectories.jsonl
  python3 -m data_pipeline.synthesis.mcts.preview data_pipeline/data/contrastive_pairs.jsonl
  python3 -m data_pipeline.synthesis.mcts.preview data_pipeline/data/mcts_trees/tree_env_0.json
  python3 -m data_pipeline.synthesis.mcts.preview data_pipeline/data/sft_trajectories.jsonl --index 0 --full
"""

import argparse
import json
import re
import sys
import os

# ANSI 颜色
C = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "system": "\033[35m",     # 紫色
    "user": "\033[34m",       # 蓝色
    "assistant": "\033[32m",  # 绿色
    "tool": "\033[33m",       # 黄色
    "header": "\033[36m",     # 青色
    "reward": "\033[31m",     # 红色
    "sep": "\033[90m",        # 灰色
}


def extract_tool_name(text: str) -> str:
    """从 assistant 消息中提取工具名"""
    m = re.search(r'"name"\s*:\s*"(\w+)"', text)
    return m.group(1) if m else ""


def truncate(text: str, max_len: int = 200) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"...({len(text)} chars)"


def preview_sft(path: str, index: int = None, full: bool = False):
    """预览 SFT 轨迹"""
    with open(path, "r") as f:
        lines = f.readlines()

    print(f"{C['header']}{C['bold']}═══ SFT 轨迹预览 ═══{C['reset']}")
    print(f"{C['dim']}文件: {path}  |  共 {len(lines)} 条{C['reset']}\n")

    indices = [index] if index is not None else range(len(lines))

    for i in indices:
        if i >= len(lines):
            print(f"  索引 {i} 超出范围 (共 {len(lines)} 条)")
            continue

        data = json.loads(lines[i])
        msgs = data.get("messages", [])
        reward = data.get("reward")
        env_idx = data.get("env_sample_idx")

        # 统计
        tools = [extract_tool_name(m["content"]) for m in msgs if m["role"] == "assistant"]
        tools = [t for t in tools if t]

        print(f"{C['header']}{'─' * 60}")
        print(f"  样本 {i}  |  {len(msgs)} 条消息  |  env={env_idx}", end="")
        if reward is not None:
            print(f"  |  {C['reward']}reward={reward:.4f}{C['header']}", end="")
        print(f"\n  工具链: {' → '.join(tools)}")
        print(f"{'─' * 60}{C['reset']}")

        for j, msg in enumerate(msgs):
            role = msg["role"]
            content = msg["content"]
            color = C.get(role, C["reset"])

            if role == "system":
                print(f"\n  {color}[SYSTEM]{C['reset']} {C['dim']}{truncate(content, 80)}{C['reset']}")

            elif role == "user":
                print(f"\n  {color}[USER]{C['reset']} {content}")

            elif role == "assistant":
                tool = extract_tool_name(content)
                label = f"[ASSISTANT → {tool}]" if tool else "[ASSISTANT]"
                print(f"\n  {color}{label}{C['reset']}")

                # 分离 think 和 tool_call
                think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
                if think_match:
                    think = think_match.group(1).strip()
                    if not full:
                        think = truncate(think, 150)
                    print(f"  {C['dim']}💭 {think}{C['reset']}")

                tc_match = re.search(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
                if tc_match:
                    tc = tc_match.group(1).strip()
                    print(f"  🔧 {tc}")

                # 无 tool_call 的纯文本（如收尾消息）
                if not tc_match and not think_match:
                    print(f"  {content}")

            elif role == "tool":
                if full:
                    try:
                        parsed = json.loads(content)
                        formatted = json.dumps(parsed, ensure_ascii=False, indent=4)
                        # 缩进每行
                        for line in formatted.split("\n"):
                            print(f"  {color}│{C['reset']} {line}")
                    except json.JSONDecodeError:
                        print(f"  {color}│{C['reset']} {content}")
                else:
                    print(f"  {color}📋 {truncate(content, 120)}{C['reset']}")

        print()


def preview_contrastive(path: str, index: int = None, full: bool = False):
    """预览 Contrastive Pairs"""
    with open(path, "r") as f:
        lines = f.readlines()

    print(f"{C['header']}{C['bold']}═══ Contrastive Pairs 预览 ═══{C['reset']}")
    print(f"{C['dim']}文件: {path}  |  共 {len(lines)} 对{C['reset']}\n")

    indices = [index] if index is not None else range(len(lines))

    for i in indices:
        if i >= len(lines):
            continue
        pair = json.loads(lines[i])

        chosen_tool = extract_tool_name(pair.get("chosen", ""))
        rejected_tool = extract_tool_name(pair.get("rejected", ""))

        print(f"{C['header']}{'─' * 60}")
        print(f"  对 {i}")
        print(f"{'─' * 60}{C['reset']}")

        # context 工具链
        context = pair.get("prompt", [])
        ctx_tools = [extract_tool_name(m["content"]) for m in context if m["role"] == "assistant"]
        ctx_tools = [t for t in ctx_tools if t]
        print(f"  {C['dim']}上下文: {' → '.join(ctx_tools) or '(空)'}{C['reset']}")

        # chosen
        cr = pair.get("chosen_reward")
        print(f"\n  {C['assistant']}✅ Chosen: {chosen_tool}  {C['reward']}(reward={cr:.4f}){C['reset']}")
        if full:
            think = re.search(r"<think>(.*?)</think>", pair.get("chosen", ""), re.DOTALL)
            if think:
                print(f"  {C['dim']}💭 {think.group(1).strip()}{C['reset']}")

        # rejected
        rr = pair.get("rejected_reward")
        print(f"  {C['reward']}❌ Rejected: {rejected_tool}  (reward={rr:.4f}){C['reset']}")
        if full:
            think = re.search(r"<think>(.*?)</think>", pair.get("rejected", ""), re.DOTALL)
            if think:
                print(f"  {C['dim']}💭 {think.group(1).strip()}{C['reset']}")

        print()


def preview_tree(path: str, max_depth: int = 3):
    """预览 MCTS 树"""
    with open(path, "r") as f:
        tree = json.load(f)

    print(f"{C['header']}{C['bold']}═══ MCTS 搜索树预览 ═══{C['reset']}")
    print(f"{C['dim']}文件: {path}{C['reset']}\n")

    def print_node(node, prefix="", is_last=True, depth=0):
        connector = "└── " if is_last else "├── "
        extend = "    " if is_last else "│   "

        tool = extract_tool_name(node.get("action", "") or "")
        visits = node.get("visit_count", 0)
        avg_r = node.get("avg_reward", 0)
        rollout_len = len(node.get("rollout_trajectory", []))

        if depth == 0:
            label = f"{C['bold']}ROOT{C['reset']}"
            extra = f"visits={visits}, avg_r={avg_r:.4f}"
            sample_idx = node.get("sample_idx")
            if sample_idx is not None:
                extra += f", sample_idx={sample_idx}"
            print(f"  {label}  {C['dim']}({extra}){C['reset']}")
        else:
            color = C["assistant"] if visits > 0 else C["dim"]
            rollout_tag = f" {C['reward']}[+{rollout_len} rollout]{C['reset']}" if rollout_len > 0 else ""
            print(f"  {prefix}{connector}{color}{tool}{C['reset']}  "
                  f"{C['dim']}v={visits} r={avg_r:.4f}{C['reset']}{rollout_tag}")

        children = node.get("children", [])
        if depth < max_depth:
            for ci, child in enumerate(children):
                is_child_last = (ci == len(children) - 1)
                child_prefix = prefix + extend if depth > 0 else prefix
                print_node(child, child_prefix, is_child_last, depth + 1)
        elif children:
            print(f"  {prefix}{extend}{C['dim']}... ({len(children)} children){C['reset']}")

    print_node(tree)
    print()


def main():
    parser = argparse.ArgumentParser(description="预览合成数据集")
    parser.add_argument("file", help="数据文件路径 (.jsonl 或 .json)")
    parser.add_argument("--index", "-i", type=int, default=None,
                        help="只看第 N 条样本 (仅 JSONL)")
    parser.add_argument("--full", "-f", action="store_true",
                        help="显示完整内容（不截断）")
    parser.add_argument("--depth", "-d", type=int, default=4,
                        help="树的最大显示深度 (仅 JSON 树)")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"文件不存在: {args.file}")
        sys.exit(1)

    if args.file.endswith(".json"):
        preview_tree(args.file, max_depth=args.depth)
    elif "contrastive" in args.file:
        preview_contrastive(args.file, index=args.index, full=args.full)
    else:
        preview_sft(args.file, index=args.index, full=args.full)


if __name__ == "__main__":
    main()
