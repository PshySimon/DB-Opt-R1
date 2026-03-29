"""
从 MCTS 搜索树中提取 SFT 轨迹数据
"""

import json
from typing import List

from .node import MCTSNode


def extract_best_trajectory(root: MCTSNode) -> List[dict]:
    """提取最优路径（沿 avg_reward 最高的路径 + 叶节点 rollout）"""
    path = []
    node = root
    while node.children:
        node = node.best_child_by_reward()
        path.append({
            "action": node.action,
            "observation": node.observation,
        })
    # 拼接叶节点的 rollout 步骤
    if node.rollout_trajectory:
        path.extend(node.rollout_trajectory)
    return path


def extract_top_k_trajectories(root: MCTSNode, k: int = 3) -> List[List[dict]]:
    """提取 top-k 条不同的轨迹"""
    trajectories = []
    _collect_leaf_trajectories(root, trajectories)

    # 按 reward 排序
    trajectories.sort(key=lambda t: t["reward"], reverse=True)

    return [t["trajectory"] for t in trajectories[:k]]


def _collect_leaf_trajectories(node: MCTSNode, results: list):
    """递归收集所有叶节点的轨迹（含 rollout）"""
    if node.is_leaf() and node.visit_count > 0:
        results.append({
            "trajectory": node.full_trajectory,
            "reward": node.avg_reward,
        })
    for child in node.children:
        _collect_leaf_trajectories(child, results)


def extract_contrastive_pairs(root: MCTSNode) -> List[dict]:
    """提取正负对比数据"""
    pairs = []
    _collect_contrastive(root, pairs)
    return pairs


def _get_best_subtraj(node: MCTSNode) -> list:
    """从 node 开始，沿 best-reward 路径采集到叶节点，包含 rollout。"""
    steps = [{"action": node.action, "observation": node.observation or ""}]
    cur = node
    while cur.children:
        cur = cur.best_child_by_reward()
        steps.append({"action": cur.action, "observation": cur.observation or ""})
    if cur.rollout_trajectory:
        steps.extend(cur.rollout_trajectory)
    return steps


def _collect_contrastive(node: MCTSNode, pairs: list):
    """递归收集对比数据：同一决策点下最优 vs 最差"""
    visited_children = [c for c in node.children if c.visit_count > 0]
    if len(visited_children) >= 2:
        sorted_children = sorted(visited_children, key=lambda c: c.avg_reward, reverse=True)
        best = sorted_children[0]
        worst = sorted_children[-1]

        # 只有差距足够大才生成对比
        if best.avg_reward - worst.avg_reward > 0.01:
            # 追溯根节点，取搜索时存入的 user_message
            root = node
            while root.parent is not None:
                root = root.parent

            pairs.append({
                "context": node.trajectory,
                "user_message": root.user_message,
                "chosen_traj": _get_best_subtraj(best),
                "chosen_reward": best.avg_reward,
                "rejected_traj": _get_best_subtraj(worst),
                "rejected_reward": worst.avg_reward,
            })

    for child in node.children:
        _collect_contrastive(child, pairs)


def format_trajectory_as_messages(trajectory: List[dict],
                                   system_prompt: str,
                                   reward: float = None,
                                   sample_idx: int = None,
                                   user_message: str = "请优化这个数据库的性能。") -> dict:
    """将轨迹格式化为 SFT 训练数据（messages 格式）"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    for step in trajectory:
        messages.append({"role": "assistant", "content": step["action"]})
        if step.get("observation"):
            messages.append({"role": "tool", "content": step["observation"]})

    # 确保以 assistant 结尾
    if messages and messages[-1]["role"] == "tool":
        messages.append({
            "role": "assistant",
            "content": "<think>调优流程已完成，以上为全部操作步骤。</think>"
        })

    result = {"messages": messages}
    if reward is not None:
        result["reward"] = round(reward, 4)
    if sample_idx is not None:
        result["env_sample_idx"] = sample_idx

    return result


def format_contrastive_as_dpo(pair: dict, system_prompt: str) -> dict:
    """将对比数据格式化为 DPO 训练数据（chosen/rejected 均为完整对话）"""
    user_message = pair.get("user_message", "请优化这个数据库的性能。")

    full_chosen = format_trajectory_as_messages(
        trajectory=pair["context"] + pair["chosen_traj"],
        system_prompt=system_prompt,
        user_message=user_message,
        reward=pair.get("chosen_reward"),
    )
    full_rejected = format_trajectory_as_messages(
        trajectory=pair["context"] + pair["rejected_traj"],
        system_prompt=system_prompt,
        user_message=user_message,
        reward=pair.get("rejected_reward"),
    )

    return {
        "chosen": full_chosen["messages"],
        "rejected": full_rejected["messages"],
        "chosen_reward": pair.get("chosen_reward"),
        "rejected_reward": pair.get("rejected_reward"),
    }


def save_jsonl(data: list, path: str):
    """保存为 JSONL"""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
