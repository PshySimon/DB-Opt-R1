"""
MCTS 节点定义
"""

import math
from typing import List, Optional


class MCTSNode:
    """蒙特卡洛树搜索节点"""

    def __init__(self, action: str = None, observation: str = None,
                 parent: 'MCTSNode' = None):
        """
        Args:
            action: 该节点的 Agent 行动（tool_call 字符串）
            observation: 工具返回结果
            parent: 父节点
        """
        self.action = action
        self.observation = observation
        self.parent = parent
        self.children: List[MCTSNode] = []

        self.visit_count = 0
        self.total_reward = 0.0
        self.depth = parent.depth + 1 if parent else 0

        # rollout 阶段的续写步骤（仅叶节点有值）
        self.rollout_trajectory: List[dict] = []

        # 搜索元数据
        self.sample_idx: Optional[int] = None  # 根节点记录环境样本索引

    @property
    def avg_reward(self) -> float:
        return self.total_reward / max(self.visit_count, 1)

    @property
    def trajectory(self) -> List[dict]:
        """从根到当前节点的轨迹（不含 rollout）"""
        path = []
        node = self
        while node.parent is not None:
            path.append({
                "action": node.action,
                "observation": node.observation,
            })
            node = node.parent
        return list(reversed(path))

    @property
    def full_trajectory(self) -> List[dict]:
        """从根到当前节点的完整轨迹（含 rollout 续写）"""
        return self.trajectory + self.rollout_trajectory

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_fully_expanded(self, max_children: int) -> bool:
        return len(self.children) >= max_children

    def ucb1(self, c: float = 1.414) -> float:
        """UCB1 分数"""
        if self.visit_count == 0:
            return float('inf')
        exploit = self.avg_reward
        explore = c * math.sqrt(math.log(self.parent.visit_count) / self.visit_count)
        return exploit + explore

    def best_child(self, c: float = 1.414) -> 'MCTSNode':
        """选 UCB1 最高的子节点"""
        return max(self.children, key=lambda child: child.ucb1(c))

    def best_child_by_reward(self) -> 'MCTSNode':
        """选平均 reward 最高的子节点"""
        return max(self.children, key=lambda child: child.avg_reward)

    def add_child(self, action: str, observation: str = None) -> 'MCTSNode':
        child = MCTSNode(action=action, observation=observation, parent=self)
        self.children.append(child)
        return child

    def to_dict(self) -> dict:
        """递归序列化为字典（用于保存为 JSON debug）"""
        d = {
            "action": self.action,
            "observation": self.observation,
            "depth": self.depth,
            "visit_count": self.visit_count,
            "total_reward": round(self.total_reward, 4),
            "avg_reward": round(self.avg_reward, 4),
            "rollout_trajectory": self.rollout_trajectory,
            "children": [c.to_dict() for c in self.children],
        }
        if self.sample_idx is not None:
            d["sample_idx"] = self.sample_idx
        return d

    @classmethod
    def from_dict(cls, d: dict, parent: 'MCTSNode' = None) -> 'MCTSNode':
        """从字典反序列化为 MCTSNode 树"""
        node = cls(action=d.get("action"), observation=d.get("observation"), parent=parent)
        node.visit_count = d.get("visit_count", 0)
        node.total_reward = d.get("total_reward", 0.0)
        node.rollout_trajectory = d.get("rollout_trajectory", [])
        node.sample_idx = d.get("sample_idx")
        for child_d in d.get("children", []):
            child = cls.from_dict(child_d, parent=node)
            node.children.append(child)
        return node

    def __repr__(self):
        action_str = (self.action[:50] + "...") if self.action and len(self.action) > 50 else self.action
        return (f"MCTSNode(depth={self.depth}, visits={self.visit_count}, "
                f"avg_reward={self.avg_reward:.3f}, action={action_str})")
