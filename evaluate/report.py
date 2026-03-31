"""
评估报告生成
"""

import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def compute_metrics(results: List[Dict], meta: Dict = None) -> Dict[str, Any]:
    """
    汇总所有 episode 结果，生成评估报告。

    Args:
        results: run_episode 返回的结果列表
        meta: 元信息（模型名、参数等）

    Returns:
        完整报告字典
    """
    n = len(results)
    if n == 0:
        return {"meta": meta, "summary": {}, "per_episode": []}

    # 核心指标
    total_rewards = [r["total_reward"] for r in results]
    steps_list = [r["steps"] for r in results]
    called_predict_list = [r["called_predict"] for r in results]
    format_pass_list = [r["format_pass"] for r in results]
    knobs_set_list = [r["num_knobs_set"] for r in results]
    valid_actions_list = [r["valid_actions"] for r in results]
    effective_actions_list = [r["effective_actions"] for r in results]

    # 成功率：调用了 predict_performance 的比例
    completion_rate = sum(called_predict_list) / n

    # 格式完美率
    format_pass_rate = sum(format_pass_list) / n

    # 有效工具调用率
    total_valid = sum(valid_actions_list)
    total_effective = sum(effective_actions_list)
    total_steps = sum(steps_list)
    effective_rate = total_effective / total_steps if total_steps > 0 else 0

    summary = {
        "num_episodes": n,
        "avg_total_reward": round(sum(total_rewards) / n, 4),
        "median_total_reward": round(sorted(total_rewards)[n // 2], 4),
        "completion_rate": round(completion_rate, 4),
        "format_pass_rate": round(format_pass_rate, 4),
        "effective_rate": round(effective_rate, 4),
        "avg_steps": round(sum(steps_list) / n, 2),
        "avg_knobs_set": round(sum(knobs_set_list) / n, 2),
        "positive_reward_rate": round(
            sum(1 for r in total_rewards if r > 0) / n, 4
        ),
    }

    report = {
        "meta": {
            **(meta or {}),
            "timestamp": datetime.now().isoformat(),
        },
        "summary": summary,
        "per_episode": [
            {
                "sample_idx": r["sample_idx"],
                "steps": r["steps"],
                "total_reward": round(r["total_reward"], 4),
                "called_predict": r["called_predict"],
                "format_pass": r["format_pass"],
                "num_knobs_set": r["num_knobs_set"],
                "valid_actions": r["valid_actions"],
                "effective_actions": r["effective_actions"],
            }
            for r in results
        ],
    }

    return report


def save_report(report: Dict, output_dir: str):
    """保存报告到 JSON 文件"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"报告已保存: {path}")


def print_summary(report: Dict):
    """打印摘要到控制台"""
    s = report.get("summary", {})
    meta = report.get("meta", {})

    print("\n" + "=" * 60)
    print(f"  评估报告 — {meta.get('model', 'unknown')}")
    print("=" * 60)
    print(f"  场景数:         {s.get('num_episodes', 0)}")
    print(f"  平均 reward:    {s.get('avg_total_reward', 0):.4f}")
    print(f"  中位数 reward:  {s.get('median_total_reward', 0):.4f}")
    print(f"  正向 reward 率: {s.get('positive_reward_rate', 0):.1%}")
    print(f"  完成率:         {s.get('completion_rate', 0):.1%}")
    print(f"  格式合规率:     {s.get('format_pass_rate', 0):.1%}")
    print(f"  有效调用率:     {s.get('effective_rate', 0):.1%}")
    print(f"  平均步数:       {s.get('avg_steps', 0):.1f}")
    print(f"  平均调参数:     {s.get('avg_knobs_set', 0):.1f}")
    print("=" * 60 + "\n")
