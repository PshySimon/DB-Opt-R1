"""
评估脚本入口（指标聚合）

用法：
    # 1. 先用 sampler 生成 eval 轨迹
    python3 -m datasets.synthesis.trajectory.sampler \\
        --scenarios datasets/data/scenarios/knob_configs_eval.json \\
        --num-rollouts 1 \\
        --cost-model cost_model/checkpoints/v7_lgbm_dedup \\
        --output-dir eval_results/run_xxx/ \\
        --model gpt-5 --api-key sk-xxx --api-base https://xxx/v1

    # 2. 再用本脚本聚合报表
    python3 -m evaluate.run \\
        --eval-data eval_results/run_xxx/sft_trajectories.jsonl \\
        --output eval_results/run_xxx/
"""

import os
import sys
import json
import logging
import argparse

logger = logging.getLogger(__name__)


def load_trajectories(path: str) -> list:
    """读取 sampler 产出的轨迹 JSONL 文件"""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def compute_eval_metrics(trajectories: list) -> dict:
    """从轨迹列表计算评估指标"""
    if not trajectories:
        return {"error": "无轨迹数据"}

    improvements = [max(0, t.get("improvement_pct", 0.0)) for t in trajectories]
    rewards = [max(0, t.get("reward", 0.0)) for t in trajectories]

    def steps(t):
        msgs = t.get("messages", [])
        return sum(1 for m in msgs if m.get("role") == "assistant")

    def called_predict(t):
        msgs = t.get("messages", [])
        return any("predict_performance" in m.get("content", "") for m in msgs)

    steps_list = [steps(t) for t in trajectories]
    predict_flags = [called_predict(t) for t in trajectories]

    total = len(improvements)
    improved = sum(1 for v in improvements if v > 0)
    good = sum(1 for v in improvements if v > 3.0)

    return {
        "total_episodes": total,
        "improved_count": improved,
        "improved_rate": round(improved / total, 4) if total else 0,
        "good_count": good,
        "good_rate": round(good / total, 4) if total else 0,
        "avg_improvement_pct": round(sum(improvements) / total, 2) if total else 0,
        "max_improvement_pct": round(max(improvements), 2) if improvements else 0,
        "median_improvement_pct": round(sorted(improvements)[total // 2], 2) if improvements else 0,
        "avg_reward": round(sum(rewards) / total, 4) if total else 0,
        "avg_steps": round(sum(steps_list) / total, 1) if total else 0,
        "predict_call_rate": round(sum(predict_flags) / total, 4) if total else 0,
    }


def print_summary(metrics: dict):
    """打印评估报表"""
    print("\n" + "=" * 50)
    print("📊 评估报表")
    print("=" * 50)
    print(f"  总场景数:           {metrics.get('total_episodes', 0)}")
    print(f"  提升 > 0%:          {metrics.get('improved_count', 0)} ({metrics.get('improved_rate', 0)*100:.1f}%)")
    print(f"  提升 > 3%（好样本）: {metrics.get('good_count', 0)} ({metrics.get('good_rate', 0)*100:.1f}%)")
    print(f"  平均提升:           {metrics.get('avg_improvement_pct', 0):.2f}%")
    print(f"  最大提升:           {metrics.get('max_improvement_pct', 0):.2f}%")
    print(f"  中位数提升:         {metrics.get('median_improvement_pct', 0):.2f}%")
    print(f"  平均 reward:        {metrics.get('avg_reward', 0):.4f}")
    print(f"  平均步数:           {metrics.get('avg_steps', 0):.1f}")
    print(f"  predict 调用率:     {metrics.get('predict_call_rate', 0)*100:.1f}%")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="评估报表生成（读取 sampler 产出的 eval_trajectories.jsonl）"
    )
    parser.add_argument(
        "--eval-data", required=True,
        help="sampler 产出的轨迹文件路径（sft_trajectories.jsonl）"
    )
    parser.add_argument(
        "--output", default=None,
        help="报表输出目录（默认与 eval-data 同目录）"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not os.path.exists(args.eval_data):
        logger.error(f"轨迹文件不存在: {args.eval_data}")
        sys.exit(1)

    logger.info(f"读取轨迹文件: {args.eval_data}")
    trajectories = load_trajectories(args.eval_data)
    logger.info(f"  共 {len(trajectories)} 条轨迹")

    metrics = compute_eval_metrics(trajectories)
    print_summary(metrics)

    # 保存 JSON 报表
    output_dir = args.output or os.path.dirname(os.path.abspath(args.eval_data))
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "eval_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"报表已保存: {report_path}")


if __name__ == "__main__":
    main()
