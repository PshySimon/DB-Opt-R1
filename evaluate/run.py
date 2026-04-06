"""
评估报表生成（三 baseline 体系）

从 sampler eval 模式产出的轨迹中提取模型设置的 knobs，
用 Cost Model 重新预测 TPS，与三个 baseline 对比：
  1. scenario_tps  — 场景原始 tps_current
  2. default_tps   — PG 默认配置的 Cost Model 预测
  3. optimal_tps   — BO 搜索的 Cost Model 最优预测

核心指标：gap_closed = (model - default) / (optimal - default)

用法：
    python3 -m evaluate.run \
        --eval-data eval_results/gpt5_v2/sft_trajectories.jsonl \
        --scenarios data_pipeline/data/scenarios/collected/collected_eval.json \
        --cost-model cost_model/checkpoints/v8_lgbm \
        --output eval_results/gpt5_v2/
"""

import os
import sys
import json
import re
import logging
import argparse
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_trajectories(path: str) -> list:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def extract_model_knobs(trajectory: dict) -> dict:
    """从轨迹的 messages 中提取模型最终设置的 knobs（累积所有 set_knob 调用）。"""
    knobs = {}
    for msg in trajectory.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', content, re.DOTALL)
        if not match:
            continue
        try:
            call = json.loads(match.group(1))
            if call.get("name") != "set_knob":
                continue
            args = call.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            ks = args.get("knobs", "{}")
            if isinstance(ks, str):
                ks = json.loads(ks)
            knobs.update(ks)
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return knobs


def bo_search_optimal(cost_model, hw_info: dict, workload: str,
                      knob_space, n_trials: int = 200) -> float:
    """用 BO 搜索 Cost Model 下给定硬件+负载的最优 TPS。"""
    from core.db.optimizer import optuna_search

    def objective(knobs):
        knobs_with_wl = {**knobs, "workload": workload}
        return cost_model.predict(knobs_with_wl, hw_info)

    best_value, _ = optuna_search(knob_space, objective, n_trials=n_trials)
    return best_value


def compute_eval_metrics(trajectories, scenarios, cost_model, knob_space,
                         n_bo_trials=200) -> dict:
    """三 baseline 体系评估。"""
    if not trajectories:
        return {"error": "无轨迹数据"}

    from core.db.knob_space import KnobSpace
    default_knobs = knob_space.get_default_config()

    # BO 缓存：(hw_hash, wl_type) → optimal_tps
    bo_cache = {}

    per_episode = []

    for t in trajectories:
        env_idx = t.get("env_sample_idx")
        if env_idx is None or env_idx >= len(scenarios):
            continue

        s = scenarios[env_idx]
        hw = s.get("hardware", {})
        wl = s.get("workload", {})
        wl_type = wl.get("type", "mixed") if isinstance(wl, dict) else str(wl)

        # 场景原始配置 TPS（用 Cost Model 预测，和其他 baseline 保持一致）
        original_knobs = s.get("knobs", {})
        ok = {**original_knobs, "workload": wl_type}
        try:
            scenario_tps = cost_model.predict(ok, hw)
        except Exception:
            scenario_tps = 0.0

        # 默认配置 TPS
        dk = default_knobs.copy()
        dk["workload"] = wl_type
        try:
            default_tps = cost_model.predict(dk, hw)
        except Exception:
            default_tps = 0.0

        # BO 最优 TPS（按 hw+wl 缓存）
        cache_key = json.dumps(hw, sort_keys=True) + "|" + wl_type
        if cache_key not in bo_cache:
            bo_cache[cache_key] = bo_search_optimal(
                cost_model, hw, wl_type, knob_space, n_bo_trials)
            logger.info(f"  BO ({len(bo_cache)}): wl={wl_type}, "
                        f"optimal={bo_cache[cache_key]:.1f}")
        optimal_tps = bo_cache[cache_key]

        # 模型预测 TPS（以场景原始 knobs 为基底，叠加模型的修改）
        model_changes = extract_model_knobs(t)
        if model_changes:
            mk = {**original_knobs, **model_changes, "workload": wl_type}
            try:
                model_tps = cost_model.predict(mk, hw)
            except Exception:
                model_tps = 0.0
        else:
            model_tps = 0.0

        # 三个 improvement（与 predict_performance 保持一致：下限 0%，上限 200%）
        IMPROVEMENT_CAP = 200.0
        raw_vs_scenario = ((model_tps - scenario_tps) / scenario_tps * 100
                           if scenario_tps > 0 else 0)
        imp_vs_scenario = min(IMPROVEMENT_CAP, max(0, raw_vs_scenario))

        raw_vs_default = ((model_tps - default_tps) / default_tps * 100
                          if default_tps > 0 else 0)
        imp_vs_default = min(IMPROVEMENT_CAP, max(0, raw_vs_default))

        raw_vs_optimal = ((model_tps - optimal_tps) / optimal_tps * 100
                          if optimal_tps > 0 else 0)
        imp_vs_optimal = min(IMPROVEMENT_CAP, max(0, raw_vs_optimal))

        gap = optimal_tps - default_tps
        gap_closed = ((model_tps - default_tps) / gap * 100
                      if gap > 0 else 0)
        gap_closed = min(IMPROVEMENT_CAP, max(0, gap_closed))

        # 行为
        msgs = t.get("messages", [])
        n_steps = sum(1 for m in msgs if m.get("role") == "assistant")
        called_predict = any("predict_performance" in m.get("content", "")
                             for m in msgs)

        per_episode.append({
            "env_sample_idx": env_idx,
            "name": s.get("name", f"env_{env_idx}"),
            "model_tps": round(model_tps, 1),
            "scenario_tps": round(scenario_tps, 1),
            "default_tps": round(default_tps, 1),
            "optimal_tps": round(optimal_tps, 1),
            "imp_vs_scenario_pct": round(imp_vs_scenario, 2),
            "imp_vs_default_pct": round(imp_vs_default, 2),
            "imp_vs_optimal_pct": round(imp_vs_optimal, 2),
            "gap_closed_pct": round(gap_closed, 2),
            "steps": n_steps,
            "num_knobs_set": len(model_changes),
            "called_predict": called_predict,
        })

    # 聚合
    n = len(per_episode)
    if n == 0:
        return {"error": "无有效轨迹"}

    imp_s = [e["imp_vs_scenario_pct"] for e in per_episode]
    imp_d = [e["imp_vs_default_pct"] for e in per_episode]
    imp_o = [e["imp_vs_optimal_pct"] for e in per_episode]
    gc = [e["gap_closed_pct"] for e in per_episode]

    summary = {
        "total_episodes": n,
        # vs 场景原始配置
        "avg_imp_vs_scenario_pct": round(np.mean(imp_s), 2),
        "median_imp_vs_scenario_pct": round(np.median(imp_s), 2),
        "improved_vs_scenario_rate": round(sum(1 for v in imp_s if v > 0) / n, 4),
        # vs PG 默认配置
        "avg_imp_vs_default_pct": round(np.mean(imp_d), 2),
        "median_imp_vs_default_pct": round(np.median(imp_d), 2),
        "improved_vs_default_rate": round(sum(1 for v in imp_d if v > 0) / n, 4),
        # vs BO 最优配置
        "avg_imp_vs_optimal_pct": round(np.mean(imp_o), 2),
        "median_imp_vs_optimal_pct": round(np.median(imp_o), 2),
        # gap closing
        "avg_gap_closed_pct": round(np.mean(gc), 2),
        "median_gap_closed_pct": round(np.median(gc), 2),
        # 行为
        "avg_steps": round(np.mean([e["steps"] for e in per_episode]), 1),
        "predict_call_rate": round(
            sum(e["called_predict"] for e in per_episode) / n, 4),
    }

    return {"summary": summary, "per_episode": per_episode}


def print_summary(metrics: dict):
    s = metrics.get("summary", {})
    print("\n" + "=" * 60)
    print("📊 评估报表（三 baseline 体系）")
    print("=" * 60)
    print(f"  总场景数:              {s.get('total_episodes', 0)}")
    print()
    print("  ── vs 场景原始配置（Cost Model 预测）──")
    print(f"  平均提升:              {s.get('avg_imp_vs_scenario_pct', 0):.2f}%")
    print(f"  中位数提升:            {s.get('median_imp_vs_scenario_pct', 0):.2f}%")
    print(f"  提升 > 0% 比例:        {s.get('improved_vs_scenario_rate', 0)*100:.1f}%")
    print()
    print("  ── vs PG 默认配置（Cost Model 预测）──")
    print(f"  平均提升:              {s.get('avg_imp_vs_default_pct', 0):.2f}%")
    print(f"  中位数提升:            {s.get('median_imp_vs_default_pct', 0):.2f}%")
    print(f"  提升 > 0% 比例:        {s.get('improved_vs_default_rate', 0)*100:.1f}%")
    print()
    print("  ── vs BO 最优配置（Cost Model 预测）──")
    print(f"  平均差距:              {s.get('avg_imp_vs_optimal_pct', 0):.2f}%")
    print(f"  中位数差距:            {s.get('median_imp_vs_optimal_pct', 0):.2f}%")
    print()
    print("  ── Gap Closing（核心）──")
    print(f"  平均 gap closed:       {s.get('avg_gap_closed_pct', 0):.2f}%")
    print(f"  中位数 gap closed:     {s.get('median_gap_closed_pct', 0):.2f}%")
    print()
    print("  ── 行为指标 ──")
    print(f"  平均步数:              {s.get('avg_steps', 0):.1f}")
    print(f"  predict 调用率:        {s.get('predict_call_rate', 0)*100:.1f}%")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="评估报表生成（三 baseline 体系）")
    parser.add_argument("--eval-data", required=True,
                        help="sampler eval 模式产出的轨迹文件")
    parser.add_argument("--scenarios", required=True,
                        help="eval 场景文件")
    parser.add_argument("--cost-model", required=True,
                        help="Cost Model 路径")
    parser.add_argument("--knob-space", default="configs/knob_space.yaml")
    parser.add_argument("--output", default=None,
                        help="报表输出目录")
    parser.add_argument("--bo-trials", type=int, default=200,
                        help="BO 搜索试验次数")
    args = parser.parse_args()

    logger.info(f"读取轨迹: {args.eval_data}")
    trajectories = load_trajectories(args.eval_data)
    logger.info(f"  共 {len(trajectories)} 条")

    logger.info(f"读取场景: {args.scenarios}")
    with open(args.scenarios, "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    logger.info(f"加载 Cost Model: {args.cost_model}")
    from cost_model.model import CostModel
    from core.db.knob_space import KnobSpace
    cost_model = CostModel.load(args.cost_model)
    knob_space = KnobSpace(args.knob_space)

    metrics = compute_eval_metrics(
        trajectories, scenarios, cost_model, knob_space, args.bo_trials)
    print_summary(metrics)

    output_dir = args.output or os.path.dirname(os.path.abspath(args.eval_data))
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "eval_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"报表已保存: {report_path}")


if __name__ == "__main__":
    main()
