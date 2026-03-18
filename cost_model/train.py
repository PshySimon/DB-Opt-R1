"""
Cost Model 训练脚本
用法: python3 -m cost_model.train --data datasets/data/cost_model/dataset.csv
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import lightgbm as lgb
from scipy import stats
from sklearn.model_selection import train_test_split, KFold
import joblib

from .preprocess import KnobPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def evaluate(y_true, y_pred, status=None):
    """评估模型，返回指标 dict"""
    metrics = {}

    # 还原到 TPS 空间
    tps_true = np.expm1(y_true)
    tps_pred = np.expm1(y_pred)

    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    metrics["r2"] = 1 - ss_res / max(ss_tot, 1e-10)

    # Spearman 相关
    spearman, _ = stats.spearmanr(y_true, y_pred)
    metrics["spearman"] = spearman

    # MAPE（仅成功数据）
    if status is not None:
        success_mask = status == "success"
        tps_true_s = tps_true[success_mask]
        tps_pred_s = tps_pred[success_mask]
    else:
        tps_true_s = tps_true
        tps_pred_s = tps_pred

    nonzero = tps_true_s > 0
    if nonzero.sum() > 0:
        mape = np.mean(np.abs(tps_true_s[nonzero] - tps_pred_s[nonzero]) / tps_true_s[nonzero])
        metrics["mape"] = mape
    else:
        metrics["mape"] = float("nan")

    # 失败识别率
    if status is not None:
        fail_mask = status != "success"
        if fail_mask.sum() > 0:
            # 预测 tps < 10 视为识别成功
            fail_identified = (tps_pred[fail_mask] < 10).sum()
            metrics["fail_recall"] = fail_identified / fail_mask.sum()
        else:
            metrics["fail_recall"] = float("nan")

    return metrics


def train(data_path: str, knob_space_path: str, output_dir: str,
          n_estimators: int = 500, max_depth: int = 8,
          learning_rate: float = 0.05, test_size: float = 0.2,
          seed: int = 42):
    """训练 Cost Model"""

    # 1. 预处理
    prep = KnobPreprocessor(knob_space_path)
    X, y, meta = prep.fit_transform(data_path)

    y_np = y.values
    status_np = meta["status"].values
    workload_np = meta["workload"].values

    # 2. 划分数据
    logger.info("=== 数据划分 ===")
    try:
        # 尝试按 workload 分层
        strat_key = np.array([f"{w}_{s}" for w, s in zip(workload_np, status_np)])
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            X, y_np, status_np,
            test_size=test_size, random_state=seed, stratify=strat_key,
        )
    except ValueError:
        # 某类样本太少，退回普通随机划分
        logger.warning("  分层采样失败（某类样本不足），使用随机划分")
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            X, y_np, status_np,
            test_size=test_size, random_state=seed,
        )
    logger.info(f"  训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

    # 3. 训练
    logger.info("=== 训练 LightGBM ===")
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        min_child_samples=5,
        random_state=seed,
        verbosity=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.log_evaluation(100)],
    )

    # 4. 评估
    logger.info("=== 测试集评估 ===")
    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred, s_test)

    logger.info(f"  MAPE:     {metrics['mape']:.1%}")
    logger.info(f"  R²:       {metrics['r2']:.4f}")
    logger.info(f"  Spearman: {metrics['spearman']:.4f}")
    if not np.isnan(metrics.get("fail_recall", float("nan"))):
        logger.info(f"  失败识别: {metrics['fail_recall']:.1%}")

    # 5. 交叉验证
    logger.info("=== 5-Fold 交叉验证 ===")
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    cv_mapes = []
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        m = lgb.LGBMRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=0.8,
            colsample_bytree=0.8, random_state=seed, verbosity=-1,
        )
        m.fit(X.iloc[tr_idx], y_np[tr_idx])
        pred = m.predict(X.iloc[val_idx])
        fold_metrics = evaluate(y_np[val_idx], pred, status_np[val_idx])
        cv_mapes.append(fold_metrics["mape"])

    cv_mean = np.nanmean(cv_mapes)
    cv_std = np.nanstd(cv_mapes)
    logger.info(f"  CV MAPE: {cv_mean:.1%} ± {cv_std:.1%}")
    metrics["cv_mape_mean"] = cv_mean
    metrics["cv_mape_std"] = cv_std

    # 6. Feature Importance
    logger.info("=== Feature Importance (Top 15) ===")
    importance = model.feature_importances_
    feat_names = prep.feature_names
    sorted_idx = np.argsort(importance)[::-1]

    importance_list = []
    for rank, idx in enumerate(sorted_idx[:15]):
        importance_list.append({
            "rank": rank + 1,
            "feature": feat_names[idx],
            "importance": int(importance[idx]),
        })
        logger.info(f"  {rank+1:2d}. {feat_names[idx]:40s} {importance[idx]}")

    # 7. 保存
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_path / "model.pkl")
    prep.save(output_dir)

    # 保存指标
    metrics_save = {k: float(v) for k, v in metrics.items()}
    metrics_save["feature_importance"] = importance_list
    metrics_save["n_train"] = int(X_train.shape[0])
    metrics_save["n_test"] = int(X_test.shape[0])
    metrics_save["n_features"] = int(X.shape[1])

    with open(output_path / "metrics.json", "w") as f:
        json.dump(metrics_save, f, indent=2, ensure_ascii=False)

    logger.info(f"\n=== 模型保存 ===")
    logger.info(f"  → {output_path / 'model.pkl'}")
    logger.info(f"  → {output_path / 'preprocessor.pkl'}")
    logger.info(f"  → {output_path / 'metrics.json'}")

    return model, prep, metrics


def main():
    parser = argparse.ArgumentParser(description="训练 Cost Model")
    parser.add_argument("--data", required=True, help="CSV 数据路径")
    parser.add_argument("--knob-space", default="configs/knob_space.yaml",
                        help="knob_space.yaml 路径")
    parser.add_argument("--output", default="cost_model/checkpoints/v1/",
                        help="输出目录")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(
        data_path=args.data,
        knob_space_path=args.knob_space,
        output_dir=args.output,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        test_size=args.test_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
