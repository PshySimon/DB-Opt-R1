"""
Cost Model 训练脚本 — Deep Ensemble (多 MLP)
用法: python3 -m cost_model.train --data datasets/data/scenarios/collected.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from .preprocess import KnobPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ===== MLP 定义 =====

class CostMLP(nn.Module):
    """单个 MLP，带残差连接和 dropout"""

    def __init__(self, input_dim: int, hidden_dims: list = None, dropout: float = 0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

        # 残差投影（如果维度不同）
        self.residual_proj = nn.Linear(input_dim, 1) if input_dim != 1 else None

    def forward(self, x):
        out = self.net(x)
        if self.residual_proj is not None:
            out = out + 0.1 * self.residual_proj(x)
        return out.squeeze(-1)


class DeepEnsemble:
    """Deep Ensemble: N 个独立训练的 MLP"""

    def __init__(self, n_models: int = 5, hidden_dims: list = None,
                 dropout: float = 0.1, device: str = None):
        self.n_models = n_models
        self.hidden_dims = hidden_dims or [256, 128, 64]
        self.dropout = dropout
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.models = []
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs: int = 200, batch_size: int = 256, lr: float = 1e-3,
            weight_decay: float = 1e-4, patience: int = 20,
            sample_weights: np.ndarray = None):
        """训练 ensemble 中的所有模型"""

        # 标准化
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        X_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y_train_scaled).to(self.device)

        # 样本权重
        if sample_weights is not None:
            w_tensor = torch.FloatTensor(sample_weights).to(self.device)
        else:
            w_tensor = None

        if X_val is not None:
            X_val_scaled = self.scaler_X.transform(X_val)
            y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1)).ravel()
            X_val_t = torch.FloatTensor(X_val_scaled).to(self.device)
            y_val_t = torch.FloatTensor(y_val_scaled).to(self.device)

        input_dim = X_train.shape[1]
        self.models = []

        for i in range(self.n_models):
            logger.info(f"  训练模型 {i+1}/{self.n_models}...")

            # 每个模型用不同的随机种子
            torch.manual_seed(42 + i * 1000)
            np.random.seed(42 + i * 1000)

            model = CostMLP(
                input_dim=input_dim,
                hidden_dims=self.hidden_dims,
                dropout=self.dropout,
            ).to(self.device)

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=lr * 0.01
            )

            if w_tensor is not None:
                dataset = TensorDataset(X_tensor, y_tensor, w_tensor)
            else:
                dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            best_val_loss = float("inf")
            best_state = None
            no_improve = 0

            for epoch in range(epochs):
                model.train()
                train_loss = 0
                for batch in loader:
                    if w_tensor is not None:
                        xb, yb, wb = batch
                    else:
                        xb, yb = batch
                        wb = None
                    optimizer.zero_grad()
                    pred = model(xb)
                    # 加权 Huber Loss
                    huber = nn.functional.huber_loss(pred, yb, reduction='none', delta=1.0)
                    if wb is not None:
                        loss = (huber * wb).mean()
                    else:
                        loss = huber.mean()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    train_loss += loss.item() * xb.size(0)
                train_loss /= len(dataset)
                scheduler.step()

                # 验证
                if X_val is not None:
                    model.eval()
                    with torch.no_grad():
                        val_pred = model(X_val_t)
                        val_loss = nn.functional.huber_loss(val_pred, y_val_t, delta=1.0).item()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        no_improve = 0
                    else:
                        no_improve += 1

                    if (epoch + 1) % 50 == 0:
                        logger.info(
                            f"    Epoch {epoch+1}/{epochs}: "
                            f"train={train_loss:.4f}, val={val_loss:.4f}"
                        )

                    if no_improve >= patience:
                        logger.info(f"    Early stop at epoch {epoch+1}")
                        break
                else:
                    if (epoch + 1) % 50 == 0:
                        logger.info(f"    Epoch {epoch+1}/{epochs}: train={train_loss:.4f}")

            # 恢复最佳权重
            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()
            model.cpu()
            self.models.append(model)

        logger.info(f"  Ensemble 训练完成: {self.n_models} 个模型")

    def predict(self, X: np.ndarray) -> tuple:
        """
        预测，返回 (mean, std)

        Returns:
            mean: 均值预测 (n_samples,)
            std: 不确定性估计 (n_samples,)
        """
        X_scaled = self.scaler_X.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)

        preds = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(X_tensor).numpy()
            preds.append(pred)

        preds = np.array(preds)  # (n_models, n_samples)

        # 反标准化
        mean_scaled = preds.mean(axis=0)
        std_scaled = preds.std(axis=0)

        mean = self.scaler_y.inverse_transform(mean_scaled.reshape(-1, 1)).ravel()
        # std 只需要乘以 scale（不加 mean）
        std = std_scaled * self.scaler_y.scale_[0]

        return mean, std


# ===== 评估 =====

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

    # WAPE（加权绝对百分误差）
    total_actual = np.sum(np.abs(tps_true_s[nonzero]))
    if total_actual > 0:
        metrics["wape"] = np.sum(np.abs(tps_true_s[nonzero] - tps_pred_s[nonzero])) / total_actual
    else:
        metrics["wape"] = float("nan")

    # log-MAE（log 空间的平均绝对误差）
    metrics["log_mae"] = np.mean(np.abs(y_true - y_pred))

    # 失败识别率
    if status is not None:
        fail_mask = status != "success"
        if fail_mask.sum() > 0:
            fail_identified = (tps_pred[fail_mask] < 10).sum()
            metrics["fail_recall"] = fail_identified / fail_mask.sum()
        else:
            metrics["fail_recall"] = float("nan")

    return metrics


# ===== 训练入口 =====

def train(data_path: str, knob_space_path: str, output_dir: str,
          n_models: int = 5, hidden_dims: str = "512,256,128",
          epochs: int = 200, batch_size: int = 256,
          lr: float = 1e-3, dropout: float = 0.1,
          test_size: float = 0.2, seed: int = 42,
          split_by_source: bool = False,
          model_type: str = "mlp"):
    """训练 Cost Model

    Args:
        model_type: 'mlp' (Deep Ensemble) 或 'lgbm' (LightGBM)
    """

    hidden = [int(x) for x in hidden_dims.split(",")]

    # 1. 预处理
    prep = KnobPreprocessor(knob_space_path)
    X, y, meta = prep.fit_transform(data_path)

    y_np = y.values
    status_np = meta["status"].values
    workload_np = meta["workload"].values
    source_np = meta["source"].values

    # 2. 划分数据
    if split_by_source:
        logger.info("=== 按 source 划分数据 ===")
        random_mask = source_np == "random_sampled"
        llm_mask = source_np == "llm_generated"

        n_random = random_mask.sum()
        n_llm = llm_mask.sum()
        logger.info(f"  random_sampled: {n_random} 条")
        logger.info(f"  llm_generated:  {n_llm} 条")

        if n_llm == 0:
            logger.warning("  无 llm_generated 数据，退回随机划分")
            split_by_source = False

    if split_by_source:
        # random → 训练 + 验证（in-distribution）
        X_random = X.values[random_mask]
        y_random = y_np[random_mask]
        s_random = status_np[random_mask]

        X_train, X_val_id, y_train, y_val_id, s_train, s_val_id = train_test_split(
            X_random, y_random, s_random,
            test_size=test_size, random_state=seed,
        )

        # llm_generated → OOD 测试集
        X_test_ood = X.values[llm_mask]
        y_test_ood = y_np[llm_mask]
        s_test_ood = status_np[llm_mask]

        logger.info(f"  训练集 (random):       {X_train.shape[0]}")
        logger.info(f"  验证集 (random hold-out): {X_val_id.shape[0]}")
        logger.info(f"  测试集 (llm OOD):      {X_test_ood.shape[0]}")

        # 用 random hold-out 做 early stopping 验证集
        X_val = X_val_id
        y_val = y_val_id
    else:
        logger.info("=== 随机划分数据 ===")
        try:
            strat_key = np.array([f"{w}_{s}" for w, s in zip(workload_np, status_np)])
            X_train, X_val, y_train, y_val, s_train, s_val_id = train_test_split(
                X.values, y_np, status_np,
                test_size=test_size, random_state=seed, stratify=strat_key,
            )
        except ValueError:
            logger.warning("  分层采样失败，使用随机划分")
            X_train, X_val, y_train, y_val, s_train, s_val_id = train_test_split(
                X.values, y_np, status_np,
                test_size=test_size, random_state=seed,
            )
        logger.info(f"  训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}")

    # 2.5 计算逆频率样本权重（按 TPS 分桶）
    tps_train = np.expm1(y_train)
    bin_edges = [0, 100, 1000, 10000, float("inf")]
    bin_indices = np.digitize(tps_train, bin_edges) - 1  # 0-indexed
    n_bins = len(bin_edges) - 1
    bin_counts = np.array([np.sum(bin_indices == b) for b in range(n_bins)])
    bin_weights = len(tps_train) / (n_bins * np.maximum(bin_counts, 1))
    sample_weights = np.array([bin_weights[b] for b in bin_indices], dtype=np.float32)
    # 归一化使均值=1
    sample_weights = sample_weights / sample_weights.mean()
    logger.info(f"  样本加权: bins={list(bin_counts)}, weights={[f'{w:.2f}' for w in bin_weights]}")

    # 3. 训练
    y_pred_std = np.zeros(len(y_val))  # LightGBM 无不确定性

    if model_type == "lgbm":
        import lightgbm as lgb
        logger.info(f"=== 训练 LightGBM ===")

        dtrain = lgb.Dataset(X_train, y_train, weight=sample_weights)
        dval = lgb.Dataset(X_val, y_val, reference=dtrain)

        lgb_params = {
            'objective': 'huber', 'metric': 'mae', 'alpha': 1.0,
            'num_leaves': 63, 'learning_rate': 0.05,
            'min_child_samples': 20, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'verbose': -1, 'seed': seed,
        }
        gbm = lgb.train(lgb_params, dtrain, num_boost_round=1000,
                        valid_sets=[dval],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])

        logger.info(f"  最佳迭代: {gbm.best_iteration}")
    else:
        logger.info(f"=== 训练 Deep Ensemble ({n_models} × MLP {hidden}) ===")
        ensemble = DeepEnsemble(
            n_models=n_models,
            hidden_dims=hidden,
            dropout=dropout,
        )
        ensemble.fit(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            epochs=epochs, batch_size=batch_size,
            lr=lr,
            sample_weights=sample_weights,
        )

    # 4. 评估
    def _predict(X_data):
        if model_type == "lgbm":
            return gbm.predict(X_data), np.zeros(len(X_data))
        else:
            return ensemble.predict(X_data)

    all_metrics = {}

    if split_by_source:
        # In-Distribution 评估（random hold-out）
        logger.info("=== In-Distribution 评估 (random hold-out) ===")
        y_pred_id, y_std_id = _predict(X_val_id)
        metrics_id = evaluate(y_val_id, y_pred_id, s_val_id)
        logger.info(f"  WAPE:     {metrics_id['wape']:.1%}")
        logger.info(f"  MAPE:     {metrics_id['mape']:.1%}")
        logger.info(f"  log-MAE:  {metrics_id['log_mae']:.4f}")
        logger.info(f"  R²:       {metrics_id['r2']:.4f}")
        logger.info(f"  Spearman: {metrics_id['spearman']:.4f}")
        all_metrics["in_distribution"] = {k: float(v) for k, v in metrics_id.items()}

        # OOD 评估（llm_generated）
        logger.info("=== OOD 评估 (llm_generated) ===")
        y_pred_ood, y_std_ood = _predict(X_test_ood)
        metrics_ood = evaluate(y_test_ood, y_pred_ood, s_test_ood)
        logger.info(f"  WAPE:     {metrics_ood['wape']:.1%}")
        logger.info(f"  MAPE:     {metrics_ood['mape']:.1%}")
        logger.info(f"  log-MAE:  {metrics_ood['log_mae']:.4f}")
        logger.info(f"  R²:       {metrics_ood['r2']:.4f}")
        logger.info(f"  Spearman: {metrics_ood['spearman']:.4f}")
        all_metrics["ood"] = {k: float(v) for k, v in metrics_ood.items()}

        metrics = metrics_ood
    else:
        logger.info("=== 验证集评估 ===")
        y_pred_mean, y_pred_std = _predict(X_val)
        metrics = evaluate(y_val, y_pred_mean, s_val_id)

        logger.info(f"  WAPE:     {metrics['wape']:.1%}")
        logger.info(f"  MAPE:     {metrics['mape']:.1%}")
        logger.info(f"  log-MAE:  {metrics['log_mae']:.4f}")
        logger.info(f"  R²:       {metrics['r2']:.4f}")
        logger.info(f"  Spearman: {metrics['spearman']:.4f}")
        if not np.isnan(metrics.get("fail_recall", float("nan"))):
            logger.info(f"  失败识别: {metrics['fail_recall']:.1%}")
        if model_type != "lgbm":
            logger.info(f"  不确定性: mean={y_pred_std.mean():.4f}, max={y_pred_std.max():.4f}")

        # 分桶 WAPE
        tps_true_val = np.expm1(y_val)
        tps_pred_val = np.expm1(y_pred_mean)
        bins_eval = [(0, 100, '<100'), (100, 1000, '100-1K'), (1000, 10000, '1K-10K'), (10000, 1e9, '10K+')]
        logger.info("  分桶 WAPE:")
        for lo, hi, label in bins_eval:
            m = (tps_true_val >= lo) & (tps_true_val < hi)
            if m.sum() > 0:
                w = np.sum(np.abs(tps_true_val[m] - tps_pred_val[m])) / np.sum(np.abs(tps_true_val[m]))
                logger.info(f"    [{label:>6s}]: {w:.1%} (n={m.sum()})")

    # 5. 保存
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if model_type == "lgbm":
        gbm.save_model(str(output_path / "lgbm_model.txt"))
    else:
        save_data = {
            "n_models": n_models,
            "hidden_dims": hidden,
            "dropout": dropout,
            "input_dim": X_train.shape[1],
            "scaler_X": ensemble.scaler_X,
            "scaler_y": ensemble.scaler_y,
            "model_states": [m.state_dict() for m in ensemble.models],
        }
        torch.save(save_data, output_path / "ensemble.pt")
    prep.save(output_dir)

    metrics_save = {k: float(v) for k, v in metrics.items()}
    metrics_save["model_type"] = model_type
    metrics_save["n_train"] = int(X_train.shape[0])
    metrics_save["n_features"] = int(X_train.shape[1])
    metrics_save["split_by_source"] = split_by_source
    if model_type == "mlp":
        metrics_save["n_models"] = n_models
        metrics_save["hidden_dims"] = hidden
    if split_by_source:
        metrics_save["n_val_id"] = int(X_val_id.shape[0])
        metrics_save["n_test_ood"] = int(X_test_ood.shape[0])
        metrics_save["detail"] = all_metrics
    else:
        metrics_save["n_test"] = int(X_val.shape[0])

    with open(output_path / "metrics.json", "w") as f:
        json.dump(metrics_save, f, indent=2, ensure_ascii=False)

    logger.info(f"\n=== 模型保存 ===")
    logger.info(f"  → {output_path}")
    logger.info(f"  → {output_path / 'metrics.json'}")

    return None, prep, metrics


def main():
    parser = argparse.ArgumentParser(description="训练 Cost Model (Deep Ensemble)")
    parser.add_argument("--data", required=True, help="数据路径（CSV 或 JSON）")
    parser.add_argument("--knob-space", default="configs/knob_space.yaml",
                        help="knob_space.yaml 路径")
    parser.add_argument("--output", default="cost_model/checkpoints/v1/",
                        help="输出目录")
    parser.add_argument("--n-models", type=int, default=5,
                        help="Ensemble 中 MLP 数量")
    parser.add_argument("--hidden-dims", default="512,256,128",
                        help="MLP 隐层维度，逗号分隔")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-by-source", action="store_true",
                        help="按 source 拆分: random→训练, llm→OOD 测试")
    parser.add_argument("--model", default="mlp", choices=["mlp", "lgbm"],
                        help="模型类型: mlp (Deep Ensemble) 或 lgbm (LightGBM)")

    args = parser.parse_args()
    train(
        data_path=args.data,
        knob_space_path=args.knob_space,
        output_dir=args.output,
        n_models=args.n_models,
        hidden_dims=args.hidden_dims,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
        test_size=args.test_size,
        seed=args.seed,
        split_by_source=args.split_by_source,
        model_type=args.model,
    )


if __name__ == "__main__":
    main()
