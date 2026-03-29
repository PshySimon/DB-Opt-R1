"""
CostModel 预测接口
用法:
    from cost_model.model import CostModel
    model = CostModel.load("cost_model/checkpoints/v1/")
    tps = model.predict({"shared_buffers": "2GB", "work_mem": "64MB", ...})
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch

from .preprocess import KnobPreprocessor
from .train import CostMLP, DeepEnsemble

logger = logging.getLogger(__name__)


class CostModel:
    """Cost Model: knob 配置 → 预测 TPS

    支持两种 checkpoint 格式（自动检测）：
    - LightGBM: checkpoints 目录含 lgbm_model.txt
    - MLP Deep Ensemble: checkpoints 目录含 ensemble.pt
    """

    def __init__(self, model, preprocessor: KnobPreprocessor,
                 model_type: str = "lgbm", metrics: dict = None):
        self.model = model       # lgb.Booster 或 DeepEnsemble
        self.model_type = model_type
        self.preprocessor = preprocessor
        self.metrics = metrics or {}

    @classmethod
    def load(cls, checkpoint_dir: str) -> "CostModel":
        """加载模型（自动检测 LightGBM 或 MLP）"""
        path = Path(checkpoint_dir)
        preprocessor = KnobPreprocessor.load(checkpoint_dir)

        # 加载指标
        metrics = {}
        metrics_path = path / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

        if (path / "lgbm_model.txt").exists():
            # LightGBM checkpoint
            import lightgbm as lgb
            model = lgb.Booster(model_file=str(path / "lgbm_model.txt"))
            logger.info(f"已加载 LightGBM: {checkpoint_dir}")
            if metrics:
                logger.info(
                    f"  log-MAE={metrics.get('global_log_mae', metrics.get('log_mae', '?')):.4f}, "
                    f"Spearman={metrics.get('global_spearman', metrics.get('spearman', '?')):.4f}"
                )
            return cls(model, preprocessor, model_type="lgbm", metrics=metrics)

        elif (path / "ensemble.pt").exists():
            # MLP Deep Ensemble checkpoint
            save_data = torch.load(path / "ensemble.pt", map_location="cpu",
                                   weights_only=False)
            ensemble = DeepEnsemble(
                n_models=save_data["n_models"],
                hidden_dims=save_data["hidden_dims"],
                dropout=save_data["dropout"],
                device="cpu",
            )
            ensemble.scaler_X = save_data["scaler_X"]
            ensemble.scaler_y = save_data["scaler_y"]
            input_dim = save_data["input_dim"]
            ensemble.models = []
            for state_dict in save_data["model_states"]:
                m = CostMLP(
                    input_dim=input_dim,
                    hidden_dims=save_data["hidden_dims"],
                    dropout=save_data["dropout"],
                )
                m.load_state_dict(state_dict)
                m.eval()
                ensemble.models.append(m)
            logger.info(f"已加载 MLP Ensemble × {save_data['n_models']}: {checkpoint_dir}")
            return cls(ensemble, preprocessor, model_type="mlp", metrics=metrics)

        else:
            raise FileNotFoundError(
                f"checkpoint 目录 {checkpoint_dir} 中未找到 lgbm_model.txt 或 ensemble.pt"
            )


    def _infer(self, X: np.ndarray) -> tuple:
        """内部推理：返回 (mean_log, std_log)"""
        if self.model_type == "lgbm":
            pred = self.model.predict(X)
            return pred, np.zeros_like(pred)
        else:
            return self.model.predict(X)

    def predict(self, knob_config: dict, hw_info: dict = None) -> float:
        """
        预测单条配置的 TPS

        Args:
            knob_config: {"shared_buffers": "2GB", "work_mem": "64MB",
                          "workload": "mixed", ...}
            hw_info: 硬件信息，可选

        Returns:
            predicted TPS
        """
        x = self.preprocessor.transform(knob_config, hw_info)
        mean, _ = self._infer(x.reshape(1, -1))
        return float(np.expm1(mean[0]))

    def predict_with_uncertainty(self, knob_config: dict,
                                 hw_info: dict = None) -> tuple:
        """
        预测 TPS 并返回不确定性（LightGBM 返回 std=0）

        Returns:
            (tps, uncertainty_std)
        """
        x = self.preprocessor.transform(knob_config, hw_info)
        mean, std = self._infer(x.reshape(1, -1))
        return float(np.expm1(mean[0])), float(std[0])

    def predict_batch(self, configs: list, hw_info: dict = None) -> list:
        """批量预测"""
        X = np.array([
            self.preprocessor.transform(cfg, hw_info) for cfg in configs
        ])
        mean, _ = self._infer(X)
        return [float(np.expm1(v)) for v in mean]

    def predict_batch_with_uncertainty(self, configs: list,
                                       hw_info: dict = None) -> tuple:
        """批量预测，返回 (tps_list, std_list)"""
        X = np.array([
            self.preprocessor.transform(cfg, hw_info) for cfg in configs
        ])
        mean, std = self._infer(X)
        return [float(np.expm1(v)) for v in mean], [float(s) for s in std]

    def rank(self, configs: list, hw_info: dict = None) -> list:
        """
        多组配置排序，返回按预测 TPS 降序排列的 (config, tps) 列表
        """
        tps_list = self.predict_batch(configs, hw_info)
        paired = list(zip(configs, tps_list))
        paired.sort(key=lambda x: x[1], reverse=True)
        return paired


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Cost Model 预测")
    parser.add_argument("--checkpoint", required=True, help="模型目录")
    parser.add_argument("--config", required=True, help="knob 配置 JSON")
    args = parser.parse_args()

    model = CostModel.load(args.checkpoint)
    config = json.loads(args.config)
    tps, unc = model.predict_with_uncertainty(config)
    print(f"预测 TPS: {tps:.0f} (±{unc:.2f})")


if __name__ == "__main__":
    main()
