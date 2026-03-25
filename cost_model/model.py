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
    """Cost Model: knob 配置 → 预测 TPS"""

    def __init__(self, ensemble: DeepEnsemble, preprocessor: KnobPreprocessor,
                 metrics: dict = None):
        self.ensemble = ensemble
        self.preprocessor = preprocessor
        self.metrics = metrics or {}

    @classmethod
    def load(cls, checkpoint_dir: str) -> "CostModel":
        """加载模型"""
        path = Path(checkpoint_dir)
        preprocessor = KnobPreprocessor.load(checkpoint_dir)

        # 加载 ensemble
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

        # 重建模型
        input_dim = save_data["input_dim"]
        ensemble.models = []
        for state_dict in save_data["model_states"]:
            model = CostMLP(
                input_dim=input_dim,
                hidden_dims=save_data["hidden_dims"],
                dropout=save_data["dropout"],
            )
            model.load_state_dict(state_dict)
            model.eval()
            ensemble.models.append(model)

        # 加载指标
        metrics = {}
        metrics_path = path / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

        logger.info(f"模型已加载: {checkpoint_dir} (Deep Ensemble × {save_data['n_models']})")
        if metrics:
            logger.info(f"  R²={metrics.get('r2', '?'):.4f}, "
                        f"Spearman={metrics.get('spearman', '?'):.4f}, "
                        f"MAPE={metrics.get('mape', '?'):.1%}")

        return cls(ensemble, preprocessor, metrics)

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
        mean, _ = self.ensemble.predict(x.reshape(1, -1))
        return float(np.expm1(mean[0]))

    def predict_with_uncertainty(self, knob_config: dict,
                                 hw_info: dict = None) -> tuple:
        """
        预测 TPS 并返回不确定性

        Returns:
            (tps, uncertainty_std)
        """
        x = self.preprocessor.transform(knob_config, hw_info)
        mean, std = self.ensemble.predict(x.reshape(1, -1))
        return float(np.expm1(mean[0])), float(std[0])

    def predict_batch(self, configs: list, hw_info: dict = None) -> list:
        """批量预测"""
        X = np.array([
            self.preprocessor.transform(cfg, hw_info) for cfg in configs
        ])
        mean, _ = self.ensemble.predict(X)
        return [float(np.expm1(v)) for v in mean]

    def predict_batch_with_uncertainty(self, configs: list,
                                       hw_info: dict = None) -> tuple:
        """批量预测，返回 (tps_list, std_list)"""
        X = np.array([
            self.preprocessor.transform(cfg, hw_info) for cfg in configs
        ])
        mean, std = self.ensemble.predict(X)
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
