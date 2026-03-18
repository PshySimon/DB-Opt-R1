"""
CostModel 预测接口
用法:
    from cost_model.model import CostModel
    model = CostModel.load("cost_model/checkpoints/v1/")
    tps = model.predict({"shared_buffers": "2GB", "work_mem": "64MB", ...})
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import joblib

from .preprocess import KnobPreprocessor

logger = logging.getLogger(__name__)


class CostModel:
    """Cost Model: knob 配置 → 预测 TPS"""

    def __init__(self, model, preprocessor: KnobPreprocessor, metrics: dict = None):
        self.model = model
        self.preprocessor = preprocessor
        self.metrics = metrics or {}

    @classmethod
    def load(cls, checkpoint_dir: str) -> "CostModel":
        """加载模型"""
        path = Path(checkpoint_dir)

        model = joblib.load(path / "model.pkl")
        preprocessor = KnobPreprocessor.load(checkpoint_dir)

        metrics = {}
        metrics_path = path / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

        logger.info(f"模型已加载: {checkpoint_dir}")
        if metrics:
            logger.info(f"  R²={metrics.get('r2', '?'):.4f}, "
                        f"Spearman={metrics.get('spearman', '?'):.4f}, "
                        f"MAPE={metrics.get('mape', '?'):.1%}")

        return cls(model, preprocessor, metrics)

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
        log_tps = self.model.predict(x.reshape(1, -1))[0]
        return float(np.expm1(log_tps))

    def predict_batch(self, configs: list, hw_info: dict = None) -> list:
        """批量预测"""
        X = np.array([
            self.preprocessor.transform(cfg, hw_info) for cfg in configs
        ])
        log_tps = self.model.predict(X)
        return [float(np.expm1(v)) for v in log_tps]

    def rank(self, configs: list, hw_info: dict = None) -> list:
        """
        多组配置排序，返回按预测 TPS 降序排列的 (config, tps) 列表
        """
        tps_list = self.predict_batch(configs, hw_info)
        paired = list(zip(configs, tps_list))
        paired.sort(key=lambda x: x[1], reverse=True)
        return paired


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Cost Model 预测")
    parser.add_argument("--checkpoint", required=True, help="模型目录")
    parser.add_argument("--config", required=True, help="knob 配置 JSON")
    args = parser.parse_args()

    model = CostModel.load(args.checkpoint)
    config = json.loads(args.config)
    tps = model.predict(config)
    print(f"预测 TPS: {tps:.0f}")


if __name__ == "__main__":
    main()
