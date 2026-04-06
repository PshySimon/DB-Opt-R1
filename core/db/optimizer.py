"""
Knob 配置优化器

提供基于 optuna（TPE）的 knob 配置搜索，接收 KnobSpace 作为参数。
"""

import logging
from core.db.knob_space import KnobSpace, parse_memory, format_memory

logger = logging.getLogger(__name__)


def optuna_search(knob_space: KnobSpace, objective_fn,
                  n_trials: int = 200, direction: str = "maximize") -> tuple:
    """用 optuna（TPE）搜索最优 knob 配置。

    Args:
        knob_space: KnobSpace 实例
        objective_fn: callable(knobs_dict) -> float，评价函数
        n_trials: 搜索轮数
        direction: "maximize" 或 "minimize"

    Returns:
        (best_value, best_knobs_dict)
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def trial_to_knobs(trial):
        knobs = {}
        for name, spec in knob_space.knobs.items():
            ktype = spec.get("type", "integer")
            lo = spec.get("min", 0)
            hi = spec.get("max", 1000)
            default = spec.get("default", lo)

            if ktype == "memory":
                lo_kb = parse_memory(str(lo))
                hi_kb = parse_memory(str(hi))
                val_kb = trial.suggest_int(name, lo_kb, hi_kb)
                knobs[name] = format_memory(val_kb)
            elif ktype == "integer":
                knobs[name] = trial.suggest_int(name, int(lo), int(hi))
            elif ktype == "float":
                knobs[name] = trial.suggest_float(name, float(lo), float(hi))
            elif ktype == "enum":
                knobs[name] = trial.suggest_categorical(
                    name, spec.get("values", [default]))
            else:
                knobs[name] = default
        return knobs

    def objective(trial):
        knobs = trial_to_knobs(trial)
        try:
            return objective_fn(knobs)
        except Exception:
            return 0.0 if direction == "maximize" else float("inf")

    study = optuna.create_study(direction=direction)

    # 默认配置作为起始点
    default_params = {}
    for name, spec in knob_space.knobs.items():
        ktype = spec.get("type", "integer")
        default = spec.get("default", spec.get("min", 0))
        if ktype == "memory":
            default_params[name] = parse_memory(str(default))
        elif ktype == "integer":
            default_params[name] = int(default)
        elif ktype == "float":
            default_params[name] = float(default)
        elif ktype == "enum":
            default_params[name] = str(default)
        else:
            default_params[name] = default
    study.enqueue_trial(default_params)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # 还原最优 knobs
    best_knobs = {}
    for name, spec in knob_space.knobs.items():
        ktype = spec.get("type", "integer")
        val = study.best_params[name]
        if ktype == "memory":
            best_knobs[name] = format_memory(int(val))
        else:
            best_knobs[name] = val

    return study.best_value, best_knobs
