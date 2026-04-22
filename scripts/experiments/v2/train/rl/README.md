# v2 RL Experiments

当前目录放 `v2` 的 RL 训练脚本。

## 目录结构

RL 这轮只保留 `verl`：

- `verl/lora/train_frontier_1q.sh`
- `verl/lora/train_hard_1q.sh`
- `verl/lora/train_frontier_plus_hard_1q.sh`
- `verl/full/train_frontier_1q.sh`
- `verl/full/train_hard_1q.sh`
- `verl/full/train_frontier_plus_hard_1q.sh`

## 数据来源

已准备好的数据集：

- `data_pipeline/data/train/v2/rl/rl_frontier_1q.jsonl`
- `data_pipeline/data/train/v2/rl/rl_hard_1q.jsonl`
- `data_pipeline/data/train/v2/rl/rl_frontier_plus_hard_1q.jsonl`

训练前会自动转换成：

- `datasets/rl_v2/<exp>/train.parquet`
- `datasets/rl_v2/<exp>/validation.parquet`

## 默认输出

所有训练脚本默认输出到：

- `model_save/experiments/v2/rl/verl/...`

## 运行方式

模型路径支持两种方式：

1. 环境变量：

```bash
MODEL_PATH=/path/to/model bash scripts/experiments/v2/train/rl/verl/lora/train_frontier_1q.sh
```

2. 位置参数（第一个参数）：

```bash
bash scripts/experiments/v2/train/rl/verl/full/train_frontier_1q.sh /path/to/model
```

剩余参数会原样透传给：

- `scripts/train_grpo_verl_lora.sh`
- `scripts/train_grpo_verl_full.sh`

因此可以继续附加 Hydra override。

## 默认依赖

- 场景文件：
  - `data_pipeline/data/scenarios/collected/collected_8c16g_hdd_20k.json`
- cost model：
  - `cost_model/checkpoints/v10_lgbm`
