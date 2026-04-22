# v2 SFT Experiments

当前目录承载 `v2` 的 SFT 训练脚本。

## 目录结构

- 训练脚本：
  - `trl/lora/train_*.sh`
  - `trl/full/train_*.sh`
  - `verl/lora/train_*.sh`
  - `verl/full/train_*.sh`
- 评估脚本：
  - `../eval/sft/eval_a0_direct_only.sh`
  - `../eval/sft/eval_a1_full3k.sh`
  - `../eval/sft/eval_b1_depth_trimmed.sh`
  - `../eval/sft/eval_b2_depth_full.sh`
  - `../eval/sft/eval_c1_gain_natural.sh`
  - `../eval/sft/eval_c2_gain_balanced.sh`

## 训练

所有训练脚本都会先基于：

- `data_pipeline/data/train/v2/sft_trajectories_v10_train.jsonl`
- `data_pipeline/data/train/v2/manifests/*.jsonl`

自动还原实验对应的数据集：

- `trl`：还原 `train.jsonl`
- `verl`：在 `train.jsonl` 基础上进一步生成 `train.parquet / validation.parquet`

默认输出目录全部落在：

- `model_save/experiments/v2/sft/...`

每次训练都会在输出目录额外写一份：

- `train_config.json`

用于回溯：

- 模型路径
- 数据路径
- 训练超参
- 多卡配置

模型路径支持两种方式：

1. 环境变量：

```bash
MODEL_PATH=/path/to/model bash scripts/experiments/v2/train/sft/trl/lora/train_a0_direct_only.sh
```

2. 位置参数（第一个参数）：

```bash
bash scripts/experiments/v2/train/sft/verl/full/train_a0_direct_only.sh /path/to/model
```

## 多卡

### `trl`

`trl` 现在支持多卡；当 `N_GPUS>1` 时会自动走 `torchrun`。

例如：

```bash
MODEL_PATH=/path/to/model \
CUDA_DEVICES=0,1,2,3 \
N_GPUS=4 \
bash scripts/experiments/v2/train/sft/trl/lora/train_b1_depth_trimmed.sh
```

### `verl`

`verl` 本来就是多卡入口，仍然通过：

- `CUDA_VISIBLE_DEVICES`
- `N_GPUS`

控制卡数。

## 评估

评估脚本统一使用：

- `data_pipeline/data/eval/v2/eval_trajectories_v2.jsonl`
- `data_pipeline/data/eval/v2/collected_eval_v2.json`

默认输出：

- `eval_results/v2/sft/<exp>/<timestamp>/`

## 评估脚本所需环境变量

- `MODEL`：必填，服务端模型名
- `PROVIDERS_CONFIG`：可选，多中转站配置
- 若不使用 `PROVIDERS_CONFIG`，则必须提供：
  - `API_BASE`
  - `API_KEY`（默认可用 `EMPTY`）

可选：

- `COST_MODEL`
- `PARALLEL`
- `MAX_TURNS`
- `WITH_BO=true`
- `BO_TRIALS`
