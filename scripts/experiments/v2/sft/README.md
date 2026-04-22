# v2 SFT Experiments

当前目录用于承载 `v2` SFT 实验的独立脚本。

## 目录约定

- 每个实验一个独立训练脚本：
  - `train_a0_direct_only.sh`
  - `train_a1_full3k.sh`
  - `train_b1_depth_trimmed.sh`
  - `train_b2_depth_full.sh`
  - `train_c1_gain_natural.sh`
  - `train_c2_gain_balanced.sh`
- 每个实验一个独立评估脚本：
  - `eval_a0_direct_only.sh`
  - `eval_a1_full3k.sh`
  - `eval_b1_depth_trimmed.sh`
  - `eval_b2_depth_full.sh`
  - `eval_c1_gain_natural.sh`
  - `eval_c2_gain_balanced.sh`

## 训练

这些训练脚本当前统一走：

- `scripts/train_sft_trl_lora.sh`

训练前会先自动用：

- `data_pipeline/build_sft_experiment_dataset_v2.py`

根据 `train-only` 原始数据和对应 manifest 还原出该实验的完整 `train.jsonl`。

默认输出：

- 数据集：`datasets/sft_v2/<exp>/train.jsonl`
- 模型：`model_save/experiments/v2/sft/<exp>/<timestamp>/`

## 评估

评估脚本统一使用：

- `data_pipeline/data/eval/v2/eval_trajectories_v2.jsonl`
- `data_pipeline/data/eval/v2/collected_eval_v2.json`

先跑：

- `python -m data_pipeline.synthesis.trajectory.sampler --mode eval`

再跑：

- `python -m evaluate.run`

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
