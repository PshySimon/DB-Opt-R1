# DB-Opt-R1

基于 GRPO 强化学习训练 LLM Agent 进行 PostgreSQL 自动调优。

## 思路

```
                  ┌─────────────┐
                  │   LLM Agent │  ← GRPO 训练
                  └──────┬──────┘
                         │ 调用工具
              ┌──────────┼──────────┐
              ▼          ▼          ▼
         观察工具     行动工具     验证工具
       (硬件/指标)   (set_knob)  (cost model)
              │          │          │
              └──────────┼──────────┘
                         ▼
                    模拟环境 / 真实 PG
```

1. **数据采集**：在真实 PG 上随机采样 knob 配置 + 跑 benchmark，收集性能数据
2. **Cost Model**：用采集的数据训练性能预测模型（LightGBM）
3. **SFT 数据合成**：纯轨迹采样（N×rollout / 场景，保留 improvement_pct > 3% 的轨迹）；MCTS 作为对比基线保留
4. **GRPO 训练**：LLM Agent 通过工具与模拟环境交互，Cost Model 预测性能作为 reward
5. **评估**：在真实 PG 上验证 Agent 的调优效果

## 项目结构

```
├── core/                               # 核心框架
│   ├── config.py                       # 统一配置管理
│   ├── tool/                           # 工具框架（Tool 基类、ToolEnv、装饰器）
│   └── db/                             # PG 公共基础设施
├── environment/
│   └── tools/                          # 12 个 DB 调优工具（real/simulated 双模式）
├── cost_model/                         # Cost Model（LightGBM）
├── datasets/                           # 数据采集 + 轨迹合成
│   └── synthesis/
│       ├── scenarios/                  # 场景生成、采集、迁移脚本
│       │   └── migrate_add_questions.py  # 为存量数据补充 question 字段
│       ├── trajectory/                 # 纯轨迹采样器（推荐）
│       │   └── sampler.py
│       └── mcts/                       # MCTS（保留为对比基线）
├── training/                           # 训练代码
│   ├── reward_score.py                 # Reward 计算（共享）
│   ├── data_utils.py                   # 数据加载工具（共享）
│   ├── trl/                            # trl 后端（不依赖 vLLM/Ray）
│   │   ├── sft.py                      # SFT 训练
│   │   └── grpo.py                     # GRPO 训练
│   └── verl/                           # verl 后端（需要 vLLM + Ray）
│       ├── main_grpo.py                # GRPO 入口
│       └── agent_ray_trainer.py        # 训练循环
├── configs/
│   ├── config.yaml                     # 全局配置
│   ├── knob_space.yaml                 # 45 个可调 knob 定义
│   └── knob_effects.yaml               # Knob 效果知识库
├── scripts/
│   ├── train_sft_trl.sh                # SFT（trl 后端）
│   ├── train_grpo_trl.sh               # GRPO（trl 后端）
│   ├── train_sft_verl.sh               # SFT（verl 后端）
│   ├── train_grpo_verl.sh              # GRPO（verl 后端）
│   └── setup_env.sh                    # 环境准备
├── requirements.txt                    # 基础依赖
├── requirements-trl.txt                # trl 后端依赖
└── requirements-verl.txt               # verl 后端依赖
```

## 环境准备

**系统要求**：Ubuntu 22.04+，Python 3.8+

```bash
# 一键安装 PostgreSQL + 初始化
bash scripts/setup_env.sh

# 安装 Python 依赖（选一个后端）
pip install -r requirements-trl.txt   # trl 后端（不依赖 vLLM/Ray，壁仞 GPU 推荐）
pip install -r requirements-verl.txt  # verl 后端（需要 vLLM + Ray）
```

---

## 使用流程

### 统一数据 Pipeline（Step 0 → 5）

所有数据采集、Cost Model 训练、MCTS 轨迹合成合并为一条流水线：

#### Step 1: 维度组合蒸馏生成 knob 配置

基于 `synthesis_dimensions.yaml` 中定义的 58 个场景模板（单瓶颈/组合瓶颈/应用场景/反模式/边界条件），按 **场景 × 负载类型 × 严重程度** 的笛卡尔积，让 LLM 系统性生成 knob 配置。默认 `--per-cell 5`，共生成 **~3,000 条**配置。

```bash
# 方式 1：多中转站轮询（推荐，高并发高吞吐）
python3 -m datasets.synthesis.scenarios.pipeline synthesize \
    --dimensions configs/synthesis_dimensions.yaml \
    --knob-space configs/knob_space.yaml \
    --output datasets/data/scenarios/knob_configs_synth.json \
    --per-cell 5 --workers 8 \
    --providers-config configs/providers.json

# 方式 2：单一 API 端点
python3 -m datasets.synthesis.scenarios.pipeline synthesize \
    --dimensions configs/synthesis_dimensions.yaml \
    --knob-space configs/knob_space.yaml \
    --output datasets/data/scenarios/knob_configs_synth.json \
    --per-cell 5 --workers 5 --model gpt-5 \
    --api-key $OPENAI_API_KEY \
    --api-base $OPENAI_API_BASE
```

#### Step 2: 随机采样 knob 配置（Cost Model 数据）

纯本地生成，不需要 LLM。采样策略 `mixed` = 40% 随机 + 40% 默认值附近 + 20% LHS。

```bash
python3 -m datasets.synthesis.scenarios.pipeline random-sample \
    --knob-space configs/knob_space.yaml \
    --output datasets/data/scenarios/knob_configs_random.json \
    --count 5000 --strategy mixed
```

#### Step 3: 统一真机采集（需要 PG）

`collect` 支持 glob，自动合并所有 `knob_configs_*.json`（LLM 生成 + 随机采样），一次采集。

```bash
# 后台执行
mkdir -p logs/scenarios
nohup python3 -m datasets.synthesis.scenarios.pipeline collect \
    --input "datasets/data/scenarios/knob_configs_*.json" \
    --output datasets/data/scenarios/collected.json \
    --host 127.0.0.1 --port 5432 \
    --user postgres --database benchmark \
    > logs/scenarios/$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > logs/scenarios/running.pid
```

数据带 `source` 标签：`llm_generated`（MCTS + Cost Model）、`random_sampled`（仅 Cost Model）。

#### Step 3.5: 贝叶斯优化搜索好配置（可选，需要 PG）

用已有 collected 数据热启动高斯过程代理模型，对每种负载跑 30 轮 BO，搜索中高 TPS 配置。搜索轨迹 ~120 条 + 围绕最优配置的扰动变体 ~200 条。

```bash
mkdir -p logs/scenarios
nohup python3 -m datasets.synthesis.scenarios.pipeline bo-search \
    --knob-space configs/knob_space.yaml \
    --collected "datasets/data/scenarios/collected*.json" \
    --output datasets/data/scenarios/collected_bo.json \
    --rounds 30 --n-perturb 50 \
    --host 127.0.0.1 --port 5432 \
    --user postgres --database benchmark \
    > logs/scenarios/bo_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > logs/scenarios/bo.pid
```

BO 产出的扰动变体（`source=bo_perturb`）需要再走一遍 `collect` 采集真实 TPS。

#### Step 4: Cost Model 训练

使用全量 `collected.json` 数据训练 Cost Model。

```bash
python3 -m cost_model.train \
    --data datasets/data/scenarios/collected.json \
    --output cost_model/checkpoints/v1
```

#### Step 5: SFT 轨迹合成（纯轨迹采样，推荐）

对每个场景并行跑 N 次独立 rollout，Cost Model 计算 `improvement_pct`，保留提升 > 3% 的轨迹作为 SFT 正样本。

**前置：场景 question 字段**

每个 `ScenarioState` 包含一个 `question` 字段（用户自然语言问题，基于采集到的真实 DB 症状生成）。存量数据请先运行迁移脚本：

```bash
python3 -m datasets.synthesis.scenarios.migrate_add_questions \
    --input datasets/data/scenarios/ \
    --model gpt-5 \
    --api-key $OPENAI_API_KEY \
    --api-base $OPENAI_API_BASE \
    --workers 10
```

**轨迹采样：**

```bash
python3 -m datasets.synthesis.trajectory.sampler \
    --scenarios datasets/data/scenarios/ \
    --cost-model cost_model/checkpoints/v7_lgbm_dedup \
    --output-dir datasets/data/trajectory/ \
    --num-rollouts 8 \
    --good-threshold 0.03 \
    --temperature 0.7 \
    --parallel 8 \
    --model gpt-5 \
    --api-key $OPENAI_API_KEY \
    --api-base $OPENAI_API_BASE
```

| 参数 | 说明 | 默认 |
|------|------|------|
| `--num-rollouts` | 每场景 rollout 次数 | 8 |
| `--good-threshold` | SFT 正样本阈值（improvement_pct / 100） | 0.03 (3%) |
| `--temperature` | LLM 采样温度 | 0.7 |
| `--parallel` | 并发 worker 数 | 4 |

输出：`datasets/data/trajectory/sft_trajectories.jsonl`

#### Step 5（备选）: MCTS 轨迹合成（保留为对比基线）

MCTS 作为对比基线保留，不再主动维护。每轮搜索输出到带时间戳的子目录。

```bash
python3 -m datasets.synthesis.mcts.run_search \
    --scenarios datasets/data/scenarios/ \
    --knob-space configs/knob_space.yaml \
    --cost-model cost_model/checkpoints/v7_lgbm_dedup \
    --output-dir datasets/data \
    --providers-config configs/providers.json \
    --parallel 4 --num-workers 2 --simulations 5
```

> **注意**：MCTS 也依赖 `scenario.question` 字段，运行前同样需要先执行迁移脚本。

输出（在 `datasets/data/run_YYYYMMDD_HHMMSS/` 下）：
- `sft_trajectories.jsonl` — SFT 训练数据
- `contrastive_pairs.jsonl` — DPO/对比学习数据
- `mcts_trees/` — 搜索树 debug 文件

#### Step 5.5: 评估集 knob 配置合成

复用 Step 1 的维度组合蒸馏，但 `--per-cell 1`（每格仅 1 条），输出到独立文件，确保与训练集零重叠。合成后无需真机采集，评估时直接用 Cost Model 模拟环境。

```bash
python3 -m datasets.synthesis.scenarios.pipeline synthesize \
    --dimensions configs/synthesis_dimensions.yaml \
    --knob-space configs/knob_space.yaml \
    --output datasets/data/scenarios/knob_configs_eval.json \
    --per-cell 1 --workers 8 \
    --providers-config configs/providers.json
```

产出约 ~696 条配置，后续评估脚本会读取该文件构建模拟环境。

#### Step 6: SFT 训练

将 MCTS 轨迹转为训练格式，进行 SFT 冷启动训练。支持 **trl** 和 **verl** 两种后端。

```bash
# === trl 后端（推荐，不依赖 vLLM/Ray）===

# 方式 1：用脚本
BASE_MODEL=~/models/Qwen2.5-3B-Instruct \
DATA_FILES="datasets/data/run_*/sft_trajectories.jsonl" \
bash scripts/train_sft_trl.sh

# 方式 2：直接调用
python -m training.trl.sft \
    --model_path ~/models/Qwen2.5-3B-Instruct \
    --data_files datasets/data/run_*/sft_trajectories.jsonl \
    --output_dir model_save/sft/

# === verl 后端（需要 vLLM + Ray）===

# 数据预处理（JSONL → Parquet）
python3 -m datasets.preprocess_sft \
    --input_files datasets/data/run_*/sft_trajectories.jsonl \
    --output_dir datasets/sft/

# SFT 训练
BASE_MODEL=Qwen/Qwen2.5-3B-Instruct \
bash scripts/train_sft_verl.sh
```

训练完成后 checkpoint 保存在 `model_save/sft/`。

#### Step 7: GRPO 训练

基于 SFT checkpoint 进行 GRPO 强化学习训练。

```bash
# === trl 后端 ===
python -m training.trl.grpo \
    --model_path model_save/sft/ \
    --scenario_dir datasets/data/scenarios/ \
    --cost_model cost_model/checkpoints/v1 \
    --output_dir model_save/grpo/

# === verl 后端 ===
bash scripts/train_grpo_verl.sh
```

#### Step 8: 模型评估

使用 Step 5.5 合成的评估集，通过 Cost Model 模拟环境，评估模型的调优能力。支持 API 和 vLLM 两种推理后端。

> **前置**：评估场景的 `question` 字段必须已填充（运行迁移脚本），否则会报错。

```bash
python3 -m evaluate.run \
    --eval-scenarios datasets/data/scenarios/knob_configs_eval.json \
    --scenarios datasets/data/scenarios/ \
    --knob-space configs/knob_space.yaml \
    --cost-model cost_model/checkpoints/v7_lgbm_dedup \
    --api-key $OPENAI_API_KEY \
    --api-base $OPENAI_API_BASE \
    --model gpt-5 \
    --output eval_results/baseline_gpt5/ \
    --max-turns 10 \
    --parallel 8
```

评估指标输出到 `eval_results/` 下的 `report.json`，包含：
- **avg_reward** — 平均 reward（Cost Model 预测的 TPS 提升比例）
- **format_pass_rate** — 工具调用格式合规率
- **effective_rate** — 有效工具调用率
- **avg_steps** — 平均交互轮数
- **completion_rate** — 成功完成率（触发 `predict_performance`）



## 配置

编辑 `configs/config.yaml`：

```yaml
tools:
  database:
    host: "127.0.0.1"
    port: 5432
    user: "postgres"
    password: "postgres"
    database: "benchmark"
```

## 参考项目

- [Agent-R1](https://github.com/0russwest0/Agent-R1) — 工具框架参考
- [DBTune](https://github.com/PKU-DAIR/DBTune) — DB 调优方法参考
- [GPTuner](https://github.com/SolidLao/GPTuner) — LLM + 代理模型调优

## License

MIT
