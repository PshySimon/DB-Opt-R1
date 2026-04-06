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
│   ├── llm/                            # 多中转站 LLM 客户端
│   ├── tool/                           # 工具框架（Tool 基类、ToolEnv、装饰器）
│   └── db/                             # PG 公共基础设施
├── environment/
│   └── tools/                          # 12 个 DB 调优工具（real/simulated 双模式）
├── cost_model/                         # Cost Model（LightGBM）
├── datasets/
│   ├── data/
│   │   ├── scenarios/                  # 场景数据
│   │   │   ├── knobs/                  # knob_configs_*.json
│   │   │   ├── collected/              # collected_*.json（真机采集数据）
│   │   │   └── seeds/                  # seeds.json
│   │   ├── train/                      # 训练数据集
│   │   │   ├── sft_trajectories.jsonl
│   │   │   └── contrastive_pairs.jsonl
│   │   └── eval/                       # 评估数据集
│   │       └── eval_scenarios.jsonl
│   └── synthesis/                      # 数据合成脚本
│       ├── scenarios/                  # 场景生成、采集、迁移
│       ├── trajectory/                 # 纯轨迹采样器（推荐）
│       └── mcts/                       # MCTS（保留为对比基线）
├── evaluate/                           # 评估脚本
├── training/                           # 训练代码
│   ├── trl/                            # trl 后端（不依赖 vLLM/Ray）
│   │   ├── sft.py                      # SFT 训练
│   │   └── grpo.py                     # GRPO 训练
│   └── verl/                           # verl 后端（需要 vLLM + Ray）
├── configs/
│   ├── config.yaml                     # 全局配置
│   ├── knob_space.yaml                 # 45 个可调 knob 定义
│   ├── knob_effects.yaml               # Knob 效果知识库
│   └── providers.json                  # 多中转站 API 配置
└── scripts/                            # 训练/环境准备脚本
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

基于 `synthesis_dimensions.yaml` 中定义的 58 个场景模板（单瓶颈/组合瓶颈/应用场景/反模式/边界条件），按 **场景 × 负载类型 × 严重程度** 的笛卡尔积，让 LLM 系统性生成 knob 配置。

默认 `--per-cell 5`，共生成 **~3,000 条**配置。

```bash
# 方式 1：多中转站轮询（推荐，高并发高吞吐）
python3 -m data_pipeline.synthesis.scenarios.pipeline synthesize \
    --dimensions configs/synthesis_dimensions.yaml \
    --knob-space configs/knob_space.yaml \
    --output data_pipeline/data/scenarios/knobs/knob_configs_synth.json \
    --per-cell 5 --workers 8 \
    --providers-config configs/providers.json

# 方式 2：单一 API 端点
python3 -m data_pipeline.synthesis.scenarios.pipeline synthesize \
    --dimensions configs/synthesis_dimensions.yaml \
    --knob-space configs/knob_space.yaml \
    --output data_pipeline/data/scenarios/knobs/knob_configs_synth.json \
    --per-cell 5 --workers 5 --model gpt-5 \
    --api-key $OPENAI_API_KEY \
    --api-base $OPENAI_API_BASE
```

#### Step 2: 随机采样 knob 配置（Cost Model 数据）

纯本地生成，不需要 LLM。采样策略 `mixed` = 40% 随机 + 40% 默认值附近 + 20% LHS。

```bash
python3 -m data_pipeline.synthesis.scenarios.pipeline random-sample \
    --knob-space configs/knob_space.yaml \
    --output data_pipeline/data/scenarios/knobs/knob_configs_random.json \
    --count 5000 --strategy mixed
```

#### Step 3: 统一真机采集（需要 PG）

`collect` 支持 glob，自动合并所有 `knob_configs_*.json`（LLM 生成 + 随机采样），一次采集。

```bash
# 后台执行
mkdir -p logs/scenarios
nohup python3 -m data_pipeline.synthesis.scenarios.pipeline collect \
    --input "data_pipeline/data/scenarios/knobs/knob_configs_*.json" \
    --output data_pipeline/data/scenarios/collected/collected.json \
    --host 127.0.0.1 --port 5432 \
    --user postgres --database benchmark \
    > logs/scenarios/$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > logs/scenarios/running.pid
```

数据带 `source` 标签：`llm_generated`（训练集来源）、`random_sampled`（仅 Cost Model 用，无 question）。

#### Step 3.5: 贝叶斯优化搜索好配置（可选，需要 PG）

用已有 collected 数据热启动高斯过程代理模型，对每种负载跑 30 轮 BO，搜索中高 TPS 配置。搜索轨迹 ~120 条 + 围绕最优配置的扰动变体 ~200 条。

```bash
mkdir -p logs/scenarios
nohup python3 -m data_pipeline.synthesis.scenarios.pipeline bo-search \
    --knob-space configs/knob_space.yaml \
    --collected "data_pipeline/data/scenarios/collected/collected*.json" \
    --output data_pipeline/data/scenarios/collected/collected_bo.json \
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
    --data data_pipeline/data/scenarios/collected/ \
    --output cost_model/checkpoints/v1
```

#### Step 5: SFT 轨迹合成（`sampler`）

`sampler` 是数据合成与评估的核心脚本。通过 `--mode` 和 `--split` 控制行为：

| 命令 | 作用 | 需要 cost model | 生成 question |
|------|------|:-:|:-:|
| `--mode generate --split train` | 生成训练集：动态 question + rollout + 过滤 | ✅ | ✅ |
| `--mode generate --split eval` | 生成评估集：动态 question，不跑 rollout | ❌ | ✅ |
| `--mode eval --eval-questions <path>` | 评估模型：固定 question + rollout，保留全部 | ✅ | ❌ |

**生成训练数据** (`--mode generate --split train`)

对每个场景并行跑 N 次独立 rollout，question 由 LLM 动态生成，保留 improvement > threshold 的轨迹。

`--scenarios` 支持指定多个文件（避免把 eval 数据混入训练集）：

```bash
python3 -m data_pipeline.synthesis.trajectory.sampler \
    --mode generate --split train \
    --scenarios \
        data_pipeline/data/scenarios/collected/collected_server1.json \
        data_pipeline/data/scenarios/collected/collected_server2.json \
        data_pipeline/data/scenarios/collected/collected_server3.json \
    --cost-model cost_model/checkpoints/v9_lgbm \
    --output-dir data_pipeline/data/train/ \
    --output-file sft_trajectories_v2.jsonl \
    --num-rollouts 8 \
    --good-threshold 0.03 \
    --parallel 8 \
    --providers-config configs/providers.json
```

> **注意**：输出文件使用 append 模式写入。如需重跑，请先删除旧文件或指定新文件名（`--output-file`）。

输出：`data_pipeline/data/train/sft_trajectories_v2.jsonl`，每条记录包含：
- `messages` — 完整多轮对话轨迹（system/user/assistant/tool）
- `question` — 动态生成的用户问题
- `reward` / `improvement_pct` — 该轨迹的 reward 值
- `env_sample_idx` — 对应的场景索引

**生成评估集 question** (`--mode generate --split eval`)

仅为 eval 场景生成用户问题，不需要 cost model，不跑 agent。

```bash
python3 -m data_pipeline.synthesis.trajectory.sampler \
    --mode generate --split eval \
    --scenarios data_pipeline/data/scenarios/collected/collected_eval.json \
    --output-dir data_pipeline/data/eval/ \
    --providers-config configs/providers.json \
    --parallel 10
```

输出：`data_pipeline/data/eval/eval_trajectories.jsonl`，每条记录包含：
- `messages` — 仅 2 条（system + user），无轨迹
- `question` — 用户问题
- `env_sample_idx` — 对应 `collected_eval.json` 中的场景索引

**数据文件对照表**

| 文件 | 来源 | 内容 | 用途 |
|------|------|------|------|
| `data_pipeline/data/train/sft_trajectories_v2.jsonl` | `generate --split train` | 完整轨迹 + question | SFT 训练 |
| `data_pipeline/data/eval/eval_trajectories_filtered.jsonl` | `generate --split eval` + 过滤 | 仅 question（598 条） | 评估时注入 |
| `data_pipeline/data/scenarios/collected/collected_eval_filtered.json` | 真机采集 + GPT-5 基线过滤 | 场景快照（598 个） | 评估场景来源 |

**参数说明**

| 参数 | 说明 | 默认 |
|------|------|------|
| `--mode` | `generate` 生成数据集 / `eval` 评估模型 | 必选 |
| `--split` | generate 模式子类型：`train` / `eval` | train |
| `--scenarios` | 场景文件或目录（支持多个文件） | 必选 |
| `--eval-questions` | eval 模式必需：预生成的 question 文件 | — |
| `--num-rollouts` | 每场景 rollout 次数（eval 模式强制 1） | 8 |
| `--good-threshold` | 正样本阈值（eval 模式强制 -999.0） | 0.03 |
| `--output-file` | 输出文件名（默认 sft_trajectories.jsonl） | 按模式自动 |
| `--temperature` | LLM 采样温度 | 0.7 |
| `--parallel` | 并发 worker 数 | 4 |
| `--source-filter` | 只保留指定 source 的场景（如 `llm_generated`） | 不过滤 |
| `--tps-min` | TPS 下限，低于此值的场景被跳过 | 不限 |
| `--tps-max` | TPS 上限，高于此值的场景被跳过 | 不限 |

#### Step 5（备选）: MCTS 轨迹合成（保留为对比基线）

MCTS 作为对比基线保留，不再主动维护。每轮搜索输出到带时间戳的子目录。

```bash
python3 -m data_pipeline.synthesis.mcts.run_search \
    --scenarios data_pipeline/data/scenarios/collected/ \
    --knob-space configs/knob_space.yaml \
    --cost-model cost_model/checkpoints/v7_lgbm_dedup \
    --output-dir data_pipeline/data \
    --providers-config configs/providers.json \
    --parallel 4 --num-workers 2 --simulations 5 \
    --source-filter llm_generated \
    --tps-min 100 --tps-max 5000
```

输出（在 `data_pipeline/data/run_YYYYMMDD_HHMMSS/` 下）：
- `sft_trajectories.jsonl` — SFT 训练数据
- `contrastive_pairs.jsonl` — DPO/对比学习数据
- `mcts_trees/` — 搜索树 debug 文件

#### 存量数据迁移：为旧轨迹补充 question 字段

针对重构前 MCTS 产出的 `sft_trajectories.jsonl`（旧轨迹中的 question 含技术指标泄露），通过 `env_sample_idx` 精确匹配每条轨迹对应的 ScenarioState（含 knobs），重新生成无泄露的 question。

```bash
python3 -m data_pipeline.synthesis.scenarios.migrate_add_questions \
    --input data_pipeline/data/mcts/run_20260330_053333/sft_trajectories.jsonl \
    --scenarios data_pipeline/data/scenarios/collected/ \
    --providers-config configs/providers.json \
    --workers 10
```

脚本按 `env_sample_idx` 分组，每个 unique 场景一次 LLM 调用批量生成 N 个 question（N=该场景的轨迹数），已有 question 的条目自动跳过（断点续跑）。

#### Step 5.5: 评估集 knob 配置合成

复用 Step 1 的维度组合蒸馏，但 `--per-cell 1`（每格仅 1 条），输出到独立文件，确保与训练集零重叠。合成后无需真机采集，评估时直接用 Cost Model 模拟环境。

```bash
python3 -m data_pipeline.synthesis.scenarios.pipeline synthesize \
    --dimensions configs/synthesis_dimensions.yaml \
    --knob-space configs/knob_space.yaml \
    --output data_pipeline/data/scenarios/knobs/knob_configs_eval.json \
    --per-cell 1 --workers 8 \
    --providers-config configs/providers.json
```

产出约 ~696 条配置，后续评估脚本会读取该文件构建模拟环境。

#### Step 6: SFT 训练

将轨迹转为训练格式，进行 SFT 冷启动训练。训练前会自动按 token 数过滤超长轨迹（超过 `--max_length` 的直接丢弃，不截断），默认 8192。

```bash
# === trl 后端（推荐，不依赖 vLLM/Ray）===

# 方式 1：用脚本
BASE_MODEL=~/models/Qwen2.5-3B-Instruct \
DATA_FILES="data_pipeline/data/train/sft_trajectories.jsonl" \
bash scripts/train_sft_trl.sh

# 方式 2：直接调用
python -m training.trl.sft \
    --model_path ~/models/Qwen2.5-3B-Instruct \
    --data_files data_pipeline/data/train/sft_trajectories.jsonl \
    --output_dir model_save/sft/ \
    --max_length 8192

# === verl 后端（需要 vLLM + Ray）===

# 数据预处理（JSONL → Parquet）
python3 -m datasets.preprocess_sft \
    --input_files data_pipeline/data/train/sft_trajectories.jsonl \
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
    --scenario_dir data_pipeline/data/scenarios/collected/ \
    --cost_model cost_model/checkpoints/v1 \
    --output_dir model_save/grpo/

# === verl 后端 ===
bash scripts/train_grpo_verl.sh
```

#### Step 8: 模型评估

评估复用 sampler 的 `--mode eval`，用待评估模型在 eval 场景上跑 agent，生成轨迹后聚合指标。

**数据流**：
```
collected_eval_filtered.json (场景)  ──┐
                                       ├→ sampler --mode eval ──→ sft_trajectories.jsonl ──→ evaluate.run ──→ eval_report.json
eval_trajectories_filtered.jsonl (question) ─┘
```

**前置条件**：
- `data_pipeline/data/eval/eval_trajectories_filtered.jsonl` 已生成（见下方数据过滤说明）
- 待评估模型已部署为 OpenAI-compatible API（vLLM / 中转站均可）

**Step 8.0: 评估集过滤（仅首次执行）**

以 GPT-5 作为基线模型，先跑一遍完整 eval（693 个场景），根据 `eval_report.json` 中 `imp_vs_scenario_pct >= 50%` 的场景进行过滤。这些场景被视为极端样本（原始性能过低导致提升虚高，或 Cost Model 预测偏差大），不适合用于评估。

过滤后产出：
- `collected_eval_filtered.json`：598 个场景（排除 95 个极端场景）
- `eval_trajectories_filtered.jsonl`：598 条（`env_sample_idx` 已重映射）

> **注意**：过滤结果依赖 GPT-5 + v9_lgbm Cost Model 的组合。若更换 Cost Model 版本，需重新执行过滤流程。

**Step 8a: 用待评估模型跑 eval 轨迹**

eval 模式自动强制 `--num-rollouts 1` 和 `--good-threshold -999.0`，无需手动指定。

```bash
python3 -m data_pipeline.synthesis.trajectory.sampler \
    --mode eval \
    --eval-questions data_pipeline/data/eval/eval_trajectories_filtered.jsonl \
    --scenarios data_pipeline/data/scenarios/collected/collected_eval_filtered.json \
    --cost-model cost_model/checkpoints/v9_lgbm \
    --output-dir eval_results/<模型名>/ \
    --model <served-model-name> \
    --api-base <vLLM或中转站地址>/v1 \
    --api-key <api-key>
```

**Step 8b: 聚合报表（不需要 LLM）**

```bash
python3 -m evaluate.run \
    --eval-data eval_results/<模型名>/sft_trajectories.jsonl \
    --scenarios data_pipeline/data/scenarios/collected/collected_eval_filtered.json \
    --cost-model cost_model/checkpoints/v9_lgbm \
    --output eval_results/<模型名>/
```

评估指标（三 baseline 体系）：
- **vs 场景原始配置** — GPT-5 基线下平均 +21%，核心参考指标
- **vs PG 默认配置** — 相对 PostgreSQL 出厂默认的提升
- **vs BO 最优配置** — 相对 TPE 200-trial 搜索最优的差距
- **Gap Closed** — 模型关闭了多少 default→BO 之间的优化空间



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
