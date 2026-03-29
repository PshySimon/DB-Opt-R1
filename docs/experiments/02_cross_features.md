# 实验 02：交叉特征对 MLP 精准度的影响

## 实验目标

验证 LightGBM 的自动特征交叉能力是否是其优于 MLP 的原因，并探索显式交叉特征能否提升 MLP 泛化精度。

## 实验现状

- LightGBM（树模型）天然支持特征交叉——每个分裂节点自动形成特征交互
- MLP 只能通过隐藏层非线性学习交互，但效率低于显式交叉
- 实验 01 结论：S3（237K 参数）是 MLP 最优规模，验证 1K+ WAPE = 35.2%

## 实验设计

### 第一步：基线对比

用同一份数据（all_with_active.json），分区间对比 LightGBM 和 MLP 的验证集 WAPE，确认 LightGBM 是否确实更好、好在哪些区间。

### 第二步：分析误差来源

按区间分析预测偏差的模式，找出哪些区间/场景精准度最差，以及这些场景的特征有什么共同点。

### 第三步：交叉特征实验（待第二步结论后设计）

根据第二步的分析结果，设计针对性的交叉特征。

## 实验结果

### 第一步：LightGBM vs MLP 分区间 WAPE（验证集）

| 区间 | 样本 | LGBM WAPE | LGBM 偏差 | MLP WAPE | MLP 偏差 | LGBM 优势 |
|---|---|---|---|---|---|---|
| 0-1K | 1407 | 27.6% | +16.0% | 26.8% | +11.4% | MLP 略好 |
| 1K-2K | 102 | 50.3% | +16.7% | 60.4% | +14.4% | LGBM +10.1% |
| 2K-3K | 49 | 40.9% | -28.3% | 51.9% | -20.4% | LGBM +11.0% |
| 3K-5K | 260 | 48.2% | +32.7% | 60.9% | +43.8% | LGBM +12.7% |
| 5K-10K | 241 | 51.6% | +21.9% | 64.1% | +26.5% | LGBM +12.5% |
| 10K+ | 389 | 26.3% | -21.3% | 31.4% | -16.0% | LGBM +5.1% |

**观察**：
- **0-1K**：两者接近，MLP 略好
- **1K-10K**：LightGBM 全面优于 MLP，差距 10-13 个百分点
- **10K+**：LightGBM 优势缩小到 5%
- **偏差方向**：两个模型在大多数区间偏差方向一致（同高估或同低估），说明这不是模型结构差异，而是数据本身的分布偏差

### 第二步：误差来源分析

**运行时指标不可用**：
- `db_metrics`：全局累积计数器没做 reset/差值，同机器所有样本值一样
- `system`（cpu/disk/mem）：pgbench 跑完后采集，进程已退出，全是 0
- `wait_events`：覆盖率仅 0.4%（49/12829）
- `latency_avg_ms`：是 pgbench 输出（和 TPS 同源），非可预测特征

**可用的只有静态特征**（knob + hardware IO + workload 类型），因为 RL 推理时只有这些信息。

**Spearman 相关性按 workload 分析（关键发现）**：

| 特征 | write_heavy | read_only | 差异 |
|---|---|---|---|
| commit_delay | -0.50 | +0.08 | 方向相反 |
| max_connections | -0.23 | +0.07 | 方向相反 |
| effective_io_concurrency | -0.29 | +0.09 | 方向相反 |
| rand_read_iops | +0.12 | +0.44 | 只读场景 3.7 倍 |

### 第三步：交叉特征设计

**新增特征类型（共 50 个，基线 122 → 172 维）：**

1. **workload × knob 交叉**（4 wl × 6 knob = 24 个）：`commit_delay`、`max_connections`、`effective_io_concurrency`、`checkpoint_completion_target`、`work_mem`、`random_page_cost` 各自与 4 种 workload one-hot 的乘积
2. **workload × IO 交叉**（4 wl × 4 io = 16 个）：`rand_read_iops`、`seq_write_mbps`、`mem_bw_gbps`、`seq_write_p99_lat_us` 各自与 4 种 workload 的乘积
3. **max_connections 悬崖特征**（6 个）：`mc > 300`、`mc > 500` 与 `high_concurrency`/`mixed` workload 的组合，捕捉连接数打满的性能跳崖
4. **IO 延迟 × 写 workload**（3 个）：`seq_write_p99_lat_us` × `write_heavy`/`mixed`/`high_concurrency`，捕捉慢盘对写 TPS 的非线性影响
5. **内存带宽 × read_only**（1 个）：`mem_bw_gbps × read_only`，捕捉只读场景瓶颈在内存带宽而非 IO 的特性

**关键数据发现（设计依据）：**
- `commit_delay` 对 write_heavy 的 Spearman 相关 = -0.50，对 read_only = +0.08，方向相反
- `max_connections > 500` 时 high_concurrency TPS 从中位 570 骤降到 200-470（连接数打满悬崖）
- 三台机器 CPU/内存完全一样，差异只在 IO，但 read_only TPS 与 IOPS 并非线性（受内存带宽约束）

### 第四步：多规模 MLP × 特征集 实验（训练/验证 WAPE）

**实验配置**：数据集 12851 条，train/val 8:2 分割（random_state=42），300 epoch + early_stopping(patience=30)

| 模型 | 特征 | 参数量 | 训练 WAPE | 验证 WAPE | gap |
|---|---|---|---|---|---|
| LightGBM | 基线 122 | — (142 iter) | 14.0% | **32.2%** | +18.3% |
| LightGBM | 交叉 172 | — (155 iter) | 12.5% | **31.6%** | +19.0% |
| MLP S2 | 基线 122 | 65K | 28.7% | 41.9% | +13.2% |
| MLP S3 | 基线 122 | 237K | 19.9% | 37.6% | +17.7% |
| MLP S4 | 基线 122 | 819K | 21.6% | 38.2% | +16.6% |
| MLP S2 | 交叉 172 | 78K | 27.3% | 37.5% | +10.2% |
| MLP S3 | 交叉 172 | 263K | 19.1% | 35.9% | +16.9% |
| MLP S4 | 交叉 172 | 870K | 16.8% | **34.4%** | +17.6% |

### 实验结论

**关于原始假设（LightGBM 优于 MLP 是因为特征交叉能力？）**

假设**被部分否定**。加了显式交叉特征后 MLP S4 验证 WAPE = 34.4%，仍比 LightGBM 差 2.8%。核心发现：

1. **LightGBM 本身泛化也受限**：LightGBM 的 train-val gap 为 18-19%，与 MLP 的 16-18% 几乎相同。两者的瓶颈都不是模型结构，而是**数据泛化问题**。

2. **LightGBM 的优势来自更强的训练集拟合**：LightGBM 训练 WAPE = 14%（MLP 约 20%），在相同 gap 下因此获得更低的验证 WAPE。树模型的归纳偏置（离散分裂、自然处理非线性边界）比 MLP 更适合此类 tabular 数据。

3. **交叉特征对 MLP 有效，对 LightGBM 无效**：MLP 验证 WAPE 从 37.6% 降到 34.4%（改善 3.2%），LightGBM 从 32.2% 降到 31.6%（改善 0.6%），说明 LightGBM 已经自动学会了这些交叉关系。

4. **泛化 gap 是核心问题**：所有配置的 train-val gap 均在 10-19%。提升验证精度的根本路径不是加特征或调模型，而是**改善数据分布覆盖**（更多样的 knob/hardware/workload 组合）或**引入正则化**（dropout、weight decay、数据增强）。

**当前最优模型**：LightGBM + 交叉 172 维，验证 WAPE = 31.6%

**下一步方向**：
- [ ] 数据策略：分析哪些 knob×workload×hardware 组合在训练集中覆盖不足
- [ ] 正则化实验：对 MLP S4 加强 dropout/weight decay，看能否缩小 gap
- [ ] 集成方案：MLP + LightGBM 加权集成，利用各自互补的误差模式
