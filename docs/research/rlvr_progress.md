# RLVR 研究进展调研报告

> **调研时间**: 2026-04-12  
> **调研范围**: 从 DeepSeekMath (2024-02) 到 2026-04 的 RLVR 研究主线  
> **关键词**: RLVR, GRPO, Reasoning, RL for LLM, Long-CoT

---

## 一、概念定义

### RLVR（Reinforcement Learning with Verifiable Rewards）

RLVR 是一种以**可验证的规则型信号**替代人类反馈（RLHF）的训练范式。其核心思路是：

- 对于有确定性 ground-truth 的任务（如数学、代码），无需训练 Reward Model
- 用 **规则函数** 直接判断输出是否正确（例如：答案是否正确、代码是否通过单元测试）
- 以此为 RL 信号进行策略优化

**优势**：  
- 无 reward model 偏差（reward model 本身可能幻觉/过拟合）  
- 无 reward hacking（最终验证器是客观程序）  
- 对有确定性 ground-truth 的推理任务效果极强

**局限**：  
- 依赖 ground-truth 可验证性，难以直接迁移到开放域（写作、主观理解等）  
- 奖励信号稀疏（只在轨迹末尾给出 0/1 反馈）

---

## 二、时间线与关键论文

### 2024-02 | DeepSeekMath + GRPO 首次提出

**论文**: *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*  
**arXiv**: 2402.03300  
**机构**: DeepSeek

#### 核心贡献

1. **GRPO（Group Relative Policy Optimization）算法首次提出**
   - 传统 PPO 需要独立的 Critic 模型来估计 Value Function，显存开销极大
   - GRPO 的核心创新：对同一个 prompt **采样一组（group）响应**，以组内响应的**平均奖励作为基线**，计算每条响应的相对 Advantage
   - 彻底消除 Critic 模型，大幅降低显存占用，同时保留 PPO 的信任域约束

2. **数学推理的 RL 范式确立**
   - 120B token 高质量数学语料预训练
   - SFT → GRPO RL 两阶段 pipeline
   - 在 MATH benchmark 上从 SFT 46.8% 提升至 GRPO 后 51.7%（无外部工具、无投票）

#### GRPO 算法要点

```
对每个 prompt q，采样 G 个响应 {o_1, ..., o_G}
计算每个响应奖励 {r_1, ..., r_G}
Advantage: A_i = (r_i - mean(r)) / std(r)
Policy gradient: 带 PPO-style clip 的重要性采样更新
```

**意义**: GRPO 成为后续所有 RLVR 工作的基础算法，DeepSeek 系列模型的核心训练工具。

---

### 2025-01 | DeepSeek-R1 / R1-Zero：RLVR 的里程碑时刻

**论文**: *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*  
**机构**: DeepSeek  
**时间**: 2025-01

#### 两个模型，两种路线

**DeepSeek-R1-Zero**：纯 RL，无 SFT

| 属性 | 详情 |
|------|------|
| 起点 | DeepSeek-V3-Base（无任何 SFT） |
| 算法 | GRPO |
| 奖励 | 规则验证（正确性 + 格式） |
| 发现 | 模型**自发涌现**长链式思维、自我验证、反思行为 |
| 问题 | 可读性差、语言混用、重复输出 |

> **核心发现**: 纯靠 RL + 可验证奖励，无需任何 SFT 数据，模型就能自发学会"思考"策略，这是 RLVR 领域最震撼的实验结果。

**DeepSeek-R1**：多阶段精炼 pipeline

```
阶段 1: Cold-Start SFT
  → 4-8K 条精心标注的 Long-CoT 样本
  → 目的：解决 R1-Zero 的格式混乱问题

阶段 2: 推理导向 GRPO
  → 大规模 RL（数学/代码/逻辑任务）
  → 奖励：规则验证（RLVR）

阶段 3: Rejection Sampling + SFT
  → 用 RL 模型做拒绝采样生成高质量数据
  → 泛化训练：写作/问答/通用任务

阶段 4: 最终 RL 对齐
  → 混合推理奖励 + 人类偏好对齐
```

#### 学术影响

- 证明 RLVR 可以在不依赖大量标注数据的情况下激发高水平推理
- 开源模型性能比肩 OpenAI o1，引爆 "reasoning model" 开发浪潮
- **GRPO + RLVR 成为 2025 年推理模型训练的标准范式**

---

### 2025-03 | DAPO：规模化 RLVR 的工程突破

**论文**: *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*  
**arXiv**: 2503.14476  
**机构**: ByteDance Seed × 清华 AIR × 港大  
**时间**: 2025-03

#### 背景

将 vanilla GRPO 扩展到大规模、长 CoT 任务时，发现了若干系统性问题。DAPO 针对这些问题提出了四项关键技术。

#### 四大技术创新

**① Clip-Higher（不对称截断）**
- 问题：标准 PPO clip 对上下界使用相同 ε，限制了策略多样性
- 方案：**提高重要性采样比的上界 clip 范围**（高 ε_high >> 低 ε_low）
- 效果：防止 Entropy Collapse（熵崩溃），保持模型探索能力

**② Dynamic Sampling（动态采样）**
- 问题：训练批次中存在"全对"或"全错"样本（advantage 为 0），白白浪费计算
- 方案：**过滤掉组内奖励无方差的样本**（only keep samples where not all rewards are identical）
- 效果：提高有效梯度比例，加速收敛

**③ Token-Level Policy Gradient Loss**
- 问题：长 CoT 场景下，按序列级别归一化 loss 会导致梯度消失
- 方案：**token-level 归一化**（每个 token 都产生 gradient，而非对整个序列 average）
- 效果：对长推理链尤为关键，解决了长序列梯度不稳定问题

**④ Overlong Reward Shaping（超长惩罚）**
- 问题：模型学会用无意义冗余内容填充来逃避验证
- 方案：**对超出长度上限的响应施加渐进式惩罚**
- 效果：减少奖励噪声，防止 Length Hacking

#### 关键结果

- 使用 **Qwen2.5-32B** 基座，在 **AIME 2024** 上达到 **50 分**
- 超越 DeepSeek-R1-Zero-Qwen-32B，训练步数仅为其 50%
- 基于 veRL 框架，完整开源

---

### 2025-04 | VAPO：价值模型的回归

**论文**: *VAPO: Value-model-based Augmented Proximal Policy Optimization*  
**arXiv**: 2504.05118  
**机构**: ByteDance Seed  
**时间**: 2025-04

#### 背景

GRPO/DAPO 等无 Critic 方法虽然高效，但在**复杂长链推理**任务上存在训练不稳定的问题。VAPO 尝试回归价值模型，同时解决价值模型本身的三大痛点。

#### 三大挑战 & 解决方案

| 挑战 | 问题描述 | VAPO 方案 |
|------|----------|-----------|
| Value Model Bias | 用 Reward Model 初始化 Value Model 引入偏差 | Value Pretraining：先对 Value Model 做专门预训练 |
| 异构序列长度 | 标准 GAE 难以处理长短不一的响应 | Length-adaptive GAE：根据响应长度动态调整 λ 参数 |
| 奖励信号稀疏 | 二值验证奖励梯度信号弱 | Decoupled GAE：解耦价值估计与奖励 |

#### Length-adaptive GAE 核心思想

```
λ_effective = f(response_length)
  → 短响应：较小 λ（近似 TD(0)，减少 bias）
  → 长响应：较大 λ（近似 MC，减少 variance）
```

#### 关键结果

- **Qwen2.5-32B（无 SFT 数据）**，AIME 2024 达到 **60.4 分**
- 显著超越 DAPO（50 分）和 DeepSeek-R1-Zero-Qwen-32B
- 5000 步内达到 SOTA，**训练过程无崩溃**
- 集成至 veRL 框架

---

### 2025 全年 | GRPO 的问题诊断与算法修复

随着 GRPO 被大规模采用，研究者发现了若干系统性问题，并提出了多种修复方案。

#### 已识别的核心问题

**① 熵崩溃（Entropy Collapse）**
- 描述：训练进展后，模型的输出分布变得过于确定性，丧失探索能力
- 表现：所有 prompt 都生成极为相似的输出，训练陷入停滞
- 后果：模型对新问题的泛化能力退化

**② 优势崩溃（Advantage Collapse）**
- 描述：同一组内所有响应获得相同奖励（全对/全错），advantage 归零
- 后果：梯度更新为 0，训练停止进展

**③ 长度偏差（Length Bias）**
- 描述：GRPO 的 token-level 加权方式可能偏向产生较长响应，特别是错误答案
- 表现：模型输出不断冗长，用重复内容填充推理过程

**④ Lazy Likelihood Displacement（LLD）死亡螺旋**
- 描述：在复杂工具调用场景下，模型系统性地降低所有 token 的生成概率
- 表现：正确/错误响应的 likelihood 同时下降，训练崩溃

#### 提出的修复方案

| 方案 | 解决的问题 | 核心机制 |
|------|-----------|---------|
| **DrGRPO** | 长度偏差 | 用全局常数归一化 token-level loss，消除序列长度的 bias |
| **DAPO Clip-Higher** | 熵崩溃 | 不对称 clip 保留上行探索空间 |
| **AEPO** | 熵崩溃 | 将熵正则化重新表述为约束优化 |
| **EDGE-GRPO** | 熵崩溃 | 将相对熵直接融入 Reward |
| **LLDS** | LLD 死亡螺旋 | 仅对降低的 token likelihood 施加正则化 |
| **Scaf-GRPO** | Advantage Collapse | 渐进式提示（Hint）注入，保证非零奖励信号 |

---

### 2025-04 | 学术争论：RLVR 真的有效吗？

**论文**: *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?*  
**arXiv**: 2504.13837  
**作者**: Yang Yue et al.  
**时间**: 2025-04

#### 核心论点（对 RLVR 的质疑）

这篇论文在学术界引发了广泛讨论，其核心主张是：

> **RLVR 并未给模型带来真正的新推理能力，而只是提升了"采样效率"。**

**实验证据**：
- 使用 `pass@1`：RLVR 训练的模型优于基座模型 ✓
- 使用 `pass@k`（k 较大时）：**基座模型经常追平甚至超越 RLVR 训练模型** ✗

**解读**：  
RLVR 的作用是**收窄输出分布，使正确路径浮现更一致**，而非真正扩展模型的推理边界。基座模型具备的正确推理路径，RLVR 只是让它更容易在第一次尝试中出现。

**RLVR vs 蒸馏（Distillation）的对比**：
- RLVR：在基座模型已有能力的范围内优化分布，不引入新知识
- **Distillation**：从更强模型学习，真正引入新的推理路径（可以突破基座模型的上限）

#### 社区反应与对立观点

支持该论点的研究：
- 2026 年 Alibaba/Qwen 团队进一步发现，RLVR 引发的参数变化极为**稀疏**，且集中在"低概率推理关键 token"上

反驳/补充观点：
- 从实用角度，`pass@1` 在工程应用中更重要；提升 pass@1 本身就有巨大价值
- RLVR 对于确定性验证任务（尤其小模型）仍然极为有效
- 争论促使研究者将 **RLVR 与 Distillation 结合**，取长补短

---

## 三、RLVR 的任务领域扩展

从 2024 年底到 2025 年，RLVR 的应用任务范围从数学/代码迅速扩展。

### 3.1 数学与逻辑推理（原始领域）

| 特点 | 说明 |
|------|------|
| 奖励形式 | 最终答案正确性（二值） |
| 验证器 | 符号比较 / Python 求值 |
| 代表任务 | GSM8K, MATH, AIME, AMC |
| 代表工作 | DeepSeekMath, R1-Zero, DAPO, VAPO |

### 3.2 代码生成与执行（RLVR 天然适配）

| 特点 | 说明 |
|------|------|
| 奖励形式 | 单元测试通过率 |
| 验证器 | 代码执行器（沙箱） |
| 代表任务 | HumanEval, LiveCodeBench, SWE-Bench |
| 代表工作 | Agent-RLVR（SWE-Bench 场景的稀疏奖励优化） |

### 3.3 多模态视觉推理

| 研究工作 | 核心贡献 |
|---------|---------|
| **R1-VL / R1-OneVision** | 首批将 GRPO 迁移至 VLM 的工作，验证 RLVR 在 VLM 上的可行性 |
| **MM-Eureka** | 多模态推理 RLVR 训练稳定化技巧与数据集构建 |
| **Perception-R1** | 引入视觉感知奖励（感知一致性），用极少训练数据达到 SOTA |
| **PRCO（双角色框架）** | 发现 RLVR 只优化推理而不优化感知，提出 Observer + Solver 双角色分别奖励 |
| **ToR（Token Reweighting）** | 动态重加权感知 token 与推理 token，解决多模态 credit assignment 问题 |

**关键发现**：RLVR 对 VLM 的"推理"部分有效，但对"视觉感知"（从图像中提取证据）改善有限，这是多模态 RLVR 的独特挑战。

### 3.4 智能体与工具使用（Agentic RLVR）

这是 2025 年最活跃的方向之一，挑战在于：
- 工具调用链路长，奖励极度稀疏
- 多轮交互使信用分配（Credit Assignment）更困难

| 研究工作 | 核心贡献 |
|---------|---------|
| **Agent-R1** | 多轮工具调用 RLVR 框架，处理复杂 agentic 场景 |
| **ARTIST** | 紧耦合"推理-工具调用"的统一 RL 框架，自主决策工具调用时机 |
| **Tool-R1** | 生成可执行 Python 代码做灵活工具调用，支持动态控制流 |
| **SWiRL** | 将多步轨迹拆分为子轨迹（step-wise），更细粒度的 RL 优化 |
| **Agent-RLVR** | 通过"教学性引导（pedagogical guidance）"解决 SWE-Bench 奖励稀疏问题 |

**我们项目（DB-Opt-R1）属于此类**：数据库调优动作构成多步工具调用链，RLVR reward 来自性能提升验证，与上述研究方向高度对齐。

### 3.5 知识密集型领域（医学/法律/科学）

| 挑战 | 这些领域缺乏简单的对错验证 |
|------|--------------------------|
| **Knowledge-to-Verification (K2V)** | 将领域知识图谱转化为可验证的填空题/QA，构造 verifiable reward |
| **Sub-task Decomposition** | 将复杂推理拆解为若干子任务，每步用规则验证 |
| **CellDuality（生物）** | 利用任务对偶性（任务 A 的输出可验证任务 B）构造自一致性奖励 |

### 3.6 开放域与主观任务（前沿探索）

| 挑战 | 无明确 ground truth，难以定义 verifier |
|------|---------------------------------------|
| **VMR（Verifiable Multiple-Choice Reformulation）** | 将开放问题转化为多选题，实现二值验证 |
| **混合奖励策略** | 规则奖励（格式、约束）+ Reward Model 奖励（质量）结合 |
| **RLVRR（Reward Chain）** | ICLR 2026 分享，通过奖励链扩展 RLVR 至半可验证领域 |

---

## 四、算法演进谱系

```
PPO（经典 RL 基准）
│
├─ GRPO（2024-02, DeepSeekMath）
│   ├─ 消除 Critic 模型
│   ├─ 组内相对优势估计
│   └─ 成为 RLVR 标准算法
│
├─ REINFORCE++（2025）
│   ├─ 全局 advantage 归一化（跨 batch）
│   ├─ token-level KL 惩罚
│   └─ PPO-style clip，无 Critic
│
├─ DrGRPO（2025）
│   ├─ 修复 length bias
│   └─ 全局常数 token loss 归一化
│
├─ DAPO（2025-03, ByteDance）
│   ├─ Clip-Higher（防熵崩溃）
│   ├─ Dynamic Sampling（动态过滤无效样本）
│   ├─ Token-level loss（长 CoT 梯度稳定）
│   └─ Overlong Reward Shaping（防 Length Hack）
│
└─ VAPO（2025-04, ByteDance）
    ├─ 回归 Value Model（PPO-style）
    ├─ Value Pretraining（消除初始化偏差）
    ├─ Length-adaptive GAE（处理异构序列长度）
    └─ 当前 AIME 2024 SOTA（60.4，32B 级别）
```

---

## 五、学术认知的演变

### Phase 1（2024-02 ~ 2024-12）：RLVR 的建立期

- DeepSeekMath 提出 GRPO，证明无 Critic 的 RL 可以有效提升推理
- 认知：RLVR 是 RLHF 在推理任务上的高效替代
- 重点问题：如何高效计算奖励、如何构造良好的 verifier

### Phase 2（2025-01）：R1-Zero 的震撼

- 纯 RL（无 SFT）涌现长链式推理，颠覆了认知
- 认知：充分的可验证奖励 + 足够大的模型 → 推理能力可以涌现
- 整个业界开始竞相复现 R1，"reasoning model" 成为热词

### Phase 3（2025-Q1 ~ Q2）：规模化工程挑战

- 真正尝试训练后，发现 GRPO 存在：熵崩溃、长度偏差、Advantage Collapse 等问题
- 认知：GRPO 算法本身有缺陷，需要工程化修复
- DAPO、DrGRPO、REINFORCE++ 等出现，针对具体问题提出补丁

### Phase 4（2025-Q2）：对 RLVR 本质的质疑

- "采样效率"论文（arXiv:2504.13837）引发争论
- 认知分裂：
  - 乐观派：RLVR 显著提升 pass@1，工程价值毋庸置疑
  - 批判派：RLVR 不创造新能力，受 Base Model 能力上限约束
- 共识：**RLVR 与 Distillation 结合**是更优策略（R1-Zero → R1 的多阶段 pipeline 正是此思路）

### Phase 5（2025-Q3 ~ 2026-Q1）：走向成熟与扩展

- 算法层：VAPO 等引入更精细的价值估计，推理性能持续突破
- 任务层：RLVR 大规模扩展到多模态、Agent、领域知识任务
- 推理计算层：RLVR 与 Test-Time Compute Scaling 深度结合
- 机制分析层：从"能做什么"转向"为什么有效"（token-level 稀疏更新分析等）
- 新认知：**RLVR 优化的是低概率推理关键 token 的概率，而非全局参数调整**（Qwen/Alibaba 分析，2026-03）

---

## 六、关键 Benchmark 进展（AIME 2024 为参照）

| 时间 | 模型/方法 | AIME 2024 | 基座 |
|------|----------|-----------|------|
| 2025-01 | DeepSeek-R1-Zero-Qwen-32B | ~47 | Qwen2.5-32B |
| 2025-03 | DAPO | 50 | Qwen2.5-32B |
| 2025-04 | VAPO | 60.4 | Qwen2.5-32B |
| 2025-XX | DeepSeek-R1 (full pipeline) | ~72 | DeepSeek-V3 |

---

## 七、对 DB-Opt-R1 项目的启示

### 7.1 任务归类

DB-Opt-R1 本质上是一个 **Agentic RLVR** 任务：
- **动作空间**：数据库配置参数的调整
- **验证器**：数据库性能指标（吞吐量/延迟的实测提升）
- **奖励信号**：性能改善率（连续值，而非 0/1）
- **挑战**：轨迹较长、奖励稀疏、环境非确定性

### 7.2 算法选择建议

参考上述进展：

| 需求 | 建议 |
|------|------|
| 防熵崩溃 | 采用 DAPO 的 Clip-Higher 机制，或显式 entropy regularization |
| 防长度 Hack | Token-level loss 归一化（DrGRPO 思路）+ Overlong 惩罚 |
| 处理稀疏 reward | 奖励塑形（中间步骤的格式正确性等作为 dense reward） |
| 防 Advantage Collapse | 确保 rollout.n >= 5，使每组有足够方差 |
| 多步轨迹优化 | 参考 SWiRL 的 step-wise sub-trajectory RL |

### 7.3 Reward Design 参考

根据上述研究，当前 DB-Opt-R1 的 reward 设计原则：

- ✅ **连续性奖励**（性能提升率 × 5）→ 比 0/1 二值奖励更丰富的梯度信号
- ✅ **允许负奖励**（性能下降时）→ 避免模型通过"不操作"来规避惩罚
- ✅ **Format 奖励**（工具调用格式正确性）→ 提供 dense reward，防止 Advantage Collapse
- ⚠️ **注意**：format_score 权重不宜过高，避免模型只优化格式而忽略性能

---

## 八、重点论文索引

| 论文 | arXiv | 时间 | 机构 | 核心贡献 |
|------|-------|------|------|---------|
| DeepSeekMath | 2402.03300 | 2024-02 | DeepSeek | GRPO 算法首次提出 |
| DeepSeek-R1 | 2501.12948 | 2025-01 | DeepSeek | RLVR 推理模型里程碑，R1-Zero 涌现现象 |
| DAPO | 2503.14476 | 2025-03 | ByteDance × 清华 | 规模化 RLVR 系统，4 项工程优化 |
| VAPO | 2504.05118 | 2025-04 | ByteDance | 价值模型回归，AIME SOTA（60.4） |
| Does RL Incentivize Reasoning? | 2504.13837 | 2025-04 | — | 质疑 RLVR 实质，"采样效率"理论 |
| DrGRPO | — | 2025 | — | 修复 GRPO Length Bias |
| Agent-RLVR | — | 2025 | — | 稀疏奖励 Agentic 场景 |
| SWiRL | — | 2025 | Stanford | Step-wise RL 多步工具使用 |
| ARTIST | — | 2025 | — | 推理-工具调用统一框架 |
| PRCO | — | 2025 | — | 多模态感知-推理双角色 RLVR |
---

## 九、DB-Opt-R1 实验结果

### 9.1 评估设置

- **评估集**: 693 个场景（`collected_eval.json`）
- **Cost Model**: LightGBM v9（log-MAE=0.254, Spearman=0.973）
- **基线体系**:
  - vs 场景原始配置（场景 `tps_current`）
  - vs PG 默认配置（Cost Model 预测）

### 9.2 结果对比

| 模型 | avg_imp_vs_scenario | median_imp_vs_scenario | 提升率 | avg_imp_vs_default | median_imp_vs_default |
|------|:---:|:---:|:---:|:---:|:---:|
| **Qwen3-4B Base**（无 SFT） | 0% | 0% | 0% | — | — |
| **Qwen3-4B SFT** | 13.32% | 0.68% | 52.7% | 104.96% | 92.27% |
| **Qwen3-4B SFT (full)** | 13.26% | 2.91% | 57.8% | 108.44% | 95.54% |
| **Qwen3-4B GRPO-LoRA (80 step)** | **15.71%** | **4.25%** | **60.9%** | **108.56%** | **94.45%** |
| GPT-5 蒸馏（上限参考） | 21.39% | 8.58% | 68.1% | 118.09% | 110.24% |

### 9.3 分析

**GRPO 相对 SFT 的提升**:
- 平均提升: 13.32% → 15.71%（+2.39pp）
- 中位数提升: 0.68% → 4.25%（+3.57pp，**6x 改善**）
- 正向比例: 52.7% → 60.9%（+8.2pp）

**关键观察**:
1. **GRPO 确实有效**: 在 SFT 基础上进一步提升了中位数和正向比例，说明 RL 策略优化起到了作用
2. **提升幅度有限**: 与 GPT-5 蒸馏的差距仍然明显（avg 15.71% vs 21.39%），可能原因：
   - 4B 模型容量有限，推理路径过于稀疏
   - 仅训练 80 step，RL 策略可能尚未充分收敛
   - LoRA rank=64 限制了参数更新空间
3. **vs 默认配置已接近上限**: 108.56% vs 108.44%（SFT full），说明模型已较好地学会了"从默认配置优化"的基本策略，GRPO 的边际贡献主要体现在"从场景配置进一步优化"

### 9.4 后续方向

- [ ] 增加训练步数（200+ step）观察收敛趋势
- [ ] 更新 reward 设计（answer 主导，允许负值）后重新训练
- [ ] 尝试 full-parameter GRPO（去除 LoRA 限制）
- [ ] 增大 group size（n=8~16）提供更丰富的对比信号

---

*最后更新: 2026-04-13*

