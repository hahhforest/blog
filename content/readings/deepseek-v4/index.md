---
title: "DeepSeek-V4：迈向高效的百万 Token 上下文智能"
date: 2026-04-25
lastmod: 2026-04-25
draft: false
description: "DeepSeek-V4 技术报告全文翻译与批注。V4 系列包含两个 MoE 模型（1.6T/284B 参数），通过混合压缩注意力（CSA+HCA）、流形约束超连接（mHC）和 Muon 优化器，将百万 token 上下文的推理 FLOPs 降至 V3.2 的 27%、KV 缓存降至 10%。后训练采用领域专家独立训练 + 在策略蒸馏（OPD）的新范式。"
tags: ["LLM", "论文精读", "MoE", "长上下文", "DeepSeek"]
categories: ["博览"]
author: "北海"
original_title: "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence"
original_url: "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf"
original_author: "DeepSeek-AI"
original_date: "2025"
content_type: "paper"
showToc: true
TocOpen: false
math: true
ShowReadingTime: true
ShowWordCount: true
comments: true
---

> 原文：[DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)
> 作者：DeepSeek-AI
> 模型下载：[https://huggingface.co/collections/deepseek-ai/deepseek-v4](https://huggingface.co/collections/deepseek-ai/deepseek-v4)

---

## 摘要

我们发布 DeepSeek-V4 系列的预览版本，包含两个强大的混合专家（Mixture-of-Experts, MoE）语言模型——DeepSeek-V4-Pro（总参数 1.6T，激活参数 49B）和 DeepSeek-V4-Flash（总参数 284B，激活参数 13B），两者均支持一百万 token 的上下文长度。DeepSeek-V4 系列在架构和优化上引入了若干关键升级：（1）混合注意力架构，结合压缩稀疏注意力（Compressed Sparse Attention, CSA）和重度压缩注意力（Heavily Compressed Attention, HCA），以提升长上下文效率；（2）流形约束超连接（Manifold-Constrained Hyper-Connections, *m*HC），增强传统残差连接；（3）Muon 优化器，加速收敛并提升训练稳定性。我们在超过 32T 多样化高质量 token 上预训练两个模型，随后通过全面的后训练流水线进一步释放和增强其能力。DeepSeek-V4-Pro-Max——DeepSeek-V4-Pro 的最大推理强度模式——重新定义了开源模型的最优表现，在核心任务上超越其前代模型。同时，DeepSeek-V4 系列在长上下文场景下极为高效。在百万 token 上下文设定下，DeepSeek-V4-Pro 的单 token 推理 FLOPs 仅为 DeepSeek-V3.2 的 27%，KV 缓存仅为其 10%。这使我们能够在日常服务中支持百万 token 上下文，从而使长期规划任务和进一步的测试时缩放成为可能。

> **Q：为什么 DeepSeek 选择在 V4 阶段将上下文长度推到 1M token，而不是继续专注于提升模型在短上下文下的质量？**
>
> **A：** 这背后有一个战略判断——测试时缩放（test-time scaling）正在成为提升推理能力的主要手段。更长的上下文意味着模型可以进行更长的推理链、处理更复杂的 agentic 工作流，同时也能将多轮对话和工具调用的完整历史保留在上下文中。换句话说，长上下文不是一个独立的功能点，而是推理能力和 agent 能力的基础设施。这是 DeepSeek 对行业趋势的前瞻性投注。

---

*以下为论文正文全文翻译。篇幅较长，建议使用目录导航。*

## 1. 引言

推理模型的兴起（DeepSeek-AI, 2025; OpenAI, 2024c）确立了测试时缩放这一新范式，为大语言模型（LLM）带来了显著的性能提升。然而，这一缩放范式从根本上受制于原始注意力机制（Vaswani et al., 2017）的二次方计算复杂度，这为超长上下文和推理过程造成了难以逾越的瓶颈。与此同时，长期规划场景和任务的涌现——从复杂的 agentic 工作流到大规模跨文档分析——也使得高效支持超长上下文成为未来进步的关键。尽管近期的开源工作（Bai et al., 2025a; DeepSeek-AI, 2024; MiniMax, 2025; Qwen, 2025）推动了通用能力的进步，但在处理超长序列方面的核心架构低效问题仍然是一个关键障碍，限制了测试时缩放带来的进一步收益，也阻碍了对长期规划场景和任务的深入探索。

为了打破超长上下文领域的效率壁垒，我们开发了 DeepSeek-V4 系列的预览版本，包括 DeepSeek-V4-Pro（总参数 1.6T，激活参数 49B）和 DeepSeek-V4-Flash（总参数 284B，激活参数 13B）。通过架构创新，DeepSeek-V4 系列在处理超长序列时的计算效率实现了跨越式提升，高效支持百万 token 的上下文长度，开创了下一代 LLM 百万级上下文的新时代。

与 DeepSeek-V3 架构相比，DeepSeek-V4 系列保留了 DeepSeekMoE 框架和多 Token 预测（Multi-Token Prediction, MTP）策略，同时在架构和优化方面引入了若干关键创新。为提升长上下文效率，我们设计了混合注意力机制，结合压缩稀疏注意力（CSA）和重度压缩注意力（HCA）。CSA 沿序列维度压缩 KV 缓存，然后执行 DeepSeek 稀疏注意力（DSA），而 HCA 对 KV 缓存进行更激进的压缩但保持密集注意力。为了增强建模能力，我们引入了流形约束超连接（*m*HC），升级传统残差连接。此外，我们将 Muon 优化器引入 DeepSeek-V4 系列的训练，加速收敛并提升训练稳定性。

通过混合 CSA 和 HCA，以及对计算和存储的精度优化，DeepSeek-V4 系列在推理 FLOPs 和 KV 缓存大小方面都实现了显著降低，尤其是在长上下文场景下。在百万 token 上下文场景下，即使拥有更多激活参数的 DeepSeek-V4-Pro，其单 token FLOPs（以等效 FP8 FLOPs 衡量）也仅为 DeepSeek-V3.2 的 27%，KV 缓存大小仅为其 10%。DeepSeek-V4-Flash 效率表现更为极致：单 token FLOPs 仅为 DeepSeek-V3.2 的 10%，KV 缓存大小仅为其 7%。

DeepSeek-V4 系列的后训练流水线采用两阶段范式：先独立培养各领域专家模型，然后通过在策略蒸馏（On-Policy Distillation, OPD）统一合并。对于每个目标领域——如数学、编程、agent 和指令跟随——分别独立训练一个专家模型，最后通过在策略蒸馏将这些不同的专业能力整合到一个统一模型中。

**核心评估结果总结**

- **知识**：DeepSeek-V4-Pro-Max 在 SimpleQA 和 Chinese-SimpleQA 等基准上显著超越领先的开源模型，大幅缩小了与领先闭源模型 Gemini-3.1-Pro 的差距。
- **推理**：DeepSeek-V4-Pro-Max 在标准推理基准上展现出优于 GPT-5.2 和 Gemini-3.0-Pro 的性能，但与 GPT-5.4 和 Gemini-3.1-Pro 仍有微小差距。
- **Agent**：在公开基准上与 Kimi-K2.6 和 GLM-5.1 等领先开源模型持平。在内部评估中超越 Claude Sonnet 4.5，接近 Opus 4.5 的水平。
- **长上下文**：在百万 token 上下文窗口下表现出色，在学术基准上甚至超越了 Gemini-3.1-Pro。

> **Q：DeepSeek 坦言自己"落后前沿闭源模型 3-6 个月"——这种坦率在技术报告中非常少见。这个判断是否合理？**
>
> **A：** 这个判断有两层含义。表面上看是在标准推理基准上 V4-Pro-Max 略逊于 GPT-5.4/Gemini-3.1-Pro。但更深层的信息是：DeepSeek 试图表明他们的架构效率路线比闭源对手更激进——V4-Pro 用 49B 激活参数做到了接近 GPT-5.2 的推理水平。如果考虑到 FLOPs/性能比，DeepSeek 的效率优势可能远超 3-6 个月。这种"坦诚落后"反而是在强调效率维度的领先。

---

## 2. 架构

总体上，DeepSeek-V4 系列保留了 Transformer 架构和多 Token 预测（MTP）模块，同时相对于 DeepSeek-V3 引入了若干关键升级：（1）引入流形约束超连接（*m*HC）来增强传统残差连接；（2）设计混合注意力架构，结合压缩稀疏注意力和重度压缩注意力，大幅提升长上下文效率；（3）采用 Muon 优化器。

### 2.1 继承自 DeepSeek-V3 的设计

**混合专家（MoE）**。DeepSeek-V4 采用 DeepSeekMoE 范式用于前馈网络（FFN），设置细粒度的路由专家和共享专家。与 DeepSeek-V3 不同的是，计算亲和力分数的激活函数从 Sigmoid(·) 改为 Sqrt(Softplus(·))。同样采用辅助无损策略进行负载均衡。此外，前几层 Transformer 块中的密集 FFN 层被替换为采用哈希路由的 MoE 层——根据输入 token ID 的预定义哈希函数来确定目标专家。

**多 Token 预测**。沿用 DeepSeek-V3 中已验证的 MTP 策略，不作修改。

### 2.2 流形约束超连接

DeepSeek-V4 系列引入了流形约束超连接（*m*HC），以增强相邻 Transformer 块之间的传统残差连接。与朴素的超连接（HC）相比，*m*HC 的核心思想是将残差映射约束到特定流形上，从而增强信号在各层间传播的稳定性，同时保持模型表达力。

**标准超连接**。标准 HC 将残差流的宽度扩展 $n_{hc}$ 倍。残差状态的更新公式为：

$$X_{l+1} = B_l X_l + C_l \mathcal{F}_l(A_l X_l)$$

其中 $\mathcal{F}_l$ 表示第 $l$ 层的变换。HC 将残差宽度与实际隐藏层大小解耦，但在堆叠多层时训练经常出现数值不稳定。

**流形约束残差映射**。*m*HC 的核心创新在于将残差映射矩阵 $B_l$ 约束到双随机矩阵（Birkhoff 多面体）的流形 $\mathcal{M}$ 上：

$$B_l \in \mathcal{M} := \\{M \in \mathbb{R}^{n \times n} \mid M\mathbf{1}_n = \mathbf{1}_n,\; \mathbf{1}_n^T M = \mathbf{1}_n^T,\; M \geq 0\\}$$

这一约束确保映射矩阵的谱范数以 1 为上界，残差变换非扩张，增强数值稳定性。输入变换 $A_l$ 和输出变换 $C_l$ 通过 Sigmoid 函数约束为非负且有界。残差映射 $\tilde{B}_l$ 通过 Sinkhorn-Knopp 算法投影到双随机矩阵流形上（$t_{max} = 20$ 次迭代）。

> **Q：为什么要把 $B_l$ 约束到双随机矩阵流形上？**
>
> **A：** 这个约束解决了深度网络的一个根本问题：残差流中的信号在逐层传播时会发生缩放漂移。普通 HC 的 $B_l$ 是无约束的矩阵，其谱范数可能大于 1（导致信号爆炸）或远小于 1（导致信号消失）。双随机矩阵的谱范数恰好以 1 为上界，且保证了特征值 1 对应的特征向量是均匀向量——这意味着残差流的"平均信号强度"在传播中保持不变。论文虽未给出消融实验数据，但 DeepSeek 在 V3 的训练中显然遭遇过 HC 的不稳定问题，*m*HC 是对此的直接回应。

### 2.3 混合注意力：CSA 与 HCA

当上下文长度达到极端规模时，注意力机制成为模型的主要计算瓶颈。DeepSeek-V4 设计了两种高效的注意力架构——压缩稀疏注意力（CSA）和重度压缩注意力（HCA），并以交错混合的方式部署。CSA 先将每 $m$ 个 token 的 KV 缓存压缩为一条，再应用 DeepSeek 稀疏注意力（DSA），每个查询 token 仅关注 $k$ 个被压缩的 KV 条目。HCA 将每 $m'$（$\gg m$）个 token 的 KV 缓存合并为单条，但保持密集注意力。

#### 2.3.1 压缩稀疏注意力（CSA）

CSA 先将每 $m$ 个 token 的 KV 缓存压缩为一条（通过 Softmax 加权求和，且相邻压缩块的索引有重叠），然后通过**闪电索引器**以低秩方式生成索引器查询，计算查询与各压缩块之间的索引分数，用 top-k 选择器保留得分最高的压缩 KV 条目。最后以共享键值的 MQA 方式执行核心注意力，并使用分组输出投影降低计算负担。

#### 2.3.2 重度压缩注意力（HCA）

HCA 采用更大的压缩率 $m'$（$\gg m$）且不进行重叠压缩，不使用稀疏注意力，对所有压缩后的 KV 条目做密集注意力。同样采用共享键值 MQA 和分组输出投影。

#### 2.3.3 其他细节

- **查询和 KV 条目归一化**：核心注意力前对查询和 KV 条目执行 RMSNorm，避免注意力 logits 爆炸。
- **部分 RoPE**：仅在最后 64 维上应用 RoPE，并在注意力输出上应用反向 RoPE 以引入相对位置信息。
- **滑动窗口注意力分支**：为 CSA 和 HCA 额外引入 $n_{win}$ 大小的滑动窗口，捕获局部细粒度依赖。
- **注意力汇聚**：设置可学习的汇聚 logits，允许注意力总分不等于 1。

#### 2.3.4 效率讨论

采用混合存储格式（RoPE 维度 BF16 + 其余 FP8），闪电索引器以 FP4 精度计算。以 BF16 GQA8（head dimension 128）为基线，DeepSeek-V4 在百万 token 上下文下 KV 缓存可降至该基线的约 2%。

> **Q：CSA 和 HCA 的混合交错策略是如何确定的？为什么不全部使用 CSA 或全部使用 HCA？**
>
> **A：** 这是一个计算量/信息保真度的权衡。CSA 保留了稀疏选择机制，能够精准定位长距离关键信息，但其索引器本身也需要计算。HCA 压缩更激进，KV 缓存更小，但放弃了稀疏选择。将两者交错使用的直觉是：某些层需要精确的长距离检索（CSA），另一些层只需粗粒度的全局上下文感知（HCA）。前两层用纯滑动窗口注意力（不需要长距离），后续层交错 CSA 和 HCA——这种分层策略让不同层专注于不同的信息粒度。

### 2.4 Muon 优化器

大部分模块采用 Muon 优化器（嵌入、预测头、*m*HC 静态偏置和 RMSNorm 仍用 AdamW）。使用混合 Newton-Schulz 迭代进行正交化——前 8 步快速收敛系数 $(a,b,c) = (3.4445, -4.7750, 2.0315)$，后 2 步精确稳定系数 $(a,b,c) = (2, -1.5, 0.5)$。由于注意力架构允许直接对查询和 KV 条目应用 RMSNorm，因此不需要 QK-Clip 技术。

---

## 3. 通用基础设施

### 3.1 专家并行中的细粒度通信-计算重叠

关键洞察：在 MoE 层中，通信延迟可被有效隐藏在计算之下。我们将专家拆分和调度为多个波次（wave），形成细粒度流水线，理论加速比从 Comet 方案的 1.42× 提升到 1.92×。已开源基于 CUDA 的 mega-kernel 实现 **MegaMoE**（[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM/pull/304) 组件）。

向硬件厂商的建议：瞄准计算-通信平衡点（每 GBps 互连带宽隐藏 6.1 TFLOP/s 计算）；为完全并发工作负载提供充足功耗余量；采用拉取式通信原语；考虑更简单的激活函数替代 SwiGLU。

### 3.2 使用 TileLang 的灵活高效核开发

采用 TileLang（DSL）开发融合核替代数百个细粒度 Torch ATen 算子。通过 Host Codegen 将 CPU 端验证开销从几十微秒降至不到一微秒。集成 Z3 SMT 求解器进行形式化整数分析。默认禁用 fast-math，确保逐位可复现性。

### 3.3 高性能批次不变和确定性核库

端到端实现批次不变、逐位确定性的核。注意力采用双核策略解决波量化问题；矩阵乘法端到端使用 DeepGEMM 替代 cuBLAS。确定性实现覆盖注意力反向传播、MoE 反向传播和 *m*HC 矩阵乘法。

### 3.4 FP4 量化感知训练

将 FP4 量化应用于 MoE 专家权重和 CSA 索引器 QK 路径。关键发现：FP4-to-FP8 反量化是无损的（FP8 E4M3 比 FP4 E2M1 拥有更大动态范围），使整个 QAT 流水线可完全复用现有 FP8 框架。索引分数从 FP32 量化到 BF16，top-k 加速 2×，KV 条目召回率 99.7%。

> **Q：FP4 量化到 FP8 的反量化居然是无损的？**
>
> **A：** 这个断言在数学上是精确的，但有关键前提：FP4 量化块内各元素的 scale 因子变化范围必须在 FP8 的动态范围内。DeepSeek "经验验证了当前权重满足此条件"——这是一个与训练动态强相关的经验发现，但确实是一个非常巧妙的工程洞察，使他们跳过了 FP4 专用推理框架的开发。

### 3.5 训练框架

- **Muon 的高效实现**：混合 ZeRO bucket 分配，密集参数用背包算法分配，MoE 参数独立优化。Newton-Schulz 迭代在 BF16 下保持稳定，MoE 梯度量化到 BF16 同步以减半通信量。
- ***m*HC 实现**：融合核 + 选择性检查点 + DualPipe 1F1B 调整，墙钟时间开销仅 6.7%。
- **上下文并行**：两阶段通信方案处理压缩注意力在 CP rank 边界的跨越问题。
- **张量级别激活检查点**：基于 TorchFX 的自动微分扩展，支持细粒度的重计算控制。

### 3.6 推理框架

异构 KV 缓存结构：经典 KV 缓存（CSA/HCA 压缩后）+ 状态缓存（SWA + 待压缩尾部 token）。磁盘 KV 缓存存储支持三种 SWA 策略（完整缓存/周期性检查点/零缓存），在存储和计算间权衡。

---

## 4. 预训练

### 4.1 数据构建

预训练语料超过 32T token，包含数学、代码、网页、长文档等。特别重视长文档数据策展。使用样本级注意力掩码（区别于 V3）。

### 4.2 预训练设置

**DeepSeek-V4-Flash**：43 层，d=4096，284B 总参数/13B 激活。CSA 压缩率 m=4，HCA 压缩率 m'=128，256 个路由专家，32T token 训练。

**DeepSeek-V4-Pro**：61 层，d=7168，1.6T 总参数/49B 激活。CSA top-k=1024，384 个路由专家，33T token 训练。

### 4.3 缓解训练不稳定性

两个关键技术：

**前瞻路由（Anticipatory Routing）**：将骨干网络和路由网络的同步更新解耦——步骤 t 的特征计算用当前参数，路由索引用历史参数 $\theta_{t-\Delta t}$。配合自动检测机制，仅在 loss spike 时动态激活。

**SwiGLU 截断**：线性分量截断到 [-10, 10]，门分量上界 10。

> **Q：前瞻路由的原理是什么？为什么将路由和骨干网络解耦就能稳定训练？**
>
> **A：** 在 MoE 中存在一个恶性循环：异常激活值 → 被路由到特定专家 → 梯度被异常值主导 → 权重更新导致更大异常值 → 更极端路由分配。使用历史参数计算路由打断了这一循环。有趣的是，作者坦言"对其底层机制缺乏全面的理论理解"——这在工业级大模型训练中非常典型：先用工程直觉找到有效的解法，理论解释留待后续。

### 4.4 评估结果

DeepSeek-V4-Flash-Base 凭借更紧凑高效的参数设计，在多数基准上超越 DeepSeek-V3.2-Base。DeepSeek-V4-Pro-Base 在几乎所有类别中全面领先，成为 DeepSeek 基础模型系列最强标杆。

---

## 5. 后训练

### 5.1 后训练流水线

关键方法论替换：混合 RL 阶段被完全替换为**在策略蒸馏（OPD）**。

**专家训练**：每个领域（数学、编程、agent、指令跟随）独立训练专家模型（SFT → GRPO RL）。支持三种推理强度模式（Non-think / Think High / Think Max）。

**生成式奖励模型（GRM）**：actor 网络原生充当 GRM，统一评价能力和生成能力的联合优化。

**交错思考**：工具调用场景中完全保留跨轮次推理历史（利用 1M 上下文窗口）。

**快速指令**：追加专用特殊 token 执行辅助任务（action、title、query 等），复用 KV 缓存避免冗余预填充。

> **Q：用 OPD 完全替代混合 RL 阶段，为什么 DeepSeek 认为这更好？**
>
> **A：** 传统混合 RL 试图用一个 RL 阶段同时优化多个目标，不同目标间存在梯度干扰。OPD 让每个领域的专家独立达到最优，然后通过 KL 散度在 logits 层面融合。这绕过了多目标 RL 的干扰问题，且蒸馏在学生自己的轨迹上进行（on-policy），避免分布偏移。论文提到使用了超过 10 个教师模型。

### 5.2 在策略蒸馏（OPD）

OPD 目标函数：$\mathcal{L}\_{\text{OPD}}(\theta) = \sum\_{i=1}^{N} w_i \cdot D\_{\text{KL}}(\pi\_\theta \| \pi\_{E_i})$。采用全词表 logit 蒸馏（而非逐 token KL 估计），产生更稳定的梯度。

### 5.3 RL 和 OPD 基础设施

- FP4 量化集成：rollout 阶段直接使用原生 FP4 权重
- 全词表 OPD 的高效教师调度：仅缓存最后一层隐藏状态，按需重建完整 logits
- 可抢占容错 Rollout 服务：逐 token WAL，避免长度偏差
- 百万 Token 上下文 RL 框架：分离轻量级元数据和重型逐 token 字段
- **DSec 沙箱平台**：Rust 实现，四种执行基底（Function Call / Container / microVM / fullVM），管理数十万并发沙箱

### 5.4 评估结果

**知识**：SimpleQA-Verified 领先所有开源基线 20 个绝对百分点。**推理**：Codeforces 排名第 23 位，Putnam-2025 达 proof-perfect 120/120。**Agent**：MCPAtlas 和 Toolathlon 表现优秀，泛化能力出色。**长上下文**：MRCR 超越 Gemini-3.1-Pro。**中文写作**：功能性写作胜率 62.7%。**代码 Agent**：通过率 67%，接近 Opus 4.5 水平。

---

## 6. 结论、局限与未来方向

DeepSeek-V4 系列旨在打破超长上下文处理的效率壁垒。通过混合注意力架构和广泛的基础设施优化，实现了对百万 token 上下文的高效原生支持。

**局限**：架构复杂度较高（保留了许多初步验证的组件和技巧）；前瞻路由和 SwiGLU 截断的底层原理仍未被充分理解。

**未来方向**：精简架构到最核心设计；探索更多稀疏性维度；低延迟架构和系统技术；整合多模态能力。

> **Q：DeepSeek-V4 的架构复杂度是否已经达到了一个需要"简化回退"的临界点？**
>
> **A：** 作者自己承认了这一点。V4 同时使用了 CSA、HCA、DSA、闪电索引器、*m*HC（含 Sinkhorn-Knopp 迭代）、Muon 优化器、前瞻路由、SwiGLU 截断、FP4 QAT、哈希路由等大量技术。每个单独看都有道理，但组合在一起的交互效应和调试难度是指数级增长的。DeepSeek 做了一个务实的选择：先发布一个"能工作"的版本，然后在后续迭代中做减法。

---

## 术语表

| 英文 | 中文翻译 |
|------|---------|
| Compressed Sparse Attention (CSA) | 压缩稀疏注意力 |
| Heavily Compressed Attention (HCA) | 重度压缩注意力 |
| Manifold-Constrained Hyper-Connections (*m*HC) | 流形约束超连接 |
| On-Policy Distillation (OPD) | 在策略蒸馏 |
| Quantization-Aware Training (QAT) | 量化感知训练 |
| Anticipatory Routing | 前瞻路由 |
| Generative Reward Model (GRM) | 生成式奖励模型 |
| Quick Instruction | 快速指令 |
| Interleaved Thinking | 交错思考 |
| Attention Sink | 注意力汇聚 |
| Batch Invariance | 批次不变性 |
