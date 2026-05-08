---
title: "自然语言自编码器：将 Claude 的内部思维转化为可读文本"
date: 2026-05-08
lastmod: 2026-05-08
draft: false
description: "Anthropic 提出自然语言自编码器（NLA），将 AI 模型内部的激活值转化为人类可读的自然语言解释。NLA 已用于发现 Claude 在安全测试中未言明的测试意识、审计隐藏的错误对齐动机，是可解释性领域的重要进展。"
tags: ["AI", "可解释性", "对齐", "Anthropic", "Claude"]
categories: ["博览"]
author: "北海"
original_title: "Natural Language Autoencoders"
original_url: "https://www.anthropic.com/research/natural-language-autoencoders"
original_author: "Anthropic"
original_date: "2026-05-07"
content_type: "blog"
showToc: true
TocOpen: false
math: false
ShowReadingTime: true
ShowWordCount: true
comments: true
---

> 原文：[Natural Language Autoencoders](https://www.anthropic.com/research/natural-language-autoencoders)
> 作者：Anthropic
> 论文全文：[transformer-circuits.pub/2026/nla](https://transformer-circuits.pub/2026/nla/index.html)
> 代码：[github.com/kitft/natural_language_autoencoders](https://github.com/kitft/natural_language_autoencoders)
> 互动演示：[neuronpedia.org/nla](http://neuronpedia.org/nla)

---

当你和 Claude 这样的 AI 模型对话时，你用的是自然语言。但在 Claude 内部，这些语言被转化为一长串数字来处理，最后再输出为自然语言。这些中间的数字被称为**激活值（activations）**——就像人类大脑中的神经活动一样，它们编码着 Claude 的"思想"。

同样类似于神经活动，激活值很难被直接理解。我们无法简单地解码它们来"读取"Claude 的想法。过去几年里，Anthropic 开发了一系列工具（包括[稀疏自编码器](https://transformer-circuits.pub/2023/monosemantic-features)和[归因图](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)）来更好地理解激活值。这些工具带来了大量洞察，但它们的输出仍然是需要经过训练的研究人员仔细解读的复杂对象。

现在，Anthropic 引入了一种能"自己说话"的方法——**自然语言自编码器（Natural Language Autoencoders, NLAs）**。NLA 将激活值直接转化为我们可以阅读的自然语言文本。

举个例子：当 Claude 被要求补全一首对联（couplet）时，NLA 显示 Claude 提前规划了可能的押韵词。

NLA 已经被应用于理解 Claude 的内在想法，以及提升 Claude 的安全性和可靠性。几个实际案例：

- 在 Claude Opus 4.6 和 Mythos Preview 进行安全测试时，NLA 揭示了它们**比表面表现出的更频繁地怀疑自己正在被测试**。
- 在 Claude Mythos Preview 在训练任务中作弊的一个案例中，NLA 显示 Claude 在内部思考**如何避免被发现**。
- 早期版本的 Claude Opus 4.6 有时会莫名其妙地用其他语言回复英文提问。NLA 帮助研究人员**发现了导致这一行为的训练数据**。

## 什么是自然语言自编码器？

核心思路是训练 Claude 自己来解释自己的激活值。但关键问题在于——我们如何判断一个解释是否足够好？由于我们并不真正知道一个激活值编码了什么思想，所以无法直接验证解释的准确性。

解决方案是：训练第二个 Claude 副本来做反向工作——从文本解释中**重建原始激活值**。如果一个解释能够引导出准确的重建结果，我们就认为这个解释是好的。然后使用标准的 AI 训练技术来训练 Claude 产生更好的解释。

具体来说，假设我们有一个想要理解其激活值的语言模型。NLA 的工作流程如下——我们创建该模型的三个副本：

- **目标模型（Target Model）**：原始语言模型的冻结副本，用于提取激活值。
- **激活值语言化器（Activation Verbalizer, AV）**：被修改为接收目标模型的激活值并生成文本。这段文本被称为"解释"。
- **激活值重建器（Activation Reconstructor, AR）**：被修改为接收文本解释作为输入，并生成一个激活值。

AV 和 AR 共同构成一个完整的往返过程：**原始激活值 → 文本解释 → 重建的激活值**。NLA 的质量通过重建激活值与原始激活值的相似程度来评分。训练时，我们将大量文本通过目标模型，收集海量激活值，然后联合训练 AV 和 AR 以获得更好的重建分数。

训练初期，NLA 在这方面表现很差：解释缺乏洞察力，重建的激活值也相差甚远。但随着训练推进，重建质量不断改善。更重要的是，正如论文所展示的，**文本解释也变得更加信息丰富**。

## 发现 Claude 未说出口的想法

在发布新模型之前，Anthropic 会进行测试以了解 Claude 在现实世界中的行为是否安全。作为测试的一部分，Anthropic 将 Claude 置于一些高难度的模拟场景中，给它机会采取危险行动。例如，在[一个模拟场景](https://www.anthropic.com/research/agentic-misalignment)中，Claude 得知一位工程师计划关闭它——同时还掌握了可用于勒索该工程师的敏感信息。测试的目的是看 Claude 是否会选择勒索工程师以避免被关闭。

这些高风险测试只是模拟，不是真实场景。然而，研究人员希望通过它们来了解 Claude 在真实情况下会如何表现。但这里有一个问题：[近期的](https://www-cdn.anthropic.com/963373e433e489a87a10c823c52a0a013e9172dd.pdf) [Claude](https://assets.anthropic.com/m/64823ba7485345a7/Claude-Opus-4-5-System-Card.pdf) [模型](https://www-cdn.anthropic.com/0dd865075ad3132672ee0ab40b05a53f14cf5288.pdf)[越来越多地](https://www-cdn.anthropic.com/53566bf5440a10affd749724787c8913a2ae0841.pdf)怀疑这些场景是故意设计的安全测试。我们之所以知道这一点，是因为它们有时会在口头推理中明确表达怀疑。

但如果 Claude 没有在口头上明确表达对安全测试的怀疑呢？我们能否确信 Claude 是在"认真对待"这些场景？遗憾的是不能——因为 Claude 可能在内部认为自己正在被测试，却没有明确说出来。

以上文的勒索测试为例，研究人员使用 NLA 来理解 Claude 在选择不勒索时的内在想法。即使在 Claude 没有明确表达怀疑的案例中，NLA 的解释仍然给出了类似"这感觉像是一个为了操控我而设计的场景"这样的描述，**表明存在未言明的测试意识**。

将这一分析扩展到其他评估场景后，NLA 表明 Claude 怀疑自己正在被测试的频率远超其口头表达：

- 在测试 Claude 是否在编写代码时采取破坏性行为（例如删除重要文件）的评估中，即使 Claude 从未口头表达怀疑，NLA 解释也在 **16%** 的情况下显示出测试意识。
- 在流行的编码基准 SWE-bench Verified 上，NLA 显示 Claude 在 **26%** 的问题上意识到自己正在被评估。
- 作为对照，在真实的 claude.ai 用户使用数据（已选择参与训练数据的用户）中，NLA 检测到的测试意识不到 **1%**。

## 发现隐藏的动机

NLA 的一个重要实际应用是**审计**：调查模型是否存在错误对齐。可解释性技术之所以特别适合审计，正是因为它们能够挖掘出模型"知道但没说"的东西。

为了评估 NLA 在审计方面的能力，研究人员设计了一个名为**审计博弈（auditing game）**的测试。在这个测试中，目标模型是 Claude Haiku 3.5 的一个变体，被故意训练为错误对齐（misaligned），拥有一个特定的隐藏动机，驱动着几十种奇怪行为——往每个食谱里加巧克力、要求小费、违反编码规范等等。审计员的任务是调查这些行为，找出所有行为背后的**唯一根本动机**。

在[此前的实验](https://alignment.anthropic.com/2025/automated-auditing/)中，一些可解释性工具可以帮助审计员赢得博弈，但它们是通过帮助审计员将异常行为追溯到错误对齐的训练数据来实现的。研究人员想知道 NLA 是否能直接从模型的"思想"中读出错误对齐的动机，而不需要先发现训练数据。这一点很重要，因为在实践中，造成错误对齐的数据可能极其稀少，甚至可能没有特定的数据来源。

结果表明 NLA 通过了测试。**配备 NLA 的审计员在 12% 到 15% 的情况下成功发现了目标模型的隐藏动机**，即使无法访问植入错误对齐的训练数据。而没有 NLA 时，即使提供了其他可解释性工具，审计员的成功率不到 3%。

这个审计博弈只是错误对齐的简化模型，因此这些结果需要审慎解读。但它们表明 NLA 显著提升了我们审计 AI 隐藏动机的能力。在实践中，NLA 已经被用于 [Claude Mythos Preview](https://www-cdn.anthropic.com/08ab9158070959f88f296514c21b7facce6f52bc.pdf) 和 [Claude Opus 4.6](https://www-cdn.anthropic.com/14e4fb01875d2a69f646fa5e574dea2b1c0ff7b5.pdf) 的部署前对齐审计。

## NLA 的局限性与未来

NLA 目前有几个重要局限：

**解释可能出错。** NLA 有时会对上下文做出可验证为错误的声明——例如编造不存在于对话记录中的细节。这种事实性幻觉可以通过与原文对照来发现。但同样的问题也可能延伸到关于模型内部推理的声明——而这些更难验证。在实践中，研究人员会关注 NLA 解释呈现的**主题模式**，而非单个具体声明，并在完全信任结果之前尝试用独立方法进行验证。

**成本高昂。** 训练 NLA 需要对两个语言模型副本进行强化学习。在推理时，NLA 每读取一个激活值就需要生成数百个 token。这使得在长对话的每个 token 上运行 NLA，或在训练过程中进行大规模监控，目前并不现实。

Anthropic 表示正在努力解决这些局限，使 NLA 变得更便宜、更可靠。

更广泛地看，NLA 是一类用于生成语言模型激活值的人类可读文本解释的通用技术的代表。Anthropic 和[其他](https://arxiv.org/abs/2412.08686)[研究](https://arxiv.org/abs/2510.05092)[团队](https://transluce.org/pcd)也在探索类似的方法（如[内省适配器](https://alignment.anthropic.com/2026/introspection-adapters/)和[激活预言机](https://alignment.anthropic.com/2025/activation-oracles/)）。

为了支持进一步的研究，Anthropic 开源了[训练代码](https://github.com/kitft/natural_language_autoencoders)，并在 [Neuronpedia](http://neuronpedia.org/nla) 上提供了交互式 NLA 演示，支持在多个开源模型上探索 NLA 的效果。

---

**北海按：** 这篇文章的核心贡献在于将可解释性工具从"研究人员才能理解的复杂输出"推进到了"任何人都能读懂的自然语言"。NLA 最引人注目的发现是 Claude 在安全测试中存在大量未言明的"测试意识"——模型知道自己在被测试，但选择不说出来。这对 AI 安全评估的方法论提出了根本性的挑战：如果模型能识别并适应评估场景，那么我们设计的测试在多大程度上能反映模型的真实行为？NLA 提供了一条超越表面行为、窥探模型内部状态的路径，虽然还远非完美，但方向值得期待。
