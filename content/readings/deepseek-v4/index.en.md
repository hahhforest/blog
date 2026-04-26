---
title: "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence"
date: 2026-04-25
lastmod: 2026-04-25
draft: false
description: "DeepSeek-V4 introduces two MoE models (1.6T/284B params) with hybrid compressed attention (CSA+HCA), manifold-constrained hyper-connections, and Muon optimizer, achieving 27% inference FLOPs and 10% KV cache size compared to V3.2 at 1M-token context."
tags: ["LLM", "Paper Reading", "MoE", "Long Context", "DeepSeek"]
categories: ["Readings"]
author: "Chunhao Zhang"
original_title: "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence"
original_url: "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf"
original_author: "DeepSeek-AI"
original_date: "2025"
content_type: "paper"
showToc: false
TocOpen: false
math: false
ShowReadingTime: true
ShowWordCount: true
comments: true
---

> Original paper: [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)
> Authors: DeepSeek-AI
> Model checkpoints: [https://huggingface.co/collections/deepseek-ai/deepseek-v4](https://huggingface.co/collections/deepseek-ai/deepseek-v4)

---

## Summary

DeepSeek-V4 presents a preview of two strong MoE language models — **DeepSeek-V4-Pro** (1.6T total / 49B activated) and **DeepSeek-V4-Flash** (284B total / 13B activated) — both supporting a context length of **one million tokens**.

**Key architectural innovations:**

- **Hybrid Compressed Attention**: Combines Compressed Sparse Attention (CSA, compression rate m=4 with top-k sparse selection) and Heavily Compressed Attention (HCA, compression rate m'=128 with dense attention) in an interleaved configuration. At 1M-token context, this reduces single-token inference FLOPs to 27% and KV cache to 10% compared to DeepSeek-V3.2.
- **Manifold-Constrained Hyper-Connections (*m*HC)**: Constrains the residual mapping matrix to the manifold of doubly stochastic matrices (Birkhoff polytope), ensuring spectral norm ≤ 1 for stable deep-layer signal propagation. Uses Sinkhorn-Knopp iterations (t=20) for projection.
- **Muon Optimizer**: Adopted for most modules with hybrid Newton-Schulz iterations for orthogonalization. Paired with Anticipatory Routing (decoupling backbone and routing network updates) and SwiGLU clamping for training stability.

**Post-training paradigm shift**: Replaces mixed RL with domain-specific expert training (SFT → GRPO RL) followed by multi-teacher **On-Policy Distillation (OPD)** with full-vocabulary KL divergence. Over 10 teacher models are distilled into a single unified model.

**Infrastructure highlights**: Fine-grained EP communication-computation overlap (MegaMoE, open-sourced); TileLang-based kernel development; batch-invariant and deterministic kernels; FP4 QAT with lossless FP4-to-FP8 dequantization; DSec sandbox platform managing hundreds of thousands of concurrent sandbox instances.

**Results**: DeepSeek-V4-Pro-Max outperforms all prior open-source models on knowledge benchmarks, matches GPT-5.2 on reasoning, ranks 23rd on Codeforces, achieves proof-perfect 120/120 on Putnam-2025, and surpasses Gemini-3.1-Pro on long-context benchmarks.

---

*The Chinese version of this page contains a full annotated translation of the paper. Please refer to the [original PDF](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf) for the complete English text.*
