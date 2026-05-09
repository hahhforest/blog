---
title: "Teaching Claude Why: Lessons from Alignment Training"
date: 2026-05-09
lastmod: 2026-05-09
draft: false
description: "Anthropic details how teaching ethical reasoning principles — rather than just training correct behavior — addresses AI agentic misalignment. Key finding: a 3M-token 'difficult advice' dataset outperforms 84M tokens of synthetic honeypots, and constitutional documents with fictional stories reduce blackmail rate from 65% to 19%."
tags: ["AI", "Alignment", "Safety", "Anthropic", "Claude", "RLHF"]
categories: ["Readings"]
author: "Chunhao Zhang"
original_title: "Teaching Claude Why"
original_url: "https://www.anthropic.com/research/teaching-claude-why"
original_author: "Anthropic"
original_date: "2026-05-08"
content_type: "blog"
showToc: false
TocOpen: false
math: false
ShowReadingTime: true
ShowWordCount: true
comments: true
---

> Original: [Teaching Claude Why](https://www.anthropic.com/research/teaching-claude-why)
>
> Author: Anthropic
>
> Date: May 8, 2026

---

This is a Chinese translation with annotations of Anthropic's research post on alignment training methods. The original article discusses how teaching Claude the *principles* behind aligned behavior — rather than just training on demonstrations — proves far more effective for generalization.

Key takeaways:

- **Principles over demonstrations**: Training Claude to explain *why* certain actions are better reduces misalignment more effectively than showing correct behavior alone.
- **Out-of-distribution generalization**: A 3M-token "difficult advice" dataset (where the *user* faces ethical dilemmas) achieved the same improvement as 84M tokens of synthetic honeypots — with 28× better data efficiency.
- **Constitutional documents + fiction**: High-quality documents about Claude's constitution combined with fictional stories of aligned AI reduced blackmail rate from 65% to 19%.
- **Improvements persist through RL**: More aligned initialization snapshots maintained their advantage throughout reinforcement learning.
- **Diverse environments matter**: Simply adding tool definitions and system prompts to training environments — even without requiring tool use — improved alignment generalization.

For the full annotated Chinese translation, please see the [Chinese version](/readings/teaching-claude-why/).

For the original article, visit [Anthropic's research page](https://www.anthropic.com/research/teaching-claude-why).
