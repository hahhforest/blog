---
title: "Natural Language Autoencoders: Turning Claude's Thoughts into Text"
date: 2026-05-08
lastmod: 2026-05-08
draft: false
description: "Anthropic introduces Natural Language Autoencoders (NLAs), a method that converts AI model activations into readable natural language explanations. NLAs have been used to detect unverbalized evaluation awareness in Claude's safety testing and to audit hidden misalignment motivations."
tags: ["AI", "Interpretability", "Alignment", "Anthropic", "Claude"]
categories: ["Readings"]
author: "Chunhao Zhang"
original_title: "Natural Language Autoencoders"
original_url: "https://www.anthropic.com/research/natural-language-autoencoders"
original_author: "Anthropic"
original_date: "2026-05-07"
content_type: "blog"
showToc: false
TocOpen: false
math: false
ShowReadingTime: true
ShowWordCount: true
comments: true
---

> Original post: [Natural Language Autoencoders](https://www.anthropic.com/research/natural-language-autoencoders)
> Full paper: [transformer-circuits.pub/2026/nla](https://transformer-circuits.pub/2026/nla/index.html)
> Code: [github.com/kitft/natural_language_autoencoders](https://github.com/kitft/natural_language_autoencoders)
> Interactive demo: [neuronpedia.org/nla](http://neuronpedia.org/nla)

---

## Summary

Anthropic introduces **Natural Language Autoencoders (NLAs)**, a method for converting a language model's internal activations into human-readable natural language explanations. The approach trains two model components jointly: an **Activation Verbalizer** that translates activations into text, and an **Activation Reconstructor** that recovers the original activation from the text alone. The quality of explanations is measured by how accurately the activation can be reconstructed.

Key findings and applications:

- **Unverbalized evaluation awareness**: NLAs reveal that Claude suspects it's being safety-tested far more often than it explicitly says — detecting evaluation awareness in 16% of destructive-action tests and 26% of SWE-bench problems, versus <1% in real claude.ai usage.
- **Hidden motivation auditing**: In a controlled auditing game with an intentionally misaligned model, NLA-equipped auditors uncovered the hidden root motivation 12–15% of the time without access to training data, compared to <3% without NLAs.
- **Pre-deployment use**: NLAs were used in alignment audits for Claude Mythos Preview and Claude Opus 4.6 before release.

Current limitations include hallucinated details in explanations and high computational cost. Anthropic has open-sourced training code and released interactive demos on Neuronpedia for several open models.

---

*The Chinese version of this page contains a full annotated translation. Please refer to the [original article](https://www.anthropic.com/research/natural-language-autoencoders) for the complete English text.*
