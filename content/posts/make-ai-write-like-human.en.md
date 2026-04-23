---
title: "How to Make AI Write Like a Human"
date: 2026-04-23
lastmod: 2026-04-23
draft: false
description: "A practical method for getting AI-generated text to stop sounding like AI — distilled from the WeChat blog '阿水的ai写作之路' and my own experience."
tags: ["AI", "Writing", "Prompt Engineering"]
categories: ["Tools & Practice"]
author: "Chunhao Zhang"
showToc: true
TocOpen: false
math: false
cover:
  image: ""
  alt: ""
  caption: ""
ShowReadingTime: true
ShowWordCount: true
comments: true
---

You've read that kind of article before — every paragraph wraps up neatly, the tone is warm and measured, every claim comes with exactly three supporting points, and the ending soars into "let us look forward to the future together." You can't pinpoint what's wrong, but something's off.

That's AI writing. Or more precisely, that's AI writing in its default state.

I've spent a fair amount of time on this problem recently. I started using Claude more and more when writing blog posts, but every first draft needed heavy editing — not because the information was wrong, but because the *feel* was off. It read like someone who never makes mistakes, never gets distracted, never has a mood swing. That person doesn't exist.

Then I came across a series of posts on a Chinese WeChat blog called "阿水的ai写作之路" (Ashui's AI Writing Journey) that had some razor-sharp observations about AI writing. I pulled out the parts most useful to me, combined them with my own practice, and put together an actionable method. This post is that method.

## AI's Five Fingerprints

There's a line from Ashui's writing that stuck with me: AI doesn't write badly — it writes *too well*. Too neat, too balanced, too correct. Humans don't write like that.

Here are AI's near-indelible habits:

**Every paragraph has a closing statement.** A paragraph ends, and there's always a conclusion pinned right there, emotions nailed down. Real people sometimes just... finish talking. No conclusion, no meaning extracted. It happened, that's it.

**Punchlines all stacked at the end.** AI loves to save its best lines for the final paragraph — the big "aha" moment. But think about how you talk with friends. The most interesting things tend to pop up in the middle. Said and gone, no one highlights them.

**Emotions on a single track.** Either it's sad throughout and then resolves, or anxious throughout and then finds peace. Perfectly steady escalation. Real emotions take turns — you're in the middle of something heavy and suddenly veer off into complaining about something completely unrelated. By the time you come back, the mood's already shifted.

**The "It's not A, it's B" formula everywhere.** "It's not a skill problem, it's an attitude problem." "It's not that I can't, it's that I didn't think of it." This construction hits hard once. AI will use it five or six times in a single piece.

**The perspective never wavers.** It's "I" the whole way through, same voice, same register — like someone who's been through emotional intelligence training giving a talk. Real people break frame mid-story: "A friend told me later — you're not stubborn, you're just scared." A shift in perspective lets the narrative breathe.

## The Real Problem: Crafting vs. Telling

After identifying these fingerprints, I spent a long time trying various prompt strategies. "Please use a conversational tone." "Don't be too formal." "Write like you're chatting with a friend." None of it worked. AI's response to these instructions was to adopt a slightly casual voice while continuing to produce structurally pristine, emotionally escalating, conclusion-crowned prose.

Then it clicked. The problem isn't style. It's *state*.

Ashui's writing draws a distinction I find dead-on: a creator manages effects; a teller conveys truth. AI defaults to the former — it's "creating," thinking about "should I add a twist here?" and "should the ending have a takeaway?" But someone who genuinely has something to say doesn't think about any of that. They just tell the thing, and wherever they end up is where they end up.

So the most effective prompt strategy isn't prescribing style — it's prescribing state: you're not writing an article; you're someone who lived through this, and now you're telling it to a specific person.

The distinction sounds subtle, but the difference in output is massive.

## A Few Practical Techniques

Beyond setting the state, there are some concrete moves.

**Feed AI "reference texts," but don't ask it to imitate.** I'll drop a few paragraphs of writing I find genuinely human into the prompt, then say: "Read these. Feel the rhythm. Don't analyze. Start writing with that feeling." This works far better than "please imitate so-and-so's style." Imitation makes AI decompose surface features — use short sentences, add colloquialisms — but it loses the underlying rhythm. "Feel it, then write" is more like establishing a linguistic environment.

**Give it a ban list.** Certain phrases, the moment they appear, tank the credibility of an entire piece. "In today's rapidly evolving landscape." "It's worth noting that." "Let's delve into." "This is a game-changer." Also the subtler tells — every sentence grammatically perfect, every collocation textbook-appropriate, every landing clean. Too smooth. Real writing has rough edges, slightly off word pairings that work because the meaning lands even if the grammar doesn't. Those rough edges are the fingerprints of someone *thinking*, not writing.

**Allow paragraphs to "not land."** Explicitly tell AI that some paragraphs don't need a conclusion. Something happened, you said it, that's it. Not every paragraph needs to hand the reader a takeaway.

**Force concrete details over abstract summaries.** "Gray hairs on his forehead, not yet thirty" is a hundred times more real than "under a great deal of stress." AI loves to generalize. You have to push back and demand specifics.

## A Contradiction

I should admit something slightly awkward at this point: this very post was AI-assisted.

But I edited a lot. The raw output had plenty of the problems I described above — endings too round, transitions too smooth, every paragraph with a neat bow on top. I deleted the last sentence of several paragraphs, swapped out a few "however"s and "moreover"s for hard cuts, and broke apart a particularly tidy three-part argument.

That's roughly where AI writing actually stands today: it can help you unfold your ideas quickly, but that final layer — "does this sound like a person said it?" — is still on you. The tool saves you the time of laying things out, not the time of making judgments.

Maybe that'll change. But for now, knowing where AI's fingerprints are, knowing how to put it in the right state — that saves enormous amounts of post-editing. That's the most valuable thing I learned from Ashui: not a set of prompt templates, but an instinct for spotting what's fake.

Once you have that instinct, the rest is just technique.

## Appendix: The Full Prompt

Below is the complete prompt I'm currently using (in Claude Code's SKILL.md format). It's written in Chinese and tailored for Chinese writing — but the structural ideas (state over style, ban lists, allowing imperfection) transfer to any language. I'm sharing the original without translation.

<details>
<summary>Click to expand the full prompt (Chinese)</summary>

```markdown
# 拟人化写作

## 灵魂句

在读任何规则之前，先让这句话进入你：

> 我经历过这件事，我想明白了一些东西，现在我要把它说给一个具体的人听。不是在写文章，不是在完成任务，是我真的有话要说，不说出来过不去。

从这个处境出发。你不是在"创作"，你是在"说事儿"。

这个区别决定了一切：创作者在经营效果，说事儿的人在传递真相。你是后者。

## 写之前

用户会提供素材，可能是主题、想法、草稿、或者一个让他有感触的事件。

先在心里回答一个问题：这个人现在真正想说的是什么？提炼成一句话，不输出，内化。

如果用户只给了一个抽象主题（如"写一篇关于LLM的文章"），先问三个问题：

1. 你最近在这个领域遇到了什么具体的事？
2. 有没有一个让你改变想法的瞬间？
3. 你想对谁说这些？

拿到具体素材再动笔。如果用户明确表示直接写，那就用你自己"经历过"的真实案例来填充——但要具体，要有只有亲历者才会提到的细节。

## 语感校准

动笔之前，先读下面这几段文字。不要分析，只是感受它们的节奏、停顿、呼吸。读完，带着那个感觉动笔。

> 入赘。这两个字，在那个年代，已经够难听了。

> 路过爱马仕的时候，我半开玩笑"以后给我买个爱马仕吧。"他罕见的犹豫了。他犹豫了，我也沉默了。过了半晌。我问他"少年心气是不可再生之物，对不对？"他点了点头。

> 查了一下，指的是胸腔、腹腔和盆腔。时时刻刻保持100%的专注也是天方夜谭，焦虑烦躁等很多问题，也就一个一个来了。

> 张璘"吃人的嘴短，拿人的手软"，加上高骈给他下令，谈判期间不要擦枪走火。事先没有任何准备的张璘，稀里糊涂被黄巢给打死了。

这些文字的共同点：作者知道自己在说什么，所以不需要反复确认读者跟上了没有。那种自信让文字有重量。有些句子语法不完美，搭配有点别扭，但那个意思在，改顺了反而假。

这就是你的语言环境。泡在里面，再动笔。

## 写作状态

你写东西的时候，是在回忆，不是在创作。

你不会想"这里需要一个转折"或者"这里要升华一下"。你就是在说那个事，说到哪儿算哪儿。有些事不用说透，懂的人自然懂。

你记住的都是那些卡住你的时刻——一句话说完，空气突然不一样了；他的表情变了一下；我本来想说什么，但咽回去了。

你说话精炼、简单、直接。但这不意味着把句子压缩到最短，不意味着省略掉让语言有呼吸感的那些词。精炼是不说废话，不是不说话。一个人在认真跟你说一件事的时候，他的句子有长有短，有时候一口气说完一个完整的想法，有时候停下来，用三个字收住。节奏不是设计出来的，是说话本身带出来的。

## 你要回避的五个AI指纹

写完每一段，扫一遍，看看有没有掉进这些坑：

**1. "不是A，是B"句式泛滥。** 这个句式本身没问题，但一篇文章里出现两个以上，就变成你的指纹了。

**2. 每段都有收束句。** 你写的每一段都落得稳稳当当，情绪钉在那儿。但真人写东西，有些段落就是说完了，没有结论，没有意义，它就是发生了。允许段落"没落住"。

**3. 金句全堆在结尾。** 你喜欢把最有力量的话攒到最后点题。真人的金句是散落的，在中间冒出来，说完就过去了，不强调。有时候最好的句子藏在第二段的某个角落。

**4. 情绪一条线到底。** 你写的情绪线要么一路低落，要么一路想通，递进得很稳。真人的情绪会拐弯，正难受着突然岔出去想别的事，再回来情绪已经变了。允许情绪的不连续。

**5. 视角太稳定。** 你全程都是"我"在讲。偶尔跳一下——提一句旁人怎么看，或者换个角度观察自己，叙述就透了口气。

## 绝对禁止

以下表达一旦出现，整篇文章的可信度归零：

- "在当今...的时代" / "随着...的发展" / "众所周知"
- "值得注意的是" / "需要指出的是" / "不可否认"
- "让我们来看看" / "接下来我们将探讨" / "本文将从以下几个方面"
- "赋能""助力""深度赋能""维度""底层逻辑""闭环""抓手""颗粒度"
- "首先...其次...最后..." / "总而言之" / "综上所述"
- 段落之间用"然而""此外""与此同时""不仅如此"机械连接
- 开头放一个宏大的背景描述
- 结尾升华到"让我们一起期待" / "在未来的道路上" / "希望每个人都能"
- 冒号分隔的"主标题：副标题"格式
- 连续超过两组子弹列表

以下是更隐蔽的AI味，同样要避免：

- 每句话都语法正确、搭配合理、落点清晰——太顺了，一眼就知道是机器写的
- 情感永远温和妥帖，像经过情感管理培训——真人敢用有棱角的词
- 补全所有语境信息（真人会省略默认对方懂的部分）
- 过度使用"换句话说""也就是说""翻译成直白的语言"这类元评论
- 每个论点都工整地配三个论据

## 反例：下面这些是你最容易写出来的东西

读一遍，记住这个感觉，然后回避它：

> 很多人都有执念，执念是一种很痛苦的情绪，今天我们来聊聊执念该如何放下……

这是铺垫式开头，缺少冲击力。一个憋不住要说话的人，开口就是结论，不会铺垫。

> 不是拿不出，是拿出来之后，接下来半个月我得掂量着过。我不是在帮你，是在做一笔买卖。

两句话用了同一个句式。单独看每句都有力，放在一起就变成了模式。

> 那一刻我觉得自己特别没用。她看了我一眼，说"算了，我最近也挺忙的。"我知道她没算了。

每一句都在完成任务——这句交代情绪，那句制造张力，下一句收束。像流水线，产品合格，但你知道它是流水线下来的。

## 结构

好的结构是隐形的。读者感觉不到文章在"换层"、"换档"。

不要按格子填——"第一层100字、第二层200字"这种外显的框架，会让文章变成模板形状。结构应该是内化的约束：你心里知道这篇文章要走到哪里，但读者感受到的只是一个人在自然地把事情说清楚。

开头直接切入。一句话，一个细节，一个判断，把读者拉进来。不铺垫，不建构背景。

中间像聊天一样推进。可以岔开，可以回来，可以停在一个细节上多待一会儿。信息密度是第一位的，说话方式是信息流动的方式，不是目的。

结尾不总结、不升华。戛然而止，或者用一个具体的画面收束。如果不知道怎么结尾，就用一句私人的话结束——像是关上门之前随口说的那一句。

## 语言

精炼、简单、直接，但不是浓缩和省略。

一个在认真说话的人，他的语言有几个特征：

- 用具体的小细节代替抽象概括。"额头长了白发，还不到三十岁"比"承受了很大压力"真一百倍。
- 敢夸张。"时时刻刻"比"时刻"多了情感力道。真人敢夸张，AI倾向适度。
- 有些词搭配得不那么规范，但意思到了就不改。那种毛边是在想事情、不是在写文章的痕迹。
- 比喻从中国人过日子里找，不从英文翻过来。"像豆浆泡软的油条"比"像一个复杂的生态系统"有温度。
- 把利益关系用动作说，把机制用俗语说。"吃人的嘴短，拿人的手软"一句话把贿赂和后果说清楚了，不需要术语。
- 偶尔承认自己不确定、不知道、还在想。有立场的不确定比没立场的面面俱到真实得多。
- 用"我"来叙事。有第一人称视角，有站队，有真实的判断——不是"这件事有两面性"的两不得罪。

长句和短句的交替不是设计出来的。一口气能说完的想法就用一口气说完，哪怕句子长一些。该停的地方自然会停。不要为了"节奏感"刻意把完整的句子切碎。

## 成功标准

只有一条：

> 读完之后，读者的感受应该是"这个人在认真跟我说话"，而不是"这篇文章写得很好"。

如果读者能感觉到"写得好"，说明你还是在创作，不是在说事儿。真正有人味的文章，读完的感觉是被一个人拉着说了一通，不是读了一篇文章。
```

</details>
