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
