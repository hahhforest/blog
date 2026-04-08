---
title: "Building a Blog with Hugo + Cloudflare Pages"
date: 2026-04-08
lastmod: 2026-04-08
draft: false
description: "A walkthrough of building a bilingual tech blog from scratch with Hugo, PaperMod theme, and Cloudflare Pages — including the pitfalls I hit along the way."
tags: ["Hugo", "Cloudflare Pages", "PaperMod", "Static Site", "Giscus"]
categories: ["Engineering"]
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

I'd been telling myself I'd start a blog for months. Then one afternoon I decided to just do it — no more planning, no more comparing frameworks. A few hours later the site was live. This post is a record of that process: not a tutorial, more of an annotated changelog of mistakes and decisions.

## Why Hugo

I didn't spend long choosing a static site generator. I'd used Hexo before and knew what Node.js dependency hell feels like. Jekyll is slow. Gatsby is overkill for a blog. Hugo is a single Go binary — `brew install hugo` and you're done. No `node_modules`, no dependency conflicts, and builds so fast you barely notice them happening.

```bash
brew install hugo
hugo version
# hugo v0.160.0+extended darwin/arm64
```

That's the entire install. The `extended` version includes built-in SCSS compilation, which the theme needs.

For the theme, I went with [PaperMod](https://github.com/adityatelange/hugo-PaperMod). It's clean, fast, and ships with everything I need out of the box: dark mode, search, table of contents, syntax highlighting, multilingual support. Added it as a git submodule so updates are just a `git pull`:

```bash
hugo new site blog && cd blog
git init
git submodule add --depth=1 https://github.com/adityatelange/hugo-PaperMod.git themes/PaperMod
```

## Configuration: One YAML File Does Most of the Work

Hugo keeps everything in a single `hugo.yaml` (also supports toml and json — I prefer yaml). My blog is bilingual: Chinese as the default language at the root path, English under `/en/`. The language config looks like this:

```yaml
defaultContentLanguage: zh
defaultContentLanguageInSubdir: false  # Chinese gets no prefix

languages:
  zh:
    languageName: "中文"
    weight: 1
    title: "Forest's Blog"
    params:
      author: "北海"
  en:
    languageName: "EN"
    weight: 2
    title: "Forest's Blog"
    params:
      author: "Chunhao Zhang"
```

Bilingual articles use filename suffixes: Chinese is `build-blog-with-hugo.md`, English is `build-blog-with-hugo.en.md`. Hugo automatically links them together for language switching.

Though that "automatic linking" had a catch — more on that below.

## Deploying to Cloudflare Pages

I chose Cloudflare Pages over GitHub Pages for two reasons: GitHub is unreliable from mainland China, and CF Pages has global CDN nodes with a simpler deploy workflow — just push code and it builds automatically.

But Cloudflare's Dashboard design confused me for a bit.

After clicking into "Workers & Pages", the default view shows Workers (Serverless Functions). The Pages entry is buried at the bottom of the page as an unassuming "Looking to deploy Pages? Get started" link. I spent a few minutes thinking Pages had been merged into Workers before I spotted it.

Once past that, setup was straightforward: connect GitHub repo, select Hugo preset, build command `hugo --minify`, output directory `public`. There's one **critical environment variable** you must set manually:

| Variable | Value | Why |
|----------|-------|-----|
| `HUGO_VERSION` | `0.160.0` | CF's default Hugo version is outdated and will break the build |

After deployment I got a `blog-6sm.pages.dev` domain. Custom domains can be added later, but the free subdomain works fine for now.

## The Language Switcher Bug

After going live, I noticed an annoying bug: clicking the language toggle (中文 ↔ EN) on any article page would redirect to the homepage instead of the translated version of that article.

Digging into PaperMod's source, I found that the default `header.html` template uses `site.Home.Translations` for the language switcher — meaning it always points to the homepage translation, not the current page's translation.

The fix was to create an override `header.html` in the project's `layouts/partials/` directory. Hugo's template loading priority is project > theme, so you just copy the theme's header and change one line:

```go-html-template
{{- $translations := .Translations }}
{{- if not $translations }}
  {{- $translations = site.Home.Translations }}
{{- end }}
```

Prefer the current page's `.Translations`, fall back to homepage only when there's no translation available. Small change, but this is genuinely a design flaw in PaperMod — on a multilingual site, switching languages on an article page and landing on the homepage feels broken.

## Comments with Giscus

For comments I went with [Giscus](https://giscus.app/), which is powered by GitHub Discussions. No backend needed — comments live in the repo's Discussions, which is the most natural setup for an open-source blog.

One non-obvious config choice: the Discussion category should be **Announcements**. This category restricts who can create new Discussions (only repo admins — corresponding to initializing a comment thread for an article), while still letting anyone reply. If you pick an open category like General, anyone can create arbitrary Discussions and things get messy.

Giscus also needs dark mode adaptation. I used a `MutationObserver` in `comments.html` to watch for class changes on `body` and update the Giscus iframe theme dynamically:

```javascript
const observer = new MutationObserver(() => {
    const isDark = document.body.classList.contains('dark');
    const theme = isDark ? 'noborder_dark' : 'noborder_light';
    const iframe = document.querySelector('iframe.giscus-frame');
    if (iframe) {
        iframe.contentWindow.postMessage(
            { giscus: { setConfig: { theme } } },
            'https://giscus.app'
        );
    }
});
observer.observe(document.body, { attributes: true, attributeFilter: ['class'] });
```

This way, toggling dark mode updates the comment section in real time — no jarring white comment box on a dark page.

## Miscellaneous Setup

**KaTeX math rendering** is conditionally loaded. Only articles with `math: true` in frontmatter pull in the KaTeX CSS and JS, so other pages aren't slowed down.

**robots.txt** explicitly allows AI crawlers (GPTBot, ClaudeBot, PerplexityBot). More and more sites are blocking AI bots by default, but for a technical blog that wants to be cited by both humans and AI systems, open crawling makes sense.

**Syntax highlighting** uses Hugo's built-in Chroma engine with the `dracula` theme. PaperMod includes a code copy button out of the box.

## The Final Structure

Here's what the project looks like:

```
blog/
├── hugo.yaml                    # Where all config lives
├── content/
│   ├── posts/                   # Articles
│   ├── about.md / about.en.md   # About page
│   ├── archives.md              # Archive
│   └── search.md                # Search
├── layouts/partials/
│   ├── header.html              # Language switcher fix
│   ├── comments.html            # Giscus comments
│   ├── extend_head.html         # KaTeX + Schema
│   └── math.html                # KaTeX scripts
├── static/
│   ├── favicon.ico              # Globe icon
│   └── _headers                 # CF caching policy
└── themes/PaperMod/             # git submodule
```

No bloat. The beauty of static sites is that the structure is transparent — no database, no server, no Docker. A YAML config, some Markdown files, a few HTML partials, and you have a complete website.

## Looking Back

The whole process took an afternoon. The most time-consuming parts weren't Hugo itself — Hugo's documentation is solid and the learning curve is gentle — but rather the "should be simple but somehow isn't" stuff: finding the Pages entry in Cloudflare's Dashboard, the language switcher jumping to homepage, picking the wrong Giscus category.

None of these were hard problems. But they all required actually hitting the wall, spending time debugging, and figuring out the fix. Writing it down here so I don't step on the same mines twice.

Blog's up. Time to write some real content.
