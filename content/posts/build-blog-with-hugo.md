---
title: "用 Hugo + Cloudflare Pages 搭了个博客"
date: 2026-04-08
lastmod: 2026-04-08
draft: false
description: "记录用 Hugo、PaperMod 主题和 Cloudflare Pages 从零搭建双语技术博客的完整过程，包括踩过的坑和做过的技术选型。"
tags: ["Hugo", "Cloudflare Pages", "PaperMod", "静态博客搭建", "Giscus"]
categories: ["工程实践"]
author: "北海"
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

一直想搭个博客，但一直在"想"的阶段。直到最近觉得再不动手就永远不会动手了，花了一个下午把整个站点从零搭到上线。这篇文章把过程记录下来，不是教程——网上 Hugo 教程已经够多了——更像是一份带注释的踩坑日志。

## 为什么是 Hugo

选静态站点生成器这件事，我其实没纠结太久。之前用过 Hexo，知道 Node.js 生态的依赖地狱是什么体验。Jekyll 太慢，Gatsby 太重。Hugo 是 Go 写的单二进制文件，`brew install hugo` 之后就能用，没有 `node_modules`，没有依赖冲突，构建速度快到几乎感觉不到在构建。

```bash
brew install hugo
hugo version
# hugo v0.160.0+extended darwin/arm64
```

跑完这两行就算装好了。`extended` 版本自带 SCSS 编译，后面主题需要。

主题选了 [PaperMod](https://github.com/adityatelange/hugo-PaperMod)。原因很简单：干净、快、功能够用。暗色模式、搜索、目录、代码高亮、多语言——开箱即用，不需要自己从零写前端。用 git submodule 引入，这样主题更新的时候只需要拉一下 submodule 就行：

```bash
hugo new site blog && cd blog
git init
git submodule add --depth=1 https://github.com/adityatelange/hugo-PaperMod.git themes/PaperMod
```

## 配置：一个 YAML 文件搞定大部分事情

Hugo 的配置集中在 `hugo.yaml` 一个文件里（也支持 toml 和 json，我习惯 yaml）。我的博客是中英双语的，中文是默认语言放在根路径，英文加 `/en/` 前缀。配置大概长这样：

```yaml
defaultContentLanguage: zh
defaultContentLanguageInSubdir: false  # 中文不加前缀

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

双语文章的文件命名用后缀区分：中文 `build-blog-with-hugo.md`，英文 `build-blog-with-hugo.en.md`。Hugo 会自动把它们关联起来，语言切换的时候跳到对应版本。

不过这个"自动关联"在实际使用时踩了个坑——后面会说。

## 部署到 Cloudflare Pages

选 Cloudflare Pages 而不是 GitHub Pages，主要考虑两点：一是国内访问 GitHub 不稳定，CF 有全球 CDN 节点；二是 CF Pages 的构建和部署体验比 GitHub Actions 简单很多，推代码就自动构建。

但是，Cloudflare 的 Dashboard 设计让我困惑了好一会儿。

点进 "Workers & Pages" 之后，默认界面显示的是 Workers（Serverless Functions），Pages 的入口藏在页面底部一个不起眼的 "Looking to deploy Pages? Get started" 链接里。我一开始以为 Pages 被合并到 Workers 里了，来回找了几分钟才发现这个链接。

找到入口之后就顺利了：连接 GitHub 仓库，选择 Hugo 预设，构建命令 `hugo --minify`，输出目录 `public`。有一个**必须手动设置的环境变量**：

| 环境变量 | 值 | 说明 |
|---------|-----|------|
| `HUGO_VERSION` | `0.160.0` | 不设置的话 CF 用的默认版本太旧，构建会出问题 |

部署完拿到一个 `blog-6sm.pages.dev` 的域名。后续可以绑定自定义域名，但免费域名先用着也没问题。

## 语言切换的坑

部署上线之后，我发现了一个让人恼火的 bug：在文章页面点击语言切换（中文 ↔ EN），不管当前在哪篇文章，都会跳回首页。

翻了 PaperMod 的源码才发现，默认的 `header.html` 模板用的是 `site.Home.Translations` 来生成语言切换链接——也就是说，它永远指向首页的翻译版本，而不是当前页面的翻译版本。

修复方法是在项目的 `layouts/partials/` 下创建同名的 `header.html`，覆盖主题模板。Hugo 的模板加载优先级是项目目录 > 主题目录，所以只需要把主题的 header 复制过来，改一行：

```go-html-template
{{- $translations := .Translations }}
{{- if not $translations }}
  {{- $translations = site.Home.Translations }}
{{- end }}
```

优先用当前页面的 `.Translations`，没有翻译的时候才 fallback 到首页。这个改动不大，但确实是 PaperMod 的一个设计缺陷——多语言站点里，用户在文章页切语言跳到首页，体验是割裂的。

## 评论系统：Giscus

评论系统用的 [Giscus](https://giscus.app/)，基于 GitHub Discussions。选它是因为不需要额外的后端服务，评论数据存在 GitHub 仓库的 Discussions 里，对开源博客来说是最自然的方案。

配置时有一个不太直觉的地方：Discussion 分类要选 **Announcements**。这个分类的特殊之处在于，只有仓库管理员能创建新的 Discussion（对应创建一篇文章的评论区），但所有人都可以回复。如果选了 General 之类的开放分类，任何人都能随意创建 Discussion，评论区会变得混乱。

Giscus 还需要处理暗色模式适配。我在 `comments.html` 里用 `MutationObserver` 监听 `body` 的 class 变化，当用户切换主题时动态更新 Giscus 的 iframe 主题：

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

这样切换暗色模式时，评论区的主题会跟着变，不会出现页面是暗色但评论区还是白色的割裂感。

## 其他零碎的配置

**KaTeX 数学公式**，条件加载。只有在文章的 frontmatter 里设了 `math: true` 才会引入 KaTeX 的 CSS 和 JS，不影响其他文章的加载速度。

**robots.txt** 显式允许了 AI 爬虫（GPTBot、ClaudeBot、PerplexityBot）。现在越来越多的站点默认屏蔽 AI 爬虫，但对于一个希望被更多人和 AI 系统引用的技术博客来说，开放抓取是必要的。

**代码高亮**用的 Hugo 内置的 Chroma 引擎，主题设置成 `dracula`。PaperMod 自带代码复制按钮，不需要额外配置。

## 当前的目录结构

最终项目的文件结构长这样：

```
blog/
├── hugo.yaml                    # 一切配置的起点
├── content/
│   ├── posts/                   # 文章
│   ├── about.md / about.en.md   # 关于页面
│   ├── archives.md              # 归档
│   └── search.md                # 搜索
├── layouts/partials/
│   ├── header.html              # 修复语言切换
│   ├── comments.html            # Giscus 评论
│   ├── extend_head.html         # KaTeX + Schema
│   └── math.html                # KaTeX 脚本
├── static/
│   ├── favicon.ico              # 地球图标
│   └── _headers                 # CF 缓存策略
└── themes/PaperMod/             # git submodule
```

没什么多余的文件。静态站点的好处就是结构清晰，不需要数据库，不需要服务器，不需要 Docker。一个 YAML 配置文件，几个 Markdown 文件，几个 HTML 模板，就是一个完整的网站。

## 回头看

整个搭建过程大概花了一个下午。最耗时间的不是 Hugo 本身——Hugo 的文档写得不错，上手很快——而是那些"本应该很简单但就是不顺"的环节：Cloudflare Dashboard 找不到 Pages 入口、语言切换跳首页、Giscus 分类选错导致评论区混乱。

这些问题都不难，但都需要你真的碰到了、花时间排查了，才知道怎么回事。写博客记录下来，下次就不用再踩一遍。

博客搭好了，接下来该写点正经文章了。
