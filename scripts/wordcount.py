#!/usr/bin/env python3
"""
博客文章字数统计与阅读时间估算。

中文字符按字计数，英文按空格分词计数。
阅读速度：中文 300 字/分钟，英文 200 词/分钟（技术内容偏保守）。

用法：
    python3 wordcount.py <markdown文件路径>
    python3 wordcount.py content/posts/     # 统计目录下所有 .md
"""

import re
import sys
from pathlib import Path


def strip_frontmatter(text: str) -> str:
    """去掉 YAML frontmatter（--- 包裹的部分）。"""
    m = re.match(r'^---\s*\n.*?\n---\s*\n', text, re.DOTALL)
    return text[m.end():] if m else text


def strip_markdown(text: str) -> str:
    """去掉 Markdown 语法标记，保留纯文本。"""
    # 代码块
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`[^`]+`', '', text)
    # 图片、链接
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[([^\]]+)\]\(.*?\)', r'\1', text)
    # HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    # Markdown 标记（标题、粗体、斜体、引用等）
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[-*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
    # 脚注
    text = re.sub(r'\[\^\d+\]', '', text)
    return text


def count_words(text: str) -> dict:
    """
    统计字数。

    返回:
        cn_chars: 中文字符数
        en_words: 英文单词数
        total: 等效总字数（中文字符 + 英文单词）
        reading_min: 预计阅读时间（分钟）
    """
    # 中文字符（含中文标点）
    cn_chars = len(re.findall(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', text))

    # 英文单词：去掉中文后按空格分词
    en_text = re.sub(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', ' ', text)
    en_words = len([w for w in en_text.split() if re.search(r'[a-zA-Z]', w)])

    total = cn_chars + en_words

    # 阅读时间：中文 300 字/分钟，英文 200 词/分钟
    reading_min = cn_chars / 300 + en_words / 200
    reading_min = max(1, round(reading_min))

    return {
        'cn_chars': cn_chars,
        'en_words': en_words,
        'total': total,
        'reading_min': reading_min,
    }


def process_file(filepath: Path) -> dict:
    """处理单个 Markdown 文件。"""
    text = filepath.read_text(encoding='utf-8')
    text = strip_frontmatter(text)
    text = strip_markdown(text)
    return count_words(text)


def main():
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <文件或目录>")
        sys.exit(1)

    target = Path(sys.argv[1])

    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = sorted(target.rglob('*.md'))
        # 排除 _index 文件
        files = [f for f in files if not f.name.startswith('_index')]
    else:
        print(f"错误: {target} 不存在")
        sys.exit(1)

    print(f"{'文件':<50} {'中文':>6} {'英文':>6} {'总字数':>7} {'阅读时间':>8}")
    print("-" * 82)

    for f in files:
        stats = process_file(f)
        name = str(f.relative_to(target.parent) if target.is_dir() else f.name)
        if len(name) > 48:
            name = '...' + name[-45:]
        print(f"{name:<50} {stats['cn_chars']:>6} {stats['en_words']:>6} "
              f"{stats['total']:>7} {stats['reading_min']:>6} 分钟")


if __name__ == '__main__':
    main()
