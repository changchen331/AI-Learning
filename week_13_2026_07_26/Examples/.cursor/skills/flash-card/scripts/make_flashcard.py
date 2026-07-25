"""
英语单词 Flash Card 生成器
=========================
为一个英语单词生成一张静态 HTML 学习卡片，包含：
  - 单词、音标、词性、释义
  - 固定 3 条中英对照例句
  - 近义词标签（位于例句上方）

用法:
    python make_flashcard.py <data.json>                  # 输出到当前目录 <word>.html
    python make_flashcard.py <data.json> -o output.html   # 指定输出路径

JSON 数据格式:
{
  "word": "resilient",
  "phonetic": "/rɪˈzɪliənt/",
  "pos": "adj.",
  "definition": "能迅速从困难中恢复过来的；有韧性的",
  "examples": [
    {"en": "...", "zh": "..."},
    {"en": "...", "zh": "..."},
    {"en": "...", "zh": "..."}
  ],
  "synonyms": ["tough", "flexible", "strong"]
}
"""
import argparse
import json
import html
from pathlib import Path


TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{word} - Flash Card</title>
<style>
  :root {{
    --bg: #f5f7fb;
    --card: #ffffff;
    --ink: #1f2937;
    --muted: #6b7280;
    --accent: #4f46e5;
    --accent-soft: #eef2ff;
    --border: #e5e7eb;
    --shadow: 0 10px 30px rgba(17, 24, 39, 0.08);
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
                 "Microsoft YaHei", Roboto, sans-serif;
    background: var(--bg);
    color: var(--ink);
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 24px;
  }}
  .card {{
    width: 100%;
    max-width: 720px;
    background: var(--card);
    border-radius: 20px;
    box-shadow: var(--shadow);
    overflow: hidden;
  }}
  .header {{
    padding: 32px 36px 24px;
    background: linear-gradient(135deg, var(--accent) 0%, #7c3aed 100%);
    color: #fff;
  }}
  .word {{
    margin: 0;
    font-size: 44px;
    font-weight: 700;
    letter-spacing: -0.5px;
  }}
  .phonetic {{
    margin-top: 8px;
    font-size: 18px;
    opacity: 0.92;
    font-style: italic;
  }}
  .body {{ padding: 28px 36px 36px; }}
  .definition {{
    font-size: 20px;
    line-height: 1.6;
    padding: 14px 16px;
    background: var(--accent-soft);
    border-left: 4px solid var(--accent);
    border-radius: 8px;
  }}
  .definition .pos {{
    color: var(--accent);
    font-weight: 600;
    margin-right: 6px;
  }}
  h2 {{
    margin: 28px 0 14px;
    font-size: 16px;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
  }}
  .synonyms {{ display: flex; flex-wrap: wrap; gap: 10px; }}
  .synonyms .tag {{
    padding: 6px 14px;
    background: var(--accent-soft);
    color: var(--accent);
    border-radius: 999px;
    font-size: 14px;
    font-weight: 500;
  }}
  .examples {{ list-style: none; padding: 0; margin: 0; }}
  .examples li {{
    padding: 14px 16px;
    margin-bottom: 10px;
    background: #fafafa;
    border: 1px solid var(--border);
    border-radius: 10px;
  }}
  .examples .en {{
    font-size: 17px;
    line-height: 1.55;
  }}
  .examples .zh {{
    margin-top: 6px;
    font-size: 14px;
    color: var(--muted);
    line-height: 1.55;
  }}
  .footer {{
    margin-top: 28px;
    padding-top: 16px;
    border-top: 1px dashed var(--border);
    font-size: 12px;
    color: var(--muted);
    text-align: center;
  }}
</style>
</head>
<body>
  <div class="card">
    <div class="header">
      <h1 class="word">{word}</h1>
      <div class="phonetic">{phonetic}</div>
    </div>
    <div class="body">
      <div class="definition">
        <span class="pos">{pos}</span>{definition}
      </div>

      <h2>近义词</h2>
      <div class="synonyms">
        {synonyms_html}
      </div>

      <h2>例句</h2>
      <ul class="examples">
        {examples_html}
      </ul>

      <div class="footer">Flash Card · 学一个词，记一组词</div>
    </div>
  </div>
</body>
</html>
"""


def render_synonyms(synonyms):
    return "\n        ".join(
        f'<span class="tag">{html.escape(s)}</span>' for s in synonyms
    )


def render_examples(examples):
    # 固定 3 条例句：不足补空，多余截断
    fixed = list(examples[:3]) + [{}] * (3 - len(examples))
    items = []
    for ex in fixed:
        en = html.escape(ex.get("en", "") or "（待补充例句）")
        zh = html.escape(ex.get("zh", "") or "（待补充翻译）")
        items.append(
            f'<li><div class="en">{en}</div>'
            f'<div class="zh">{zh}</div></li>'
        )
    return "\n        ".join(items)


def build_html(data):
    return TEMPLATE.format(
        word=html.escape(data["word"]),
        phonetic=html.escape(data.get("phonetic", "")),
        pos=html.escape(data.get("pos", "")),
        definition=html.escape(data.get("definition", "")),
        examples_html=render_examples(data.get("examples", [])),
        synonyms_html=render_synonyms(data.get("synonyms", [])),
    )


def main():
    parser = argparse.ArgumentParser(description="生成英语单词 Flash Card HTML")
    parser.add_argument("data", help="JSON 数据文件路径")
    parser.add_argument("-o", "--output",
                        help="输出 HTML 路径（默认当前目录下 <word>.html）")
    args = parser.parse_args()

    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_path = Path(args.output) if args.output else Path.cwd() / f"{data['word']}.html"
    out_path.write_text(build_html(data), encoding="utf-8")
    print(f"已生成: {out_path}")


if __name__ == "__main__":
    main()
