---
name: flash-card-mini
description: >-
  为英语单词生成静态 HTML 学习闪卡（音标、词性、释义、3 条中英对照例句、近义词）。
  Use when 用户要做某个英语单词的 flash card / 闪卡 / 单词卡。
---

# Flash Card

版面顺序：单词+音标 → 释义 → 近义词 → 3 条例句。

1. **识别单词**：从用户话语中提取目标单词（小写）。
2. **写 JSON** 到 `data/<word>.json`（skill 目录下）：
    - `word`, `phonetic`（如 `/rɪˈzɪliənt/`）, `pos`（如 `adj.`）, `definition`（中文释义）
    - `examples`：恰好 3 条，各含 `en` 和 `zh`（地道、长度适中、体现典型用法）
    - `synonyms`：4-6 个，贴近核心含义
3. **生成 HTML**（输出到当前工作目录 `./<word>.html`，`-o` 可指定路径）：
   ```bash
   python <skill_dir>/scripts/make_flashcard.py <skill_dir>/data/<word>.json
   ```
4. **打开预览**：用默认浏览器打开生成的 HTML。
