# ARCHITECTURE.md — 原生 Skills ReAct Agent

> 教学主题： **Agent 使用 Skills 的能力** —— 渐进式加载（Progressive Skill Loading）+ 原生 ReAct 循环。
> 用单个 `flash-card` skill 走通「预加载 meta → LLM 主动 read_skill 取说明 → Thought/Action/Observation → 产出」全链路。

---

## 1. 项目定位

| 维度       | 说明                                                                                                                                                                                         |
|------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 教学目标   | 让学生看清「Agent 是怎么用 skill 的」：系统提示**始终只放 meta**（名字+描述），完整 SKILL.md 由 LLM 在 ReAct 循环里**主动发 `read_skill` 取回** —— 读不读、读哪个、何时读，都由 LLM 自己判断 |
| 核心机制   | ① 渐进式加载（meta 预加载 + LLM 驱动 read_skill） ② ReAct 循环（stop token 控制轮流） ③ Action 解析与双形态执行 ④ 「未读先调」拦截                                                           |
| 模型       | DeepSeek `deepseek-v4-flash`（即 `deepseek-chat`），OpenAI 兼容接口                                                                                                                          |
| 样本 skill | `flash-card`（复制自课程 `week13 skills和harness/skills/flash-card`），未做任何改动                                                                                                          |
| 交互形式   | CLI，逐步打印 Phase1（meta）/ Phase2（read_skill 取回全文）/ 每步 ReAct，最后自动用默认浏览器打开产物                                                                                        |

**方案对比**：

- vs **朴素做法**（把所有 skill 全文一次性塞进系统提示）：渐进式在 skill 数量增长时 token 是 O (被读到) 而非 O (全部)。
- vs **agent 自动注入**（Phase 1 选完由 agent 把选中 skill 全文塞进系统提示）：本设计把「何时读」交给 LLM 的
  Action，read_skill 是 trace 里可见的一步，真正展示「Agent 主动用 skill 的能力」。
- vs smart_cockpit_agent 的完整 SSE Web UI：本项目刻意只做 CLI trace，把注意力压到「加载机制」本身。

---

## 2. 整体流水线

```
用户请求
   │
   ▼
┌──────────────────────────────────────────────────────────┐
│ Phase 1 · 预加载 meta                                      │
│   系统提示 = registry.meta_summary()  （只含 name+desc）    │
│   完整 SKILL.md 一字未进上下文，留给 LLM 自己 read_skill     │
└──────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────┐
│ ReAct 循环（react_agent.run，max_steps=8）                 │
│   LLM 输出 → 解析                                          │
│   ├─ 含 "Final Answer:" → 结束，自动打开 HTML              │
│   └─ 含 "Action:" → parse_action 拆出 skill 名             │
│        ├─ read_skill(name=...) → load_full 返回完整 SKILL.md │
│        │   记入 _read 集合；这是 Phase 2，LLM 主动取回       │
│        ├─ 业务 skill 未读过？→ 拦截，退回让 LLM 先 read_skill│
│        └─ 业务 skill 已读？→ skill_executor.execute(action) │
│             ├─ 脚本有 run() → run(**params)               │
│             └─ 否则 CLI 形态 → params 落盘 JSON + 子进程跑脚本│
│   Observation 回喂 messages，继续下一轮                     │
└──────────────────────────────────────────────────────────┘
```

每步对应脚本：

| 步骤                                      | 脚本                                          | 关键函数                                                      |
|-------------------------------------------|-----------------------------------------------|---------------------------------------------------------------|
| 扫描 skill / 解析 frontmatter（含块标量） | `src/skill_registry.py`                       | `SkillRegistry._scan` / `_parse_frontmatter` / `meta_summary` |
| read_skill 取完整说明                     | `src/skill_registry.py`                       | `load_full`                                                   |
| Phase1 预加载 + ReAct + read_skill 拦截   | `src/react_agent.py`                          | `SkillsReActAgent.run`                                        |
| Action 解析 + 执行                        | `src/skill_executor.py`                       | `parse_action` / `SkillExecutor.execute`                      |
| 产物渲染                                  | `skills/flash-card/scripts/make_flashcard.py` | （skill 自带，未改）                                          |

---

## 3. 各环节技术选型

### 3.1 渐进式加载为什么是 LLM 驱动的 read_skill

朴素做法：启动时把所有 skill 的完整 SKILL.md 拼进系统提示。问题：skill 一多，token 爆炸，且大量 skill 本次根本用不到。

「agent 自动注入」做法：Phase 1 让 LLM 选 skill，agent 再把选中 skill 的全文塞进系统提示。问题：读取时机和范围被 agent
包办，trace 里看不到「LLM 主动去读」这一步，教学上把最关键的主动权藏掉了。

本设计（LLM 驱动）：

- **Phase 1 预加载 meta**：`_scan` 用 `_parse_frontmatter` 抓 frontmatter（含 `>-` 块标量多行 description）的 `name`/
  `description`，拼成 `meta_summary`（本项目 1 个 skill ~50 tokens）塞进系统提示。完整 SKILL.md 此刻 **不在**上下文。
- **Phase 2 = LLM 主动 `read_skill`**：系统提示里告诉 LLM「调用业务 skill 前必须先 read_skill」。LLM 在 ReAct 里发出
  `read_skill(name="flash-card")` → agent 调 `load_full` 把完整 SKILL.md 作为 Observation 返回。读不读、读哪个、何时读，全是
  LLM 自己的 Action 决策。
- **「未读先调」拦截**：`run` 里维护 `_read` 集合，业务 skill 若没 read_skill 过，agent 退回一条 Observation 让 LLM 先读 ——
  保证 read_skill 这步必然发生、必然可见。

> 教学量化：本项目 meta ~50 tokens vs 完整 SKILL.md ~600 tokens。skill 涨到 20 个时，朴素做法系统提示 ~12000 tokens，本设计
> Phase 1 仍 ~1000 tokens，且只有 LLM 实际 read 的那几个 skill 的全文才会进上下文。

### 3.2 ReAct 的 stop token 控制

```python
resp = client.chat.completions.create(model=MODEL, messages=messages,
                                      stop=["Observation:"])
```

模型输出 `Thought` + `Action: flash-card(data={...})` 后，会自然想继续写 `Observation:` —— `stop=["Observation:"]`
让它在那时截断，把控制权交回 agent 去真实执行工具。执行完把 `Observation: <结果>` 作为 user 消息追加，下一轮 LLM
据此继续。这是「模型推理」与「真实工具执行」轮流的关键。

### 3.3 Action 解析与双形态执行

Action 形如 `flash-card(data={"word":"crazy",...})`，`parse_action` 用正则拆出 skill 名与参数串，再用 `ast.literal_eval`
解析参数（支持单/双引号、字典、列表字面量，比 `json.loads` 容错）。

执行器支持两种 skill 脚本形态（`skill_executor.execute`）：

| 形态                     | 判定             | 执行方式                                                                   | 代表                         |
|--------------------------|------------------|----------------------------------------------------------------------------|------------------------------|
| `run(**kwargs)` 标准入口 | 脚本定义了 `run` | `mod.run(**params)`                                                        | CLAUDE.md §12.2 规范的 skill |
| CLI 脚本                 | 无 `run`         | params 写成 JSON 丢进 `data/`，`python <script> <json>` 在 `outputs/` 下跑 | 本项目 `make_flashcard.py`   |

> flash-card 的 `make_flashcard.py` 是 argparse CLI 脚本（无 `run`），走 CLI 形态：executor 把 LLM 产出的 `data={...}` 写成
> `skills/flash-card/data/<word>.json`，再以 `cwd=outputs/` 跑脚本，脚本默认输出 `./<word>.html` 自然落到 `outputs/`。

---

## 4. 实测结果（真实跑通）

请求 `给我做一张 crazy 词的闪卡`：

| 阶段         | 实际发生                                                                                      |
|--------------|-----------------------------------------------------------------------------------------------|
| Phase 1      | 系统提示只含 `- flash-card: 为一个英语单词生成静态 HTML 学习闪卡…`，完整 SKILL.md 未进上下文  |
| ReAct Step 1 | LLM **主动** `read_skill(name="flash-card")` 取回完整 SKILL.md（~600 tokens）作为 Observation |
| ReAct Step 2 | LLM 按读到的流程，**自己产出** crazy 的完整学习数据，调用 `flash-card(data={...})`            |
| 执行         | executor 落盘 `data/crazy.json` + 跑 `make_flashcard.py` → `outputs/crazy.html`               |
| ReAct Step 3 | `Final Answer` 返回 HTML 路径，agent 自动用默认浏览器打开                                     |

产物 `skills/flash-card/data/crazy.json` 实测含真实内容（非占位）：

```json
{
  "word": "crazy",
  "phonetic": "/ˈkreɪzi/",
  "pos": "adj.",
  "definition": "疯狂的，发疯的；狂热的，着迷的",
  "examples": [
    {
      "en": "He's crazy about playing basketball.",
      "zh": "他对打篮球着迷。"
    },
    {
      "en": "That idea sounds crazy, but it just might work.",
      "zh": "那个想法听起来很疯狂，但也许真的管用。"
    },
    {
      "en": "The crowd went crazy when the band came on stage.",
      "zh": "乐队上台时，人群沸腾了。"
    }
  ],
  "synonyms": [
    "mad",
    "insane",
    "wild",
    "foolish",
    "absurd",
    "irrational"
  ]
}
```

`outputs/crazy.html` 正确渲染上述全部字段（单词/音标/释义/近义词标签/3 条中英例句）。

---

## 5. 关键工程决策与踩坑

| 问题                                                                        | 根因                                                                                    | 解法                                                                                                                                                     |
|-----------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| 想展示「LLM 主动读 skill」但原设计把全文自动注入系统提示                    | Phase 2 由 agent 包办 load + 注入，LLM 没有读取动作                                     | 改为内置 `read_skill(name=...)` 动作：系统提示只放 meta，完整 SKILL.md 由 LLM 在 ReAct 里主动 read_skill 取回；并加「未读先调」拦截保证这步必然发生      |
| meta 显示成 `- flash-card: >-`（描述丢失）                                  | frontmatter 用了 YAML 块标量 `description: >-`，旧正则只抓到行尾 `>-`，没收后续缩进行   | `_parse_frontmatter` 识别 `>`/`-`/`                                                                                                                      |` 块标量，收集后续缩进行拼成完整描述 |
| `flash-card(data={...})` 跨多行时解析失败，白费一轮重试                     | `action_text = ...splitlines()[0]` 只取首行，把多行字典字面量截断成 `flash-card(data={` | 去掉 `splitlines()[0]`，取 Action 后的完整文本；`ast.literal_eval` 本就支持多行字面量                                                                    |
| LLM 用 `read_skill("flash-card")` 位置参数取不到名字，白费多轮              | `parse_action` 只解析关键字参数，位置参数落到 `_raw` 被忽略                             | `parse_action` 扩展为同时收集位置参数到 `_args` 列表；read_skill 处理处兼容 `name=` 与位置两种写法                                                       |
| 闪卡渲染成空壳、Final Answer 幻觉内容                                       | LLM 只传 `word="crazy"` 没给学习数据，脚本按占位渲染                                    | 提示词显式要求「skill 要你产出数据时必须把完整数据作为参数传入，脚本不补全」；executor 把 `data=` 参数解包成真实数据再落盘                               |
| skill 名含连字符 `flash-card`                                               | 朴素 Action 正则 `\w+` 不匹配 `-`                                                       | Action 名正则改为 `[a-zA-Z][\w-]*`，允许连字符                                                                                                           |
| LLM 尝试调用不存在的 `open-file` skill                                      | SKILL.md「打开预览」让 LLM 以为有对应 skill                                             | 不增 skill：agent 在 Final Answer 命中产物路径时自动 `os.startfile` 打开；LLM 遇错误后自行回退到 Final Answer，本身是 ReAct 错误恢复的教学点             |
| `load_full` 多次读盘                                                        | 每次调用都 `read_text`                                                                  | `_full_cache` 字典缓存，命中走内存                                                                                                                       |
| 子进程崩溃时 Observation 灌出整段 Python traceback（如 `KeyError: 'word'`） | CLI 分支原样返回 `stderr`，含 traceback + 内部脚本路径/行号，对 LLM 是噪音且泄漏实现    | `proc.returncode != 0` 时只取 stderr 最后一行（真正的异常行）回吐 `{"error": "脚本执行失败：<异常行>"}`。**治标**：万一 LLM 误调，错误至少干净可据以修参 |

---

## 6. 目录结构

```
skills_react_agent/
├── src/
│   ├── skill_registry.py     # 渐进式加载：_scan 只读 frontmatter；load_full 按需读全文
│   ├── skill_executor.py     # Action 解析 + 双形态执行（run() / CLI 脚本）
│   └── react_agent.py        # Phase1 选 skill → Phase2 注入 → ReAct 循环（入口）
├── skills/
│   └── flash-card/           # 样本 skill（复制自课程材料，未改动）
│       ├── SKILL.md          # frontmatter(name/desc) + 完整流程说明
│       ├── data/<word>.json  # 运行时 LLM 产出的单词数据落这里
│       └── scripts/make_flashcard.py   # argparse CLI：JSON → HTML
├── outputs/                  # 生成的 .html 闪卡落这里
├── requirements.txt          # 仅 openai
└── ARCHITECTURE.md
```

---

## 7. 运行方式

```bash
# 1. 装依赖
pip install -r requirements.txt
# 2. 配 DeepSeek key
set DEEPSEEK_API_KEY=sk-xxxx        # Windows / $env:DEEPSEEK_API_KEY="sk-xxxx" (PS)
# 3. 跑
python src/react_agent.py "给我做一张 crazy 词的闪卡"
# 或不传参，走默认 crazy
python src/react_agent.py
```

控制台会依次打印 Phase 1（meta + LLM 选择）、Phase 2（注入的完整 SKILL.md）、每步 ReAct（LLM 输出 + Observation）、Final
Answer，并在结束时用默认浏览器打开 `outputs/crazy.html`。

---

## 8. 教学延伸建议

- **加第二个 skill**：再放一个带 `run(**kwargs)` 的 skill（如 `word_quiz`）进来，让学生对比 CLI 形态与 run 形态两条执行路径，以及
  LLM 在多 skill 下主动 read 哪个、跳过哪个。
- **量化 token 节省**：在 Phase 1 与每次 read_skill 后各打印一次系统提示 + 已读内容的 token 数（`tiktoken` 或 DeepSeek
  usage），把「只读 meta vs 读完 SKILL.md」的数字差呈现在课堂。
- **read_skill 缓存复用**：让学生在多轮对话里复用同一 agent 实例，观察第二次请求同一 skill 时 `_full_cache` 命中、`_read`
  集合仍在 —— LLM 可直接调用而无需重读，展示「记忆」与「按需重读」的取舍。
