# USAGE_GUIDE — 运行与调用指南

## 一、环境准备

### 1.1 依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 内容：

- `openai>=1.0.0`（LLM 调用，用 OpenAI 兼容接口调 DeepSeek）
- `fastapi>=0.100` + `uvicorn>=0.20`（Web Demo 服务端）

### 1.2 API Key

默认用 DeepSeek（`deepseek-chat`，约 0.001 元/次）。设置环境变量：

```bash
# Linux / Mac
export DEEPSEEK_API_KEY="sk-xxxxxx"

# Windows PowerShell
$env:DEEPSEEK_API_KEY = "sk-xxxxxx"

# Windows cmd
set DEEPSEEK_API_KEY=sk-xxxxxx
```

### 1.3 工作目录

所有命令默认在项目根目录 `self_evolving_agent/` 下执行。

---

## 二、三种运行方式

### 方式 A：Web 步进式 Demo（推荐，课堂演示）

```bash
uvicorn serve:app --host 0.0.0.0 --port 8000 --reload
```

`--reload` 模式下改代码后服务会自动重启，不必手动停。

打开浏览器 `http://localhost:8000`，按钮驱动流程：

1. **点击「基线评估（60 题）」**
    - 流式显示每题对错（绿/红圆点 + 问题文字）
    - 完成后显示分类准确率（左下栏 + 评估卡片尾部）
    - 典型基线准确率 ~22%（初始只有 2 个 Skill，多数类别 Agent 答"需要联系人工客服"）

2. **点击「▶ 第 N/8 块：xxx」**
    - 该块 10 题逐一回答， **失败样本会累积到 `block_failed_turns`**
    - 如果 10/10 全对：显示绿色 ✓ 卡片「本块全对 — 跳过本轮进化」，Skill 库不变
    - 如果有失败：紫色 Nudge 卡片，Reviewer 分析 → create/patch 操作 → Probe eval

3. 重复步骤 2 完成 8 块

4. **点击「▶ 最终评估（60 题）」** → 60 题评估 + 基线对比表

任何时候：

- 点击 Nudge 卡片标题可折叠（讲解完收起来）
- 点击任意 Q&A 条目可展开完整问答
- 点击右侧 Skill 查看当前内容和版本历史（点击版本号看 diff）
- 点击右上角「↺ 重置初始态」彻底还原

**典型一次完整运行时长：8-15 分钟（取决于 LLM 响应速度和跳过块数）**

### 方式 B：纯命令行全流程（无 UI）

```bash
python src/demo_runner.py
```

控制台输出：基线 → 8 块 × (10 题 + Nudge 或跳过) → 最终。结果落盘至 `outputs/evolution_log.json`。

### 方式 C：独立规则评估（验证已有 Skill 库能力）

```bash
python src/rule_eval_with_review.py
```

用当前 `skills/` 目录的 Skill 跑一次全量 60 题评估， **不触发 Nudge、不修改 Skill**。 控制台打印准确率，所有答案保存到
`outputs/rule_eval_full.json` 便于手工复核。

---

## 三、核心合约说明（理解系统的关键）

本项目有一个 **显式契约**，理解它才能正确解读准确率数字：

**Agent 的行为合约**（`src/agent.py` 的系统提示）：

- 能从 Skill 答出 → 给具体完整答案， **不加推脱尾巴**
- 不能从 Skill 答出 → **仅回答一句**"需要联系人工客服"

**Evaluator 的判定规则**（`src/evaluator.py`）：

- 答案含 `联系人工` → 判定"Agent 推脱"（硬性失败）
- 否则检查 required 关键词全部出现、forbidden 关键词未被命中

因此三种失败原因互斥清晰：

- **Agent 推脱**：Skill 库没覆盖这类问题
- **缺少关键词**：Skill 有但信息不完整
- **出现禁止词**：Skill 有但 Agent 错误引用

---

## 四、脚本模块化调用

### 4.1 作为模块使用各组件

```python
import sys
sys.path.insert(0, "self_evolving_agent/src")

from skill_manager import SkillManager
from evaluator import Evaluator
from agent import CustomerServiceAgent
from background_reviewer import BackgroundReviewer

sm = SkillManager("self_evolving_agent/skills")
ev = Evaluator("self_evolving_agent/data/eval_set.json")
agent = CustomerServiceAgent(sm, nudge_interval=0)
reviewer = BackgroundReviewer("self_evolving_agent/data/policies.md", sm)

# 单题应答
answer = agent.answer("我是银卡VIP，45天了可以退货吗？")
ok, reason = ev.evaluate_answer(answer, question_id=11)
print(f"对错: {ok} | 原因: {reason}")

# 收集一批失败，手动触发 Reviewer
failed_turns = [
    {"question": "我买了游戏点卡想退款", "answer": "需要联系人工客服",
     "fail_reason": "Agent 推脱（含 '联系人工'）"},
    # ... 更多失败样本
]
actions = reviewer.review(failed_turns)
for act in actions:
    if act["action"] == "create":
        sm.create(act["skill_name"], act["content"], reason=act["reason"])
    elif act["action"] == "patch":
        sm.patch(act["skill_name"], act["old_text"], act["new_text"],
                 reason=act["reason"])
```

### 4.2 查询 Skill 版本历史

```python
sm = SkillManager("self_evolving_agent/skills")

# 所有 Skill 的版本摘要
for name, versions in sm.get_all_version_summaries().items():
    print(f"{name}: {len(versions)} 个版本")

# 指定 Skill 的完整历史
refund_history = sm.get_version_history("refund")
for v in refund_history:
    print(f"v{v['version']} [{v['action']}] {v['reason']}")
```

### 4.3 HTTP API（serve.py 运行后）

| 接口                                  | 方法 | 用途                             |
|---------------------------------------|------|----------------------------------|
| `GET /state`                          | —    | 当前阶段、进度、历次评估结果     |
| `GET /skills`                         | —    | 所有 Skill 的当前内容 + 版本历史 |
| `GET /skill_version/{name}/{version}` | —    | 取某个 Skill 的特定版本内容      |
| `POST /reset`                         | —    | 还原到初始态（删除所有进化历史） |
| `GET /stream/baseline`                | SSE  | 基线评估流（60 题事件流）        |
| `GET /stream/block/{N}`               | SSE  | 第 N 块（10 题 + Nudge 或跳过）  |
| `GET /stream/final`                   | SSE  | 最终评估流                       |

**SSE 事件类型**：

| 事件                                 | 时机                        | 关键字段                                        |
|--------------------------------------|-----------------------------|-------------------------------------------------|
| `eval_q`                             | 基线/最终每题               | `run_id`、`id`、`correct`、`category`、`answer` |
| `eval_complete`                      | 基线/最终整体完成           | `run_id`、`accuracy`、`by_category`             |
| `question_start` / `question_result` | 块内每题                    | `seq`、`correct`、`answer`                      |
| `block_complete`                     | 块的 10 题跑完              | `correct`、`total`、`accuracy`                  |
| `nudge_start`                        | 有失败，将调 Reviewer       | `nudge_num`、`failure_count`                    |
| `nudge_skipped`                      | 全对，跳过 Reviewer         | `reason`                                        |
| `reviewer_analysis`                  | Reviewer 返回分析文本       | `analysis`                                      |
| `skill_action`                       | Reviewer 的某个操作执行成功 | `action`、`skill_name`、`reason`                |
| `nudge_complete`                     | Nudge 所有操作完成          | `num_actions`                                   |
| `probe_start` / `probe_result`       | Probe eval 开始/结束        | `accuracy`、`by_category`                       |
| `phase_change`                       | 状态迁移                    | `phase`、`current_block`                        |
| `done`                               | 当前流结束                  | —                                               |

---

## 五、关键配置项

### 5.1 改 Nudge 间隔

编辑 `data/demo_script.json`：

```json
{
  "nudge_interval": 10,
  "probe_question_ids": [...]
}
```

### 5.2 改评估关键词

编辑 `data/eval_set.json`。给某题加 `_note` 字段记录修改意图：

```json
{
  "id": 14,
  "ground_truth": {
    "_note": "required 从 '免' 改 '平台'：Agent 更常用'由平台承担'而非'免运费'",
    "required": ["平台"],
    "forbidden": ["自付", "自己付"]
  }
}
```

### 5.3 切换 LLM 提供商

默认用 DeepSeek。切换到 DashScope（`src/agent.py` 和 `src/background_reviewer.py`）：

```python
self.client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
self.model = "qwen-plus"
```

---

## 六、FAQ / 常见问题

### Q1: 基线准确率为什么这么低（~22%）？

因为 Agent 按契约规定， **对 Skill 未覆盖的问题必须说"需要联系人工客服"**， 评估器据此判定推脱失败。初始只有 refund 和
vip_benefits 两个 Skill， logistics / promotion / digital_goods / payment_account 四类的问题 Agent 会诚实地 推脱，准确率约
0%。加上 refund_basic 100%、vip_refund 部分命中，整体约 22%。

**这才是诚实的基线。** 以前的"基线 48%" 是 Agent 靠预训练常识猜对的，不是 Skill 能力的真实反映。

### Q2: block 1 全对却触发了 Nudge？

如果你看到这个现象，说明服务还在跑旧代码。 **当前实现**严格保证：

- 全对 → emit `nudge_skipped` 事件，不调 Reviewer，Skill 不变
- 有失败 → 仅失败样本送 Reviewer，不掺入历史

重启 uvicorn（推荐加 `--reload`）解决。

### Q3: Reviewer 为什么有时候只 patch 一条就返回？

这是 **刻意设计**。Reviewer 系统提示里明确："按失败频次从高到低 **只修复 1~2 类**，留出进化梯度"。如果一次把所有失败都修完，后续块就没进化空间了——
学生看到的曲线会变成"一次跃升"而不是"每块都有小幅改进"。

### Q4: LLM 输出有随机性，同一实验跑两次准确率不一样？

LLM 即使 temperature=0 也不完全确定，±5% 波动正常。关注 **趋势**（基线 → 最终 提升幅度）而非绝对值。如果波动 >10%，检查：

- 是否切换了不同 LLM
- 是否修改过 `eval_set.json` 的 ground truth
- 初始 Skill 有没有被改过（`diff skills/ outputs/skills_original/`）

### Q5: 基线评估/最终评估的问题点击不能展开

**已修复**。若仍遇到，说明看到的是缓存页面——硬刷新（Ctrl+Shift+R）即可。

问题根因：基线题和最终题都用同样的 DOM id 前缀，导致 `getElementById` 返回第一个 匹配（基线的）。修复方案：服务端 eval_q 事件带
`run_id` 字段，客户端用
`q_b{id}`（baseline）和 `q_f{id}`（final）分开，永不冲突。

### Q6: 想从第 N 块继续跑，不从头开始？

当前不支持断点续跑（课堂演示通常需要整体连贯）。变通方案：

- Web UI 的重置按钮还原所有状态
- 不点重置时，后端维护 `current_block`，可以从该块继续
- 如需跳过某块：暂无 UI 入口，需手动修改后端状态

### Q7: 怎么看 Agent 某题用的是哪个版本的 Skill？

`outputs/eval_runs/{run_id}.json` 有 `skill_versions_active` 字段：

```json
{
  "skill_versions_active": {
    "refund": 3,
    "vip_benefits": 2,
    "digital_goods_refund": 1
  }
}
```

### Q8: evolution_log.json 里的 question_comparison 怎么用？

`question_comparison[qid]` 包含这道题在所有评估中的答案历史：

```python
import json
log = json.load(open("outputs/evolution_log.json", encoding="utf-8"))
for h in log["question_comparison"]["11"]["history"]:
    versions = h["skill_versions"]
    mark = "✓" if h["correct"] else "✗"
    print(f"[{h['label']:<30}] refund=v{versions.get('refund','-')} "
          f"{mark} {h['answer'][:60]}")
```

### Q9: 怎么自己加评估题？

编辑 `data/eval_set.json`：

```json
{
  "id": 61,
  "category": "refund_basic",
  "difficulty": "medium",
  "question": "...",
  "ground_truth": {
    "required": ["..."],
    "forbidden": ["..."]
  },
  "initial_skill_handles": true/false,
  "note": "为什么这道题值得测"
}
```

然后把它加到 `demo_script.json` 的 `questions` 数组或 `probe_question_ids`
列表里让它参与演示流程。

### Q10: forbidden 关键词被否定前置误伤怎么办？

当前算法检查 forbidden 前 4 字是否含 `不/无/非/未/没`。如果：

- 合法答案："不可直接取消" → "直接取消" 前有"不"→ 跳过，不算命中 ✓
- 错误答案："可以直接取消" → "直接取消" 前无否定 → 命中，标为 fail ✓

如果你的题遇到跨句否定（如"虽然不支持。但建议..."），4 字窗口抓不到跨句否定。 这是规则评估的已知局限，可以：

- 把 forbidden 改成更长的正向错误词（如 "可以直接取消" 代替 "直接取消"）
- 在 `_note` 里记录这个 trade-off
- 接受偶尔的误判

---

## 七、典型输出示例

命令行运行 `python src/demo_runner.py` 的片段：

```text
────────────────────────────────────────────────────────────
基线评估（初始 Skills，无进化）
────────────────────────────────────────────────────────────
总体准确率: 13/60 = 21.7%
分类准确率:
  digital_goods          1/11  █ 9%
  logistics              0/ 8   0%
  payment_account        0/ 7   0%
  promotion_refund       0/12   0%
  refund_basic           9/10  ██████████████████ 90%
  vip_refund             3/12  █████ 25%

...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  本块 [refund_basic] 完成: 10/10 = 100.0%
  ✓ 本块全对，跳过 Nudge 和 Probe eval
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  本块 [digital_goods] 完成: 0/10 = 0.0%
  🔔 Nudge 触发（10 条失败样本注入 Reviewer）
  [Reviewer] 分析：本轮失败 10 条，全部是数字商品退款类问题。
             Agent 均推脱给人工客服。根本原因是现有 Skill 未覆盖数字商品...
  [SkillManager] ✓ 创建 Skill: digital_goods_refund
  ✓ 执行了 1 个 Skill 操作
  Probe eval: 18/30 = 60.0%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
