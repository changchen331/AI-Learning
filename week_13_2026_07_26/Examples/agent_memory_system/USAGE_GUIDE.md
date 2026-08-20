# USAGE_GUIDE.md — 代码调用与测试指南

## 一、环境准备

### 依赖安装

```bash
cd agent_memory_system
pip install -r requirements.txt
```

| 包                  | 用途                               |
|---------------------|------------------------------------|
| openai>=1.0.0       | LLM + Embedding（OpenAI 兼容接口） |
| faiss-cpu>=1.7.4    | Layer 4 语义记忆向量库             |
| fastapi>=0.110.0    | Web 服务框架                       |
| uvicorn>=0.29.0     | ASGI 服务器                        |
| pydantic>=2.0.0     | 请求/响应数据验证                  |
| numpy>=1.24.0       | 向量操作                           |
| apscheduler>=3.10.0 | HEARTBEAT 定时调度器               |

### API Key 配置

默认使用 DeepSeek，备选 Qwen：

```powershell
# Windows PowerShell（DeepSeek，默认）
$env:DEEPSEEK_API_KEY = "sk-xxx"

# 切换为 Qwen
$env:LLM_PROVIDER = "qwen"
$env:DASHSCOPE_API_KEY = "sk-xxx"
```

```bash
# Linux/Mac
export DEEPSEEK_API_KEY="sk-xxx"
# export LLM_PROVIDER="qwen"  # 可选，默认为 deepseek
```

Embedding 无论哪个模式都需要 DashScope Key（DeepSeek 无 Embedding API）：

```bash
export DASHSCOPE_API_KEY="sk-xxx"
```

---

## 二、Step 1 — CLI 版演示（agent.py）

### 启动

```bash
python src/agent.py
```

### 内部流程

1. 打印当前 LLM 提供商和四层记忆加载情况（含每日日志层）
2. 进入对话循环
3. 每次输入前：Layer 4 混合检索（FAISS 向量 + FTS5/BM25）相关历史记忆
4. 组装 Context：System Prompt（Layer 3 + 今+昨日志）+ 混合检索记忆（Layer 4）+ 会话历史（Layer 2）+ 当前输入
5. 流式输出 LLM 回复
6. 消息写入 SQLite

### 内置命令

| 命令      | 作用                                      |
|-----------|-------------------------------------------|
| `/flush`  | 手动触发 Memory Flush，打印三个 Pass 进度 |
| `/memory` | 显示当前 USER.md 和 MEMORY.md 内容        |
| `/layers` | 重新打印四层记忆加载情况                  |
| `/new`    | 开始新会话（不触发 flush）                |
| `/exit`   | 退出（自动触发 flush）                    |

### 预期输出

```
Agent 记忆系统 — CLI 演示
当前模型：DeepSeek V4 Flash  （切换：LLM_PROVIDER=deepseek 或 qwen）
────────────────────────────────────────────────────────────
  四层记忆加载情况
────────────────────────────────────────────────────────────
  🧠 Layer 3a  SOUL.md（人格定义）     [1200 字符]
  🫧 Layer 2   每日日志（今天 + 昨天） [0 字符]  ← 首次为空
  👤 Layer 3b  USER.md（用户画像）     [312 字符]
  📋 Layer 3c  AGENTS.md（操作规范）   [900 字符]
  💾 Layer 3d  MEMORY.md（长期记忆）   [0 字符]  ← 首次为空
  🔍 Layer 4   混合检索（向量 0.7 + BM25 0.3）  [暂无命中]
────────────────────────────────────────────────────────────
```

---

## 三、Step 2 — Web 版演示（serve.py + index.html）

### 启动服务

```bash
uvicorn src.serve:app --host 0.0.0.0 --port 8000
```

服务启动时自动完成：

- 加载 SQLite / FAISS / Markdown 文件
- 启动 HEARTBEAT 调度器（读取 HEARTBEAT.md，注册 cron job）
- 建立当前会话

### 访问

浏览器打开 `http://localhost:8000`

### 界面功能

**顶栏按钮：**

- **新会话**：关闭当前会话、开始新会话（不清记忆）
- **出厂重置**：调 `/reset` 回到出厂初始态（二次确认，清空所有记忆、对话、索引并重载调度器）

**左侧聊天区：**

- 正常对话，支持斜杠命令（见下节）
- 普通回复用白色气泡，HEARTBEAT 推送用紫色小气泡（带 ⏰ 标记）

**右侧记忆面板（可折叠）：**

- ⚡ 四层记忆加载：每次对话显示哪层被加载、加载了多少字符（含每日日志层）
- 🔍 混合检索结果：命中的历史记忆条目（相似度百分比 + 来源 vector/bm25/both）
- 💾 Memory Flush：点击按钮，逐步展示三个 Pass 进度
- 📊 记忆统计：条目数、FAISS 向量数、FTS 条目数、历史会话数
- 各 Markdown 文件查看器（USER/MEMORY/SOUL/AGENTS/HEARTBEAT）

### Web 斜杠命令

在聊天输入框输入以下命令：

| 命令      | 作用                                     |
|-----------|------------------------------------------|
| `/flush`  | 触发 Memory Flush，右侧面板实时展示进度  |
| `/memory` | 展开并滚动到 USER.md / MEMORY.md 查看器  |
| `/layers` | 展开并滚动到四层记忆加载面板             |
| `/new`    | 开始新会话                               |
| `/reset`  | 回到出厂初始态（二次确认，清空所有记忆） |
| `/help`   | 显示命令列表                             |

### SSE 事件类型（教学参考）

**对话流（`/chat`）：**

| 事件               | 触发时机         | 内容                                                    |
|--------------------|------------------|---------------------------------------------------------|
| `memory_load`      | 对话开始         | 四层加载结果（文件名 + 字符数）                         |
| `semantic_search`  | Layer 4 检索完成 | Top-K 结果（类别/标题/内容/分数/来源 vector/bm25/both） |
| `context_assembly` | Context 组装完成 | 总字符数、历史轮数                                      |
| `token`            | LLM 流式输出     | 每个 token 片段                                         |
| `done`             | 回复完成         | 完整回复、消息计数                                      |

**Flush 流（`/flush`）：**

| 事件               | 触发时机        | 内容                                          |
|--------------------|-----------------|-----------------------------------------------|
| `flush_start`      | Flush 开始      | 会话 ID、消息数                               |
| `flush_pass1`      | Pass 1 完成     | USER.md 更新项列表                            |
| `flush_pass2`      | Pass 2 完成     | 新增记忆条目列表                              |
| `flush_pass3`      | Pass 3 完成     | 向量化条目数、FAISS 索引总数（同步写入 FTS5） |
| `flush_compaction` | Compaction 触发 | 压缩前后条目数                                |
| `flush_done`       | Flush 结束      | 完整摘要                                      |

**HEARTBEAT 广播流（`/stream`，持久连接）：**

| 事件                       | 触发时机               | 内容                   |
|----------------------------|------------------------|------------------------|
| `heartbeat_connected`      | 页面建立连接时         | 当前任务列表           |
| `heartbeat_task_added`     | 对话意图检测到新建任务 | 任务名/trigger/描述    |
| `heartbeat_task_cancelled` | 对话意图检测到取消任务 | 被取消的任务名         |
| `heartbeat_start`          | 定时任务触发           | 任务名/action/触发时间 |
| `heartbeat_message`        | 任务执行完成           | LLM 生成的消息内容     |
| `heartbeat_reloaded`       | HEARTBEAT.md 热重载    | 当前任务数             |

---

## 四、HEARTBEAT 定时任务演示

### 快速测试（每分钟触发）

在 `memory/HEARTBEAT.md` 的 `TASKS_START/END` 之间添加：

```markdown
### TASK: test_task
trigger: */1 * * * *
enabled: true
action: send_message
description: 每分钟测试消息
prompt: 生成一条简短的测试问候。
added: 2026-05-08
```

保存后调度器 60 秒内自动重载，约 1 分钟后前端出现紫色气泡。

### 通过对话设置任务（自然语言）

直接在聊天框说：

- "每天早上 9 点提醒我喝水"
- "帮我每周一发一条本周计划提醒"
- "每天晚上帮我汇总今天的对话"

系统后台自动走：正则初筛 → LLM 判断 → 写入 HEARTBEAT.md → 调度器重载 → 紫色气泡确认。

### 通过对话取消任务

- "取消早上的提醒"
- "不要再发消息了"
- "停止那个每天的任务"

### 支持的 action 类型

| action                 | 说明                                               |
|------------------------|----------------------------------------------------|
| `send_message`         | LLM 根据 `prompt` 和用户画像生成一条消息推送到前端 |
| `summarize_sessions`   | 汇总今日对话，追加 MEMORY.md [event] 条目          |
| `compact_memory`       | 触发 Memory Compaction，压缩旧记忆                 |
| `user_profile_refresh` | 重新分析全部记忆，刷新 USER.md                     |

---

## 五、作为模块调用

```python
import sys
sys.path.insert(0, "path/to/agent_memory_system")

from src.session_db import SessionDB
from src.memory_loader import MemoryLoader
from src.vector_store import VectorStore
from src.fts_store import FTSStore
from src.retrieval import HybridRetriever
from src.memory_flush import MemoryFlusher

db = SessionDB()
loader = MemoryLoader()
vs = VectorStore()
fts = FTSStore()
retriever = HybridRetriever(vs, fts)   # 向量 0.7 + BM25 0.3 混合
flusher = MemoryFlusher()              # 内部已自建 vs + fts 并在 flush 中同步

# 开始会话
sid = db.new_session()

# 构建 System Prompt（含 SOUL/今+昨日志/USER/AGENTS/MEMORY 近期条目）
prompt_result = loader.build_system_prompt(recent_memory_limit=10)
print(f"System Prompt 共 {prompt_result.total_chars} 字符")

# 混合检索（FAISS 语义 + FTS5/BM25 关键词，取并集）
results = retriever.search("用户的饮食偏好", top_k=3)
for r in results:
    print(f"[{r['category']}] {r['title']}  分数 {r['score']:.2%}  来源 {r.get('source')}")

# 记录消息
db.add_message(sid, "user", "我每天早上喝咖啡")
db.add_message(sid, "assistant", "记住了，你喜欢喝美式。")

# Memory Flush（Pass1 更新 USER.md；Pass2 写 MEMORY.md + 每日日志；Pass3 同步 FAISS + FTS5）
messages = db.get_session_messages(sid)
result = flusher.flush(messages, sid)
print(result.summary())
```

---

## 六、备份与恢复（reset.py）

### 命令速查

| 命令                                 | 作用                           |
|--------------------------------------|--------------------------------|
| `python src/reset.py backup`         | 保存当前状态快照（时间戳命名） |
| `python src/reset.py backup <名称>`  | 指定名称保存快照               |
| `python src/reset.py restore`        | 恢复最近一次快照               |
| `python src/reset.py restore <名称>` | 恢复指定快照                   |
| `python src/reset.py restore --list` | 列出所有快照                   |
| `python src/reset.py factory`        | 回到出厂初始态                 |

### 备份内容（7 项）

- `memory/USER.md` — 用户画像
- `memory/MEMORY.md` — 跨会话记忆条目
- `memory/HEARTBEAT.md` — 定时任务配置
- `memory/SOUL.md` — Agent 人格
- `memory/AGENTS.md` — 操作规范
- `data/vector_index/` — FAISS 向量索引
- `outputs/sessions/` — SQLite 会话数据库

### 推荐工作流

```bash
# 首次使用前保存初始状态
python src/reset.py backup initial

# 演示后回到空白状态（CLI）
python src/reset.py factory

# 或通过 Web UI 输入 /reset（带二次确认弹窗）
```

注意：`/reset` 还会立即重载调度器，停止所有已注册的定时任务。

---

## 七、调试与常见问题（FAQ）

**Q: 启动时报 `EnvironmentError: 使用 DeepSeek V4 Flash 需要设置 DEEPSEEK_API_KEY`**

A: 设置 API Key：`$env:DEEPSEEK_API_KEY = "sk-xxx"`。或切换到 Qwen：`$env:LLM_PROVIDER = "qwen"`。

**Q: 前端没有收到 HEARTBEAT 推送消息**

A: 查看服务日志是否有 `[broadcast] heartbeat_message，当前监听数：1`。若监听数为 0，说明浏览器的 `/stream` 连接断开了，刷新页面重连即可。

**Q: 说了"每天提醒我"，但没有设置成定时任务**

A: 后台 LLM 判断不一定每次都认为是调度请求。直接编辑 `memory/HEARTBEAT.md` 手动添加任务更可靠，调度器 60 秒内自动重载。

**Q: /reset 后定时任务还在跑**

A: 确认使用的是 Web `/reset` 命令或 `python src/reset.py factory`（两者都会重载调度器）。如果用 CLI 的 `cmd_factory()`
，需要手动调用 `hb_scheduler._load_tasks()`。

**Q: Memory Flush 后 USER.md 没有变化**

A: 对话内容需包含明显的用户信息。尝试说"我叫XX，是一名XX工程师"，再 `/flush`。

**Q: 混合检索总是返回空结果**

A: Layer 4 需要先有记忆条目。先对话 → `/flush` → 再对话，此后才会有命中。若 FAISS 有条目但 FTS 为空（如从旧版升级未回填），可调
`flusher.fts.rebuild_from_entries(flusher._parse_memory_entries_for_faiss())` 同步；FTS5 不可用时 `HybridRetriever`
会自动退化为纯向量，仍能返回结果。

**Q: FAISS 报 `ImportError`**

A: `pip install faiss-cpu`，不要装 `faiss-gpu`（除非有 GPU 环境）。

**Q: Compaction 何时触发？**

A: MEMORY.md 条目数 ≥ 50 时。保留最新 20 条，压缩其余。阈值可在 `memory_flush.py` 的 `COMPACTION_THRESHOLD` 修改。

**Q: serve.py 启动后访问 / 报 404**

A: 确认 `index.html` 在项目根目录（与 `src/` 平级）。
