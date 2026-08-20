# RESUME_GUIDE.md — 求职简历指导

## 一、可量化数据（填写实测结果后使用）

| 数据项                     | 参考值                                                  | 你的实测值 |
|----------------------------|---------------------------------------------------------|------------|
| 记忆层数                   | 4 层（工作/短期/长期/语义）+ 近端日志（今+昨 48h）      | —          |
| Memory Flush 平均耗时      | 约 3~5 秒（3 次 LLM 调用）                              | —          |
| 单会话自动 Flush 阈值      | 20 条消息                                               | —          |
| Compaction 阈值            | 50 条，压缩保留最新 20 条                               | —          |
| FAISS 向量维度             | 1536 维（text-embedding-v3）                            | —          |
| Layer 4 检索策略           | 混合检索：向量 0.7 + BM25 0.3，取并集                   | —          |
| 全文索引                   | SQLite FTS5 + bm25()（与 SQLite 同库，零依赖）          | —          |
| 中文分词                   | 逐字空格分词（兼容 unicode61，支持任意子串召回）        | —          |
| 混合检索延迟               | < 10ms（本地 FAISS + FTS5）                             | —          |
| Markdown 记忆文件数        | 5 个配置（SOUL/USER/MEMORY/AGENTS/HEARTBEAT）+ 每日日志 | —          |
| HEARTBEAT 支持 action 类型 | 4 种（send_message/summarize/compact/refresh）          | —          |
| 调度器热重载延迟           | ≤ 60 秒（watcher 轮询），即时触发时 0 秒                | —          |
| SSE 广播监听数             | 支持多客户端，每连接独立 Queue                          | —          |

---

## 二、项目名称怎么写

| 写法                                                                                           | 评价                            |
|------------------------------------------------------------------------------------------------|---------------------------------|
| ❌ "做了一个聊天机器人"                                                                        | 完全没有技术含量                |
| ❌ "实现了带记忆的 AI 助手"                                                                    | 模糊，不知道怎么实现的          |
| ✅ "基于四层记忆架构的 AI 个人助手（SQLite + FAISS + FTS5 + Markdown + APScheduler）"          | 清晰展示技术栈                  |
| ✅ "LLM Agent 记忆系统：Memory Flush + HEARTBEAT 自主调度 + 向量/BM25 混合检索 + 跨会话持久化" | 突出核心机制，体现 Agent 主动性 |

---

## 三、按岗位写法

### 算法工程师 / NLP 工程师

```
【项目】基于四层记忆架构的 AI 个人助手
【技术】Python / DeepSeek API / FAISS / FTS5 / SQLite / FastAPI / APScheduler / SSE
【亮点】
· 设计四层记忆体系：Context Window（工作）/ SQLite 会话历史 + 每日日志（短期/近端）/
  Markdown 配置文件（长期）/ FAISS 向量库 + FTS5 全文索引（语义），覆盖毫秒到跨周的记忆时效
· 实现 Two-Pass Memory Flush：Pass 1 LLM 提取用户信息更新画像，
  Pass 2 提取记忆条目追加长期记忆 + 每日日志，Pass 3 向量化写入 FAISS + FTS5 同步
· 实现 Memory Compaction：条目超 50 时 LLM 语义压缩旧记忆，
  保留信息密度同时控制 Context 占用
· 实现 HEARTBEAT 自主调度：对话中正则+LLM 双层检测调度/取消意图，
  自动写入 HEARTBEAT.md，APScheduler 热重载，任务触发结果通过 SSE 实时推送前端
```

### 后端工程师

```
【项目】Agent 记忆系统 — FastAPI + SSE 实时可视化服务
【技术】Python / FastAPI / SQLite / FTS5 / FAISS / APScheduler / SSE / uvicorn
【亮点】
· FastAPI lifespan 单例管理：向量索引、FTS5 索引、数据库、调度器在启动时一次性初始化，
  多请求复用，避免重建开销
· 双 SSE 通道设计：/chat 做对话流式推送（token 级），/stream 做
  HEARTBEAT 广播长连接（每20秒 keepalive，支持多客户端独立 Queue）
· SQLite 三表设计（sessions + messages + memory_fts FTS5），覆盖会话生命周期全流程：
  创建→写消息→Flush 标记→关闭，支持跨会话历史查询；FTS5 与 FAISS 在同一 HybridRetriever
  做 0.7/0.3 加权并集检索，FTS5 不可用时自动降级纯向量
· APScheduler 内嵌 FastAPI event loop，cron 任务执行结果通过 asyncio.Queue
  广播到所有已连接的 SSE 客户端
```

### Agent / AI 应用工程师

```
【项目】具有自主调度能力的 AI 个人助手
【技术】Python / LLM API / FAISS / APScheduler / FastAPI / SSE
【亮点】
· 实现 Agent 四层记忆体系，解决 LLM 无状态导致的跨会话遗忘问题
· 设计 HEARTBEAT 自主调度机制：Agent 通过对话理解用户调度意图，
  自动写入任务配置文件并驱动调度器，无需用户显式操作 API
· 实现调度意图双向检测（新建/取消），正则初筛零成本过滤，
  LLM 二次判断保证准确率，取消意图优先防止误触发
· HEARTBEAT.md 作为任务"配置语言"：人类可读可编辑，支持热重载，
  LLM 原生理解，Git 可版本管理
```

---

## 四、按经验层级写法

### 应届生版

```
基于四层记忆架构的 AI 个人助手
- 独立设计实现四层记忆体系（工作/短期/长期/语义），覆盖不同时效的信息存储与检索
- 实现 Memory Flush 机制：LLM 从对话自动提取用户偏好和记忆条目，写入持久化文件
- 实现 HEARTBEAT 定时调度：APScheduler 驱动，对话中自然语言即可设置/取消定时任务
- 基于 FastAPI + SSE 构建可视化演示服务，实时展示记忆加载、Flush 进度和任务推送
```

### 1~3年版

```
AI Agent 记忆与调度系统
- 设计四层记忆架构：针对不同时效信息采用不同存储策略（SQLite/Markdown/FAISS+FTS5），
  通过向量+BM25 混合检索在 Context 容量限制内最大化记忆利用率
- 实现 Two-Pass Memory Flush：分离"提取"和"写入"两个 LLM 任务，
  避免单 Pass 任务混淆导致的低质量输出
- 设计 HEARTBEAT 自主调度：正则初筛 + LLM 判断双层意图检测，
  Markdown 文件作为任务配置，APScheduler 热重载，任务结果 SSE 实时广播
- 实现 Compaction 机制控制记忆增长，类比 LLM 上下文压缩原理
```

### 3年以上版

```
AI Agent 长期记忆与自主调度系统
- 主导设计四层分级记忆架构，解决 LLM 无状态导致的跨会话记忆丢失问题；
  动态 Top-K 混合检索（FAISS 向量 0.7 + FTS5/BM25 0.3 加权并集）+ 全量注入（Markdown + 今+昨日志），
  在 Context 容量限制内最大化记忆利用率
- 设计 LLM-as-Extractor 的 Memory Flush 流程，Two-Pass 设计分离提取与写作任务，
  覆盖隐式偏好和跨句推断场景；Compaction 机制通过语义摘要替代简单截断
- 设计 HEARTBEAT 自主调度机制：对话驱动而非 API 驱动，正则初筛+LLM 判断双层
  意图检测，Markdown 配置文件热重载，APScheduler 嵌入 FastAPI event loop，
  任务结果通过 SSE 多路广播到前端
- 建立完整的状态管理体系：备份/恢复/出厂重置 CLI，/reset 端点同步重载调度器，
  确保记忆状态与运行时状态一致性
```

---

## 五、好句 vs 差句对比

| 差句               | 好句                                                                                                                         |
|--------------------|------------------------------------------------------------------------------------------------------------------------------|
| 实现了记忆功能     | 设计四层分级记忆体系，覆盖工作/短期/长期/语义四个时效维度                                                                    |
| 用了 FAISS 做检索  | FAISS IndexFlatIP + L2 归一化做余弦语义检索，再叠 SQLite FTS5/BM25 关键词检索，0.7/0.3 加权并集，每轮动态注入 Top-K 相关记忆 |
| 做了记忆压缩       | 实现 Compaction：条目超 50 时 LLM 语义压缩旧条目并重建向量索引，类比 LLM 上下文压缩原理                                      |
| 会话结束后保存记忆 | Two-Pass Memory Flush：Pass 1 更新用户画像，Pass 2 提取事件记忆，Pass 3 向量化写入 FAISS                                     |
| 实现了定时任务     | 对话中正则+LLM 双层意图检测，自动写入 HEARTBEAT.md，APScheduler 热重载，任务结果 SSE 广播                                    |
| 用 SSE 推送消息    | 双 SSE 通道：/chat 做 token 级流式对话，/stream 做 HEARTBEAT 广播长连接，keepalive 防超时断连                                |

---

## 六、面试常见问题

**Q: 为什么要四层记忆，不直接把所有历史都塞进 Context？**

A: 不同信息有不同时效性和访问模式。当前对话（Layer 1）必须完整保留；会话历史（Layer 2）按时间顺序拼入；每日日志（近端，加载今+昨）给
48 小时连续感；用户画像（Layer 3）每次全量注入保证一致性；历史记忆可能有几十上百条（Layer 4），不能全塞，需要检索取最相关的。全塞会撑满
Context，检索越用越精准，越用越懂你。

**Q: Layer 4 为什么用向量 + BM25 混合，不只用向量？**

A: 向量擅长"措辞不同但语义相近"（"爱喝咖啡" ↔ "每天来杯美式"），但对精确符号/专名（PostgreSQL、某 API
名）召回弱，且中文向量易漂移。BM25/FTS5 擅长精确关键词命中，正好互补。两者取并集、按 0.7/0.3 加权，任一方法命中都进候选池，兼顾召回率和精度。FTS5
用 SQLite 自带扩展零依赖，不可用时自动降级纯向量，可用性有保障。

**Q: Memory Flush 为什么用 LLM 提取，不用关键词规则？**

A: 用户不会说"我的偏好是咖啡"，会说"最近天气热，每天都要来一杯美式"。规则只能处理显式格式，LLM 能跨句推断、识别隐式偏好。实测同一段对话，规则提取覆盖率约
30%，LLM 两阶段提取覆盖率约 85%+。

**Q: Two-Pass Flush 和 Single-Pass 区别？**

A: Single-Pass 让 LLM 同时理解+提取+写文档，容易混淆任务边界，要么提取不完整要么格式乱。Two-Pass 把 1a（提取→JSON）和
1b（JSON+现有文档→更新文档）分开，每步单一任务，LLM 表现稳定得多。

**Q: HEARTBEAT 和普通 cron 有什么区别？**

A: 普通 cron 是静态配置，用户要懂 crontab 语法，改配置需要 SSH。HEARTBEAT
是"对话驱动的调度"——用户说自然语言，系统自动识别意图写入配置文件并热重载，任务结果也推回对话界面，全程无需离开聊天界面。核心亮点是
Agent 自主决定"何时主动行动"。

**Q: 调度意图检测为什么要两层（正则 + LLM）？**

A: 正则初筛成本接近零（纯内存操作），绝大多数普通消息直接跳过，不走 LLM；只有可能含调度意图的消息才触发 LLM 判断，控制 API
调用成本。两层设计既保证覆盖率（正则宽松）又保证准确率（LLM 精判），避免把普通聊天误识别为定时任务。

**Q: Compaction 和 RAG 的 Chunking 有什么区别？**

A: Chunking 是把外部文档切片，用于检索。Compaction 是压缩 Agent 自己积累的记忆，核心是"保留信息密度、减少条目数量"，用 LLM
做语义摘要而不是机械截断。两者都解决 Context 容量问题，但 Compaction 处理的是 Agent 的"个人历史"，不是外部知识库。

**Q: 为什么 /reset 后调度器也要重载？**

A: reset 写回了 HEARTBEAT.md 的初始内容（无用户定义的 job），但 APScheduler 的 job 存在内存里，不感知文件变化。如果不主动调
`_load_tasks()`，旧 job 继续触发，页面还会收到"已清除"任务的消息，状态不一致。这是运行时状态和持久化状态同步的典型工程问题。

**Q: SQLite 为什么不用 ORM？**

A: 教学项目，表结构固定且简单（sessions + messages 两张关系表 + memory_fts 一张 FTS5 虚拟表），直接用 `executescript` +
参数化查询足够，ORM 引入额外抽象层反而掩盖了 SQL 操作的教学细节。每次操作独立建连接关闭的方式也直观展示了数据库连接管理的基本概念。
