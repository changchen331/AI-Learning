# ARCHITECTURE.md — GraphRAG 年报知识图谱问答

## 1. 项目定位

**场景**：把 15 份 A 股上市公司年报（5 家×3 年：贵州茅台/五粮液/海康威视/宁德时代/中国平安）构建成知识图谱存入 Neo4j，用
GraphRAG 的 Local / Global 双路径做多跳问答。

**为什么是年报**：实体类型丰富（公司/子公司/高管/股东/产品/财务指标/地区）、天然多跳（子公司链、跨年财务趋势、实控人关系）、真实工程噪声（256
张表/份、层级 section）。规模适中——15 份不会跑不动，但足够体现构建成本。

**核心方案对比**：

| 方案                               | 实现                                         | 适合                       |
|------------------------------------|----------------------------------------------|----------------------------|
| GraphRAG（本项目）                 | 手写抽三元组 + Neo4j + Leiden + Local/Global | 多跳关系、可溯源、结构化   |
| 向量 RAG（复用 rag_annual_report） | FAISS + BM25 + Rerank                        | 简单事实召回、文本语义匹配 |
| 长 LLM 上下文                      | 单文档塞进 prompt                            | 单文档分析，无持久知识库   |

本项目不用微软 graphrag 库（它不走 Neo4j），而是手写 GraphRAG 思路 + Neo4j 存储，让学生看清每一步。社区检测走 Python
networkx+leidenalg（Neo4j GDS 插件没装，优雅降级）。

## 2. 整体流水线

```
离线索引（一次性）                         在线查询（每次请求）
┌─────────────────────────┐              ┌──────────────────────┐
│ parsed blocks (15份)     │              │ 用户问题              │
│      ↓                  │              │   ↓                  │
│ extract_triples.py      │              │ 实体链接(embedding)   │ ← retrieve.py
│  LLM 抽 (s,p,o) 三元组   │              │   ↓                  │
│      ↓                  │              │ 邻居扩展(1跳,非指标优先)│
│ build_graph.py          │              │   ↓                  │
│  实体规范化 + Neo4j MERGE │              │ 子图提取              │
│      ↓                  │              │   ↓                  │
│ community_detect.py     │              │ 上下文构建            │ ← /query/debug
│  Leiden 分社区 + LLM 摘要 │              │   ↓                  │
│      ↓                  │              │ LLM 合成答案          │
│ communities.json        │              │   ↓                  │
└─────────────────────────┘              │ 答案 + 引用           │
                                         │                      │
                                         │ Global: 社区摘要 →    │
                                         │   Map → Reduce        │
                                         └──────────────────────┘
```

各步对应脚本：`extract_triples.py` → `build_graph.py` → `community_detect.py` → `retrieve.py`（Local+Global）→ `serve.py`
（HTTP）→ `eval_compare.py`（对比）。

## 3. 各环节技术选型

### 3.1 LLM / Embedding 提供商

- **LLM**：DeepSeek `deepseek-chat`（即 deepseek-v4-flash），OpenAI 兼容接口。便宜、快、支持 `response_format=json_object`。
- **Embedding**：DashScope `text-embedding-v3`（1024 维）。DeepSeek 不提供 Embedding API，固定走阿里云。
- 统一封装在 `llm_client.py`，换提供商只改 `base_url` + 模型名。

### 3.2 三元组抽取（extract_triples.py）—— 对应 PPT 3.3

- **传统 vs LLM**：传统 NER+RE 需 CRF/BiLSTM + 标注，迁移难；LLM 用 Prompt + 结构化 JSON 输出，零标注跨域即用，关系自由定义。
- **Schema 约束**：7 种实体类型 + 8 种关系类型在 Prompt 里显式枚举，LLM 输出可直接入库。
- **成本控制**：15 份×1500 blocks 全抽太贵，按关键 section（公司简介/业务/财务/股东/董事/子公司/研发…）抽前 25 块/份 = 375
  块。 **索引成本比向量 RAG 高 5~10× 是教学点**，不是缺陷。

### 3.3 实体规范化（build_graph.py）—— 核心教学点

LLM 抽取时同一实体出现多种写法（贵州茅台酒股份有限公司 / 贵州茅台 / 茅台），不规范化会把一个实体拆成多个节点 →
图谱破碎、多跳推理断裂。解法： **别名表**把 5 家上市公司统一到规范名 + 挂 stock_code。其他类型仅 strip。年份放边属性而非单独建
CompanyYear 节点——更简单，仍支持跨年查询。

### 3.4 社区检测（community_detect.py）—— 对应 PPT 4.3

- **Leiden 算法**：把密集相连的实体归为一组（一个"主题社区"），社区天然对应公司边界。
- **优雅降级**：本地 Neo4j 没装 GDS 插件，改用 Python networkx + igraph + leidenalg。算法透明可见、零插件依赖。GDS 装了可换，但教学上
  Python 路线更清晰。
- **社区摘要**：每社区取 top-15 度数代表实体，LLM 生成 100 字主题描述，供 Global Search。

### 3.5 在线检索（retrieve.py）—— 对应 PPT 4.4

- **Local Search**：实体链接（embedding 相似度跨越"茅台/贵州茅台酒股份有限公司"字面差异）→ 1 跳邻居扩展（非 Indicator
  优先，避免指标节点挤掉子公司/人物）→ 子图截断 40 节点 → 文本上下文 → LLM 合成。
- **Global Search**：社区摘要 embedding → 取 top-3 相关社区 → Map（每社区生成部分答案）→ Reduce（聚合）。

### 3.6 可视化（static/index.html + viz/force_graph.js）

- **可视化代码隔离**：force-directed 图绘制在独立的 `static/viz/force_graph.js`（vanilla JS 无依赖），主流程 UI 在
  index.html。学生关注主流程，viz 仅辅助。
- `/query/debug` 逐步返回实体链接/子图/上下文/答案，上下文步骤默认展开（看清 LLM 收到什么）。

## 4. 实验结果（真实跑出）

### 4.1 索引规模

| 指标     | 数值                                  |
|----------|---------------------------------------|
| 处理报告 | 15 份                                 |
| 抽取块数 | 375 块（25/份）                       |
| 三元组   | **1153 条**（3.07 条/块）             |
| 抽取耗时 | 659s（~11 分钟）                      |
| 图节点   | **649**                               |
| 图边     | **681**                               |
| 社区数   | **13**（最大 132，最小 4，平均 49.9） |

节点类型：Indicator 411 / Person 69 / Company 56 / Product 45 / Subsidiary 36 / Region 18 / Segment 14 关系类型：REPORTS
414 / SERVES_AS 147 / PRODUCES 45 / HAS_SUBSIDIARY 36 / OPERATES_IN 23 / INVESTS_IN 11 / CONTROLS 5

### 4.2 GraphRAG vs 向量 RAG（15 题，子串宽松匹配）

| 指标       | GraphRAG (Local) | 向量 RAG (BM25+Rerank) |
|------------|------------------|------------------------|
| 答案命中   | 14/15            | 15/15                  |
| 上下文命中 | 14/14            | 14/14                  |
| 平均延迟   | **1.76s**        | 6.02s                  |

**结果解读**：

- GraphRAG **快 3.4×**：向量 RAG 每次查询要 jieba 分词 + BM25 + Rerank 初始化开销；GraphRAG 只做一次 embedding 查询 +
  子图提取。
- 向量 RAG **准确率略高**（15/15 vs 14/15）：GraphRAG 漏了"茅台2022年董事长"——子图截断（非指标优先策略下 Person 丁雄军被挤出
  40 节点子图），导致上下文缺管理层信息。
- 两方案在"比亚迪2024营收"（不在图谱/年报集）上都正确拒绝 → 幻觉控制生效。
- 子串宽松匹配有局限（paraphrase 时低估真实准确率），结论方向仍可靠： **多跳关系类两方案接近，GraphRAG 换来 3.4× 速度 +
  可溯源子图，代价是子图选择策略需调优**。

### 4.3 社区质量（Leiden）

社区天然按公司边界分裂——社区1海康威视杭州子公司群 (132)、社区2茅台 (126)、社区3中国平安 (103)、社区4五粮液 (79)、社区6海康财务指标
(17)…摘要质量高，Global Search 能基于此回答主题类问题。

## 5. 优化方向

| 层面     | 方向                                                                                              |
|----------|---------------------------------------------------------------------------------------------------|
| 数据     | 增大抽取块数（25→50/份）、补抽控股股东/实控人专章（CONTROLS 仅 5 条偏少）                         |
| 模型     | 抽取用更强模型提升实体类型准确度（部分子公司被误标 Company）                                      |
| 子图策略 | 改按查询意图选邻居（问"子公司"优先 HAS_SUBSIDIARY、问"董事长"优先 SERVES_AS），而非统一非指标优先 |
| Global   | 社区摘要纳入关键财务数字，让跨公司数值对比类问题可用                                              |
| 工程     | 实体名向量缓存到磁盘（现每次启动重编码 649 个 ~30s）；Neo4j 装全文索引加速实体链接                |

## 6. 关键工程决策与踩坑

| 问题                                          | 根因                                           | 解法                                                                                |
|-----------------------------------------------|------------------------------------------------|-------------------------------------------------------------------------------------|
| DeepSeek `deepseek-v4-flash` model 名报错     | CLAUDE.md 写的是别名                           | 实际 API 用 `deepseek-chat`，两者同一模型                                           |
| DashScope embedding 单批超 10 报错            | 官方宣传 25，实测上限 10                       | `EMBED_BATCH=10` 分批                                                               |
| LLM 输出 JSON 被截断                          | 董监高表格块三元组多，max_tokens=1024 不够     | 提到 2048                                                                           |
| Neo4j `set-initial-password` 不生效           | 只在 DBMS 首次启动前生效，本机早已初始化       | `conf/neo4j.conf` 设 `auth_enabled=false`，driver `auth=None`                       |
| neo4j 4.4 需 JDK11+，默认 java 是 1.8         | 系统装的是 JRE8                                | 启动前 `JAVA_HOME=C:/Program Files/Java/jdk-14`                                     |
| `leidenalg.find_partition` 报 `missing graph` | leidenalg 0.12 + igraph 1.0.0 的便捷函数有 bug | 用 `ModularityVertexPartition(H)` + `Optimiser().optimise_partition(part)` 显式路径 |
| GDS 插件没装                                  | 本地 community 版无 GDS                        | networkx+leidenalg Python 降级，反而更透明                                          |
| 子图截断把核心实体（宁德时代）丢掉            | all_uids 截断时 linked 节点被挤出              | linked 节点永远在前                                                                 |
| 子图 30 节点全指标、无子公司                  | Indicator 占 63%，随机截断挤掉子公司           | 非 Indicator 邻居优先排序                                                           |
| Local 子图复杂 Cypher 在 4.4 跑不通           | `CALL {}` 子查询 + `[..$n]` 切片不稳           | 拆成两步查询（邻居 uid → 边）                                                       |
| matmul 维度不匹配                             | `embed_query` 返回 [1,1024]                    | `query_vec.reshape(-1)`                                                             |
| StaticFiles 路径错                            | STATIC_DIR 指成了 src/static                   | 改 `parent.parent/static`                                                           |

## 7. 目录结构

```
graphrag_financial_report/
├── src/
│   ├── llm_client.py          # 统一 LLM(DeepSeek)/Embedding(DashScope) 客户端
│   ├── extract_triples.py     # LLM 抽三元组（按 section 抽样）
│   ├── build_graph.py         # 实体规范化 + Neo4j MERGE
│   ├── community_detect.py    # Leiden 社区检测 + LLM 摘要
│   ├── retrieve.py            # Local / Global 双检索
│   ├── serve.py               # FastAPI /query + /query/debug
│   └── eval_compare.py        # GraphRAG vs 向量 RAG 对比
├── static/
│   ├── index.html             # 三栏调试页（主流程 UI）
│   └── viz/force_graph.js     # force 可视化（隔离，非教学重点）
├── outputs/
│   ├── triples.jsonl          # 1153 条三元组
│   ├── communities.json       # 13 社区摘要
│   ├── extract_stats.json
│   └── eval_compare.json
├── data/                      # 复用 rag_annual_report 的 parsed（软链接/路径引用）
├── requirements.txt
├── ARCHITECTURE.md
├── USAGE_GUIDE.md
└── RESUME_GUIDE.md
```
