# 技术架构说明

> 上市公司年报 RAG 问答系统 — 整体方案、选型决策与设计原理

---

## 一、项目定位

本项目以"上市公司年度报告智能问答"为场景，构建一套接近企业级落地标准的 RAG（检索增强生成）系统。数据来自巨潮资讯网（证监会指定披露平台），包含 5 家公司 × 3 年 = 15 份年报 PDF，总量约 85 MB。

项目同时提供两套实现：

| 实现版本 | 定位 | 关键差异 |
|----------|------|---------|
| **原生版**（`src/`） | 企业级生产参考 | 手动控制每个环节，混合检索 + Rerank，代码量约 700 行 |
| **LangChain 版**（`src_langchain/`） | 框架快速原型参考 | 用框架抽象链路，聚焦 LCEL 链路设计，代码量约 200 行 |

---

## 二、整体流水线

```
原始 PDF
    │
    ▼ download_reports.py
数据获取（巨潮资讯 API）
    │
    ▼ parse_pdf.py
PDF 解析（文字 + 表格 + OCR + 章节结构）
    │
    ▼ chunk_documents.py
文档分块（三种策略可切换）
    │
    ▼ build_index.py
向量化 + 索引构建（DashScope API / 本地 BGE）
    │
    ▼ rag_pipeline.py
问答流水线：查询 → 检索 → 重排 → 生成
    │
    ▼ evaluation/
评估（RAGAS 四项指标 + 消融实验）
```

---

## 三、各环节技术选型

### 3.1 数据获取

**巨潮资讯网 API**（`cninfo.com.cn`）是证监会指定的上市公司信息披露平台，数据公开合法。

关键参数：`announceType=report`，按 `searchkey`（公司名+年份）全文搜索。

选型原因：
- 原始 PDF 未经任何清洗，包含扫描件、嵌套表格、页眉页脚噪声，具有真实的工程挑战性
- 同一公司多年数据可测试**时效性冲突处理**
- 不同行业（消费品、金融、新能源、科技）覆盖多种 PDF 复杂度

---

### 3.2 PDF 解析

**组合策略**：pdfplumber（表格）+ PyMuPDF/fitz（文字+字体信息）+ pytesseract（OCR，可选）

| 库 | 职责 | 选型原因 |
|----|------|---------|
| `pdfplumber` | 表格提取 | 基于规则的表格算法，对财务报表的行列识别更准确 |
| `PyMuPDF(fitz)` | 文字+字体元数据 | 提供每个 span 的字体大小和加粗信息，用于识别标题层级 |
| `pytesseract` | 扫描页 OCR | 部分年报附件（审计报告原件）为扫描件，需 OCR 降级处理 |

解析输出的三种块类型：
- `title`：识别依据是字体大于 14pt 或加粗且行长小于 50 字
- `table`：直接转为 Markdown 格式（方便 LLM 读取）
- `text`：正常文本段落

每个块保留完整元数据：`page_num`、`section_path`（章节路径栈）、`is_ocr`、`block_type`。

---

### 3.3 文档分块（Chunking）

提供三种策略，通过修改 `STRATEGY` 变量切换：

#### 策略 A：固定大小分块（`fixed`）
```
chunk_size=500字符，overlap=50字符
```
- 优点：简单可预测
- 缺点：无视句子/段落边界，表格被截断
- 用途：作为 baseline，体现"不好的分块"的代价

#### 策略 B：语义分块（`semantic`）—— 默认
```
遇标题强制切块，段落合并不超过 800 字符
```
- 遇 `title` 块：先 flush 缓冲区，标题单独成块
- 遇 `table` 块：单独成块（不与文字混合）
- 遇 `text` 块：累积到 800 字符再切
- 效果：保留语义完整性，章节边界清晰

#### 策略 C：层级分块（`hierarchical`）
```
父块：完整章节（~1500字符）
子块：父块内的细粒度分割（~300字符）
```
- 子块用于精确向量匹配，父块内容在检索时附加给 LLM 作为扩展上下文
- 即"Small-to-Big"检索策略，缓解长文档召回粒度不匹配问题

**实际产出**（语义分块）：10,353 个 chunks，平均 526 字符

---

### 3.4 Embedding 模型

两套实现使用不同的 Embedding 方案，形成对比：

#### 原生版：DashScope text-embedding-v3（API）

| 参数 | 值 |
|------|----|
| 模型 | `text-embedding-v3` |
| 维度 | 1024（可选 768/512） |
| 批次上限 | **10 条**（API 硬限制） |
| 接口 | OpenAI 兼容接口 |
| 费用 | ~0.0007 元/千 token |

调用方式：
```python
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
resp = client.embeddings.create(
    model="text-embedding-v3",
    input=batch,          # 最多 10 条
    dimensions=1024,
)
```

#### LangChain 版：本地 BAAI/bge-small-zh-v1.5

| 参数 | 值 |
|------|----|
| 模型 | `BAAI/bge-small-zh-v1.5` |
| 维度 | 512 |
| 运行方式 | 本地 CPU 推理（sentence-transformers） |
| 模型大小 | ~90 MB，下载到项目 `models/` 目录 |

选型对比教学价值：两种方案召回结果对同一问题给出相同答案（`14,769,360.50万元`），但BGE 完全离线，text-embedding-v3 按量计费、无需本地 GPU。

---

### 3.5 向量库

**FAISS IndexFlatIP**（精确内积检索）

```
检索时间：10,353 条向量 × 1024 维 → 单次查询 < 1ms
```

选型原因：
- `IndexFlatIP`：暴力内积，精确不近似，数据量 < 10 万时速度足够
- 向量已 L2 归一化，内积等价于余弦相似度（`cos(q, d) = q·d`）
- 持久化为单文件（`faiss_index.bin`），无需服务进程

扩展路径：数据量 > 50 万时可换 `IndexIVFFlat`（需 `train`）或 `IndexHNSWFlat`（无需 train）。

**文件产出**：
- `vectorstore/faiss_index.bin`：原生版索引，41 MB（10,353 × 1024 × 4 bytes）
- `vectorstore/faiss_meta.json`：元数据，15 MB（含每条 chunk 的文本和溯源信息）
- `vectorstore/faiss_lc/`：LangChain 版索引，21 MB（BGE 维度 512）

---

### 3.6 检索策略

原生版实现三路检索并融合：

#### 向量检索（Vector）
- 查询文本 → DashScope API → 查询向量 → FAISS 内积搜索
- 优势：语义相似，理解同义词和近义表达

#### BM25 关键词检索
- 分词：`jieba`（中文分词，默认词典）
- 打分：`BM25Okapi`（`rank_bm25` 库）
- 优势：精确匹配数字、股票代码、专有名词（"营业收入"、"600519"）

#### RRF 混合融合（Reciprocal Rank Fusion）
```
score(d) = Σ 1 / (k + rank_i(d))    k = 60（经验值）
```
- 将向量排名和 BM25 排名合并为统一分数
- 互补两路检索的盲区：向量检索对精确数字弱，BM25 对语义理解弱

#### CrossEncoder Rerank（可选）
- 模型：`BAAI/bge-reranker-base`（本地，278 MB）
- 对 RRF 融合后的候选集（top-10→top-20）精排，输出 top-4 给 LLM
- 原理：双向 attention 对 (query, doc) 对打分，比 bi-encoder 更准确但不可向量化
- 若未下载模型则自动降级为直接截断

---

### 3.7 LLM 生成

**DashScope qwen-plus**（两套实现统一）

| 参数 | 值 |
|------|----|
| 模型 | `qwen-plus` |
| 接口 | OpenAI 兼容，`/v1/chat/completions` |
| Temperature | 0.1（低随机性，保证数字准确性） |
| 上下文窗口 | 放入 top-4 个 chunk（平均约 2000 字） |

系统提示核心约束：
1. 只从参考资料中回答，不编造数据
2. 引用具体数字时标注来源编号（如 `[1]`）
3. 参考资料不足时主动拒绝回答

**可选查询改写**（`--query-rewrite`）：
用 `qwen-turbo` 将模糊问题改写为检索友好形式，例如：
```
原始：茅台最近怎么样
改写：贵州茅台2023年营业收入净利润同比增长率经营情况
```

---

### 3.8 LangChain LCEL 链

LangChain 版的核心是用 LCEL（LangChain Expression Language）的管道操作符组装链：

```python
chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

与原生版的关键差异：
- 无 BM25 路径（单路向量检索）
- 无 Rerank
- 框架自动管理 batch embedding、Document 格式转换
- 代码约 80 行完成同等功能

---

## 四、评估体系

### 4.1 评测题集（20 题）

| 类型 | 题数 | 考察能力 | 示例 |
|------|------|---------|------|
| `simple_fact` | 5 | 基础检索 | 茅台2023年营业收入？ |
| `precise_number` | 5 | BM25数字召回 | 宁德时代2021研发费用占比？ |
| `cross_doc_compare` | 4 | 多文档综合 | 茅台vs五粮液2022毛利率对比 |
| `time_trend` | 3 | 跨年版本整合 | 茅台2021-2023营收趋势 |
| `should_refuse` | 3 | 幻觉控制 | 茅台股价是多少（应拒绝）|

### 4.2 RAGAS 四项指标

| 指标 | 计算方式 | 含义 |
|------|---------|------|
| **Faithfulness** | LLM 判断 | 答案是否完全来自检索内容（无幻觉） |
| **Answer Relevancy** | Embedding 相似度 | 答案是否切题 |
| **Context Precision** | LLM 判断 | 检索内容中有用比例 |
| **Context Recall** | LLM 判断 | 需要的内容被检索到了多少 |

评估中 LLM 打分和 Embedding 均使用 DashScope，无需 OpenAI key。

### 4.3 消融实验矩阵

```
分块策略（3）× 检索方式（3）= 9 种组合
  策略: fixed / semantic / hierarchical
  检索: vector_only / bm25_only / hybrid
```

指标：`Hit Rate @4`（命中率）和 `MRR`（平均倒数排名），不依赖 LLM，运行快。

---

## 五、关键工程决策与踩坑

| 问题 | 根因 | 解法 |
|------|------|------|
| 巨潮 API `stock` 参数返回 0 结果 | API 已不支持股票代码过滤 | 改用 `searchkey` 全文搜索 |
| 五粮液搜索无结果 | cninfo 存储为 "五 粮 液"（字间有空格）| 拆字加空格作为 fallback 关键词 |
| DashScope embedding 报错 batch too large | text-embedding-v3 上限为 10 条，非文档宣传的 25 | `BATCH_SIZE = 10` |
| 开启 BM25 时答案始终"无法找到" | 阈值 0.25 对余弦相似度设计，RRF 分数 ~0.016 量纲完全不同 | 阈值检查始终读 `vec_score` |
| LangChain 1.x 找不到 `langchain.text_splitter` | 该包已拆分为独立包 | `from langchain_text_splitters import ...` |
| 表格内容在 table 和 text 块中重复 | `parse_pdf.py` 收集了 `table_bboxes` 但未用于过滤文字区域 | 已知 Bug，影响 chunk 数量但不影响答案质量 |

---

## 六、目录结构

```
rag_annual_report/
├── src/                         # 原生版（DashScope API）
│   ├── download_reports.py      # 巨潮 API 下载 PDF
│   ├── parse_pdf.py             # PDF → 结构化 JSON blocks
│   ├── chunk_documents.py       # blocks → chunks（三种策略）
│   ├── build_index.py           # chunks → FAISS 向量索引
│   ├── rag_pipeline.py          # 问答流水线（BM25+向量+RRF+Rerank+LLM）
│   ├── serve.py                 # FastAPI HTTP 服务
│   └── static/index.html        # 教学可视化 Web 页面
│
├── src_langchain/               # LangChain 版（本地 BGE + DashScope LLM）
│   ├── download_model.py        # BGE 模型下载到 models/ 目录
│   ├── build_index_lc.py        # PDF → FAISS（LangChain 链路）
│   └── rag_chain_lc.py          # LCEL RAG 链
│
├── evaluation/
│   ├── questions.json           # 20 道标准测试题 + ground truth
│   ├── evaluate.py              # RAGAS 四项指标自动评估
│   └── compare_strategies.py    # 消融实验（策略 × 检索方式）
│
├── data/
│   ├── raw_pdf/                 # 15 份年报 PDF（85 MB）
│   ├── manifest.json            # PDF 元数据索引
│   ├── parsed/                  # 解析后的 JSON（每份 PDF 一个）
│   └── chunks/                  # 分块后的 JSON（all_semantic.json 等）
│
├── vectorstore/
│   ├── faiss_index.bin          # 原生版索引（41 MB）
│   ├── faiss_meta.json          # 元数据（15 MB）
│   └── faiss_lc/                # LangChain 版索引（21 MB）
│
├── models/
│   └── bge-small-zh-v1.5/       # 本地 BGE 模型（~90 MB）
│
├── requirements.txt
├── ARCHITECTURE.md              # 本文档
├── USAGE_GUIDE.md               # 代码调用与测试指南
└── PROJECT_LOG.md               # 开发日志（决策记录与踩坑）
```
