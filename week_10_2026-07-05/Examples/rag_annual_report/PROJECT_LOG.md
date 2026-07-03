# RAG 教学项目记录文档

> 持续更新，记录每个环节的决策、踩坑和补充信息

---

## 项目背景

**目标**：为大模型 RAG 教学课程构建一个接近企业级落地的完整项目，让学生获得真实的工程经验。

**场景选型**：上市公司年度报告问答系统
- 数据来源：巨潮资讯网（cninfo.com.cn，证监会指定披露平台，公开合法）
- 选择理由：原始 PDF 格式复杂（数字+扫描+表格混合）、信息量大、有业务语义、跨文档对比问题丰富

---

## 数据获取（2026-04-16 完成）

### 已下载数据

| 文件名 | 大小 | 备注 |
|--------|------|------|
| 600519_2021_贵州茅台_贵州茅台2021年年度报告.pdf | 3.2M | 上交所，茅台，消费品，排版规范 |
| 600519_2022_贵州茅台_贵州茅台2022年年度报告.pdf | 3.2M | |
| 600519_2023_贵州茅台_贵州茅台2023年年度报告.pdf | 3.4M | |
| 000858_2021_五粮液_2021年年度报告.pdf | 3.4M | 深交所，标题不含公司名 |
| 000858_2022_五粮液_2022年年度报告.pdf | 2.1M | |
| 000858_2023_五粮液_2023年年度报告.pdf | 4.5M | |
| 601318_2021_中国平安_中国平安2021年年度报告.pdf | 7.4M | 上交所，保险业，大量精算表格 |
| 601318_2022_中国平安_中国平安2022年年度报告.pdf | 7.9M | |
| 601318_2023_中国平安_中国平安2023年年度报告.pdf | 6.9M | |
| 300750_2021_宁德时代_2021年年度报告.pdf | 6.1M | 创业板，新能源，近年成长型 |
| 300750_2022_宁德时代_2022年年度报告.pdf | 5.3M | |
| 300750_2023_宁德时代_2023年年度报告.pdf | 5.8M | |
| 002415_2021_海康威视_2021年年度报告.pdf | 6.5M | 深交所，科技，研发信息披露细 |
| 002415_2022_海康威视_2022年年度报告.pdf | 9.4M | |
| 002415_2023_海康威视_2023年年度报告.pdf | 9.7M | |

**总计**：15 份，约 85 MB

### 数据特点（教学价值）

- 文件大小差异大（2.1M ~ 9.7M），PDF 内容密度不同
- 茅台/平安：标题含公司名；宁德/海康/五粮液：标题不含公司名 → 搜索策略要适配
- 中国平安年报含大量嵌套保险精算表格，解析难度最高
- 海康威视年报含研发投入细节拆解，适合技术类问答测试
- 同公司三年数据可测试**时效性冲突**处理

### 踩坑记录

1. **巨潮 API `stock` 参数失效**
   - 现象：传股票代码（如 `600519,sh`）到 `stock` 字段，返回 0 结果
   - 原因：API 已不支持该格式直接过滤，需用 `searchkey` 全文搜索
   - 解法：用 `"公司名+年份+年度报告"` 作为 `searchkey`

2. **"五 粮 液" 字间有空格**
   - 现象：搜索 "五粮液" 关键词返回 0，但实际有数据
   - 原因：深交所部分股票在巨潮数据库的 `secName` 被分词存储为 "五 粮 液"
   - 解法：fallback 策略——将公司名每字拆开加空格重新搜索（`" ".join(list("五粮液"))`）

3. **年报发布时间滞后一年**
   - 2023年年报在 2024年1-4月发布，`seDate` 需设为次年区间

### 脚本位置

- 下载脚本：`src/download_reports.py`
- 原始数据：`data/raw_pdf/`
- 元数据索引：`data/manifest.json`

---

---

## 完整项目实现（2026-04-16 完成）

### 技术选型决策

| 环节 | 原生版 | LangChain 版 |
|------|--------|-------------|
| PDF 解析 | pdfplumber + PyMuPDF + tesseract | langchain PyMuPDFLoader（简化）|
| 分块策略 | 固定/语义/层级三种可切换 | RecursiveCharacterTextSplitter |
| Embedding | **DashScope text-embedding-v3**（API，无需本地模型）| **本地 BAAI/bge-small-zh-v1.5**（下载到 models/）|
| 向量库 | FAISS（手动管理）| LangChain FAISS 封装 |
| 关键词检索 | jieba + rank_bm25 | 无（LangChain 版聚焦框架演示）|
| 融合策略 | RRF（Reciprocal Rank Fusion）| 无 |
| Reranker | CrossEncoder bge-reranker-base（可选）| 无 |
| 查询改写 | qwen-turbo（可选，--query-rewrite 开启）| 无 |
| LLM | **DashScope qwen-plus** | **DashScope qwen-plus** |
| 接口统一性 | OpenAI 兼容接口 | OpenAI 兼容接口 |

**DashScope 接入方式**（两套实现统一）：
```python
from openai import OpenAI
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
```

### 文件结构

```
rag_annual_report/
├── src/                         # 原生实现
│   ├── download_reports.py      ✅ 数据下载（已运行）
│   ├── parse_pdf.py             ✅ PDF 解析（待运行）
│   ├── chunk_documents.py       ✅ 文档分块（待运行）
│   ├── build_index.py           ✅ FAISS 索引（DashScope embedding）
│   ├── rag_pipeline.py          ✅ 问答流水线（BM25+混合+Rerank+QueryRewrite）
│   ├── serve.py                 ✅ FastAPI HTTP 服务
│   └── static/index.html        ✅ 教学可视化 Web 页面
├── src_langchain/               # LangChain 实现
│   ├── download_model.py        ✅ BGE 模型下载到 models/ 目录
│   ├── build_index_lc.py        ✅ FAISS 索引（本地 BGE embedding）
│   └── rag_chain_lc.py          ✅ LCEL RAG 链
├── evaluation/                  # 评估体系
│   ├── questions.json           ✅ 20 道标准测试题 + ground truth
│   ├── evaluate.py              ✅ RAGAS 四项指标自动评估
│   └── compare_strategies.py   ✅ 消融实验（分块策略 × 检索方式）
├── models/                      # 本地模型目录（download_model.py 后生成）
├── data/raw_pdf/                ✅ 15 份年报 PDF（已下载）
├── vectorstore/                 # 索引目录（运行后生成）
├── requirements.txt             ✅ 含所有依赖
└── PROJECT_LOG.md               本文档
```

### 运行顺序

```bash
# 0. 环境准备
pip install -r requirements.txt
set DASHSCOPE_API_KEY=sk-xxx          # Windows
# export DASHSCOPE_API_KEY=sk-xxx     # Linux/Mac

# 原生版流程
python src/parse_pdf.py            # 解析 PDF → data/parsed/
python src/chunk_documents.py      # 分块 → data/chunks/
python src/build_index.py          # 建向量索引（调用 DashScope）
python src/rag_pipeline.py         # 交互式问答
# 单次查询：python src/rag_pipeline.py --query "茅台2023年营收"
# 过滤公司：python src/rag_pipeline.py --query "..." --stock 600519
# 查询改写：python src/rag_pipeline.py --query "..." --query-rewrite

# LangChain 版流程
python src_langchain/download_model.py   # 下载 BGE 到 models/（约90MB）
python src_langchain/build_index_lc.py
python src_langchain/rag_chain_lc.py  # 交互式

# 评估
python evaluation/evaluate.py --pipeline native
python evaluation/evaluate.py --pipeline langchain
python evaluation/evaluate.py --pipeline both
python evaluation/compare_strategies.py
```

### 原生版 vs LangChain 版核心对比

| 维度 | 原生版 | LangChain 版 |
|------|--------|-------------|
| 代码量 | ~700 行（04+05）| ~200 行（01+02）|
| 检索质量 | 高（混合+Rerank）| 中（单路向量）|
| 可控性 | 高（每步可审查）| 中（框架封装）|
| 扩展性 | 需手动写 | 框架提供 Agent/Memory/Tool |
| 适合场景 | 生产级定制 | 快速原型/标准需求 |

### 评测题设计（20 题）

| 类型 | 题数 | 考察重点 |
|------|------|---------|
| 简单事实 | 5 | 基础检索能力 |
| 精确数字 | 5 | BM25 数字召回能力 |
| 跨文档对比 | 4 | 多文档综合能力 |
| 时间趋势 | 3 | 多版本文档整合 |
| 应拒绝回答 | 3 | 幻觉控制能力 |

### 待改进项（已知 Bug）

1. `parse_pdf.py`：`table_bboxes` 收集了但未用于过滤文字提取区域，导致表格内容在 `table` 块和 `text` 块中重复出现。对 RAG 效果影响较小，但会增加 chunk 数量。
2. 消融实验需要手动分别运行三种策略的 03+04，后续可增加批量脚本。

---

## 运行验证（2026-04-16 完成）

### 运行结果

| 步骤 | 状态 | 输出 |
|------|------|------|
| parse_pdf.py | ✅ | 15 个 JSON，茅台~1558块，平安~8000块 |
| chunk_documents.py | ✅ | 10,353 个 chunks，均长 526 字符 |
| build_index.py | ✅ | faiss_index.bin=41MB，faiss_meta.json=15MB |
| rag_pipeline.py（原生版）| ✅ | 正常返回答案，含来源引用 |
| build_index_lc.py（LangChain）| ✅ | vectorstore/faiss_lc/，~21MB |
| rag_chain_lc.py（LangChain）| ✅ | 正常返回答案 |

**验证问题**：贵州茅台2023年营业收入是多少？

原生版答案：
> 贵州茅台2023年营业收入为人民币14,769,360.50万元[2]
> 来源：600519 2023年报 · 第十节 · 第56页

LangChain版答案：
> 贵州茅台2023年营业收入为人民币14,769,360.50万元（即约1476.94亿元）
> 来源：贵州茅台2023年年度报告，第56页

### 调试期 Bug 修复

1. **DashScope embedding 批次限制**（`build_index.py`）
   - 现象：`batch size is invalid, it should not be larger than 10`
   - 原因：代码写了 `BATCH_SIZE = 25`，但 text-embedding-v3 实际上限为 10
   - 修复：`BATCH_SIZE = 10`，日志进度改为每 100 批打印一次（共 ~1036 批）

2. **相关性阈值量纲错误**（`rag_pipeline.py`）
   - 现象：开启 BM25（默认）时答案始终返回"无法找到相关内容"
   - 原因：阈值 `SCORE_THRESHOLD=0.25` 设计给余弦相似度（0~1），但代码拿 `rrf_score`（量纲为 1/(60+rank) ≈ 0.016）来判断
   - 修复：阈值检查改为优先读 `vec_score`，RRF 分数不再用于阈值判断

3. **LangChain text splitter 路径变更**（`src_langchain/build_index_lc.py`）
   - 现象：`ModuleNotFoundError: No module named 'langchain.text_splitter'`
   - 原因：LangChain 1.x 将 TextSplitter 迁移到独立包 `langchain-text-splitters`
   - 修复：`from langchain_text_splitters import RecursiveCharacterTextSplitter`

---

*最后更新：2026-04-16*
