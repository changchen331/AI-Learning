# USAGE_GUIDE.md — 代码调用与测试指南

## 1. 环境准备

### 1.1 依赖安装

```bash
cd graphrag_financial_report
pip install -r requirements.txt
```

### 1.2 API Key 配置

```bash
export DEEPSEEK_API_KEY="sk-xxx"     # LLM 对话（DeepSeek）
export DASHSCOPE_API_KEY="sk-xxx"    # Embedding（阿里云）
```

### 1.3 Neo4j 启动（本地 4.4.8，认证已关闭）

```bash
# 需 JDK11+，本机用 jdk-14
export JAVA_HOME="C:/Program Files/Java/jdk-14"
export PATH="$JAVA_HOME/bin:$PATH"
"D:/neo4j-community-4.4.8/bin/neo4j.bat" console
# 看到 INFO Started. 即就绪，bolt 7687 / http 7474
```

> 本地教学环境在 `conf/neo4j.conf` 设了 `dbms.security.auth_enabled=false`，driver 用 `auth=None`。正式环境用环境变量
> `NEO4J_URI` / `NEO4J_PASSWORD` 切回带认证。

### 1.4 数据

年报已解析的 blocks 复用 `../rag_annual_report/data/parsed/*.json`（15 份，5 家×3 年），无需重新下载解析。

## 2. 各步骤流程

### Step 1：抽取三元组

```bash
# 快速演示（1 份报告前 5 块，约 12s）
python src/extract_triples.py --max-reports 1 --max-blocks-per-report 5

# 全量（15 份×25 块，约 11 分钟）
python src/extract_triples.py --max-blocks-per-report 25
```

内部流程：读 parsed blocks → 按 KEY_SECTION_KEYWORDS 筛关键章节 → 每块调 DeepSeek 结构化输出抽 (s,p,o) → 写
`outputs/triples.jsonl` + `extract_stats.json`。 预期输出：`1153 条三元组 / 375 块 / 15 份报告`。

### Step 2：建图

```bash
python src/build_graph.py --clear
```

内部流程：读 triples.jsonl → 实体规范化（别名表，茅台/贵州茅台酒股份有限公司→贵州茅台）→ Neo4j MERGE 节点 (唯一约束
uid=type:name)+边 (year/role/ratio 属性)。 预期输出：`649 节点 / 681 边`，打印各类型分布。

### Step 3：社区检测

```bash
python src/community_detect.py
```

内部流程：Neo4j 读图 → networkx/igraph 转换 → Leiden (`Optimiser().optimise_partition`) → community id 写回节点 → 每社区
top-15 代表实体 LLM 生成摘要 → `outputs/communities.json`。 预期输出：`13 个社区，最大 132 节点`。

### Step 4：检索（CLI 自测）

```bash
python src/retrieve.py
```

跑两个内置问题验证 Local + Global：Local 查"宁德时代子公司"、Global 查主题问题。

### Step 5：HTTP 服务 + Web 调试页

```bash
uvicorn src.serve:app --host 0.0.0.0 --port 8000
# 浏览器开 http://localhost:8000
```

- `GET /health` → 图库统计
- `POST /query` `{question, mode}` → 标准答案
- `POST /query/debug` → 逐步返回 entity_linking / subgraph / context_build / final_context / answer
- Web 页三栏：输入区 + 步骤卡片（上下文默认展开）+ 答案。子图步骤画 force 图（可拖拽）。

### Step 6：对比评估

```bash
python src/eval_compare.py
```

15 题各跑 GraphRAG Local + 向量 RAG（复用 rag_annual_report），输出答案命中/上下文命中/延迟对比表 +
`outputs/eval_compare.json`。

## 3. 作为模块调用

```python
import sys; sys.path.insert(0, "src")
from retrieve import GraphRAGLocal, GraphRAGGlobal

local = GraphRAGLocal()                          # 加载 neo4j + 节点名向量
r = local.search("宁德时代有哪些子公司")          # debug=False
print(r["answer"])
print([e["name"] for e in r["linked_entities"]]) # 链接到的实体
print(len(r["subgraph"]["nodes"]), "节点")       # 子图规模

global_rag = GraphRAGGlobal()                    # 需先跑过 community_detect
r = global_rag.search("海康威视的业务布局概况")
print(r["answer"])
```

## 4. 调试与常见问题

**Q: `neo4j.exceptions.ServiceUnavailable`？**
A: Neo4j 没启动，按 Step 1.3 起。确认 `netstat | grep 7687` 有 LISTENING。

**Q: `The client is unauthorized`？**
A: 还在用带认证连接。本地已关认证，确认 `neo4j.conf` 第 27 行 `dbms.security.auth_enabled=false`（非 `#` 注释），driver 用
`auth=None`。

**Q: `ModularityVertexPartition.__init__() missing 'graph'`？**
A: leidenalg 0.12 + igraph 1.0 的 `find_partition` bug。本项目已用 `ModularityVertexPartition(H)` +
`Optimiser().optimise_partition(part)` 显式路径绕开，不要改回 `find_partition`。

**Q: Local 搜不到相关实体、子图为空？**
A: 先确认图已建（`python src/build_graph.py --clear`），再确认社区检测跑过（Global 需要 `communities.json`）。

**Q: 抽取 JSON 解析失败多？**
A: `max_tokens` 已设 2048。若仍偶发（dense 表格块），属正常 LLM 输出不稳定，被跳过不影响整体，本身就是教学点（对比 BERT+CRF
的稳定输出）。

**Q: 想换更强 LLM？**
A: 改 `src/llm_client.py` 的 `DEEPSEEK_MODEL`，或加新提供商的 `base_url` + 环境变量，代码结构不变。
