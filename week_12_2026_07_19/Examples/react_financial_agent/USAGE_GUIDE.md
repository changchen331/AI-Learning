# USAGE_GUIDE.md — 代码调用与测试指南

## 1. 环境准备

### 安装依赖

```bash
pip install openai faiss-cpu akshare fastapi uvicorn numpy
```

### 配置 API Key

```bash
export DASHSCOPE_API_KEY="sk-xxx"       # 必填，用于 LLM 推理和 RAG Embedding
export AGENT_MODEL="qwen-max"           # 可选，默认 qwen-max，可换 deepseek-v3 等
```

Windows PowerShell：

```powershell
$env:DASHSCOPE_API_KEY = "sk-xxx"
$env:AGENT_MODEL = "qwen-max"
```

---

## 2. 各脚本使用说明

### 2.1 agent.py — 统一命令行入口

```bash
cd react_financial_agent/src

# 手写 Prompt 解析版（默认）
python agent.py --mode manual --question "茅台和五粮液2023年毛利率差多少？"

# Function Calling 版
python agent.py --mode fc --question "宁德时代2023年营业收入是多少？"

# 调整最大步数（默认10）
python agent.py --mode manual --question "..." --max_steps 8
```

**预期输出**（手写版）：

```
============================================================
问题: 茅台和五粮液2023年毛利率差多少？
模型: qwen-max  实现: 手写Prompt解析
============================================================

[Step 1]
🧠 Thought: 需要先获取两家公司的股票代码...
🔧 Action:  company_lookup
   Input:   {"name": "贵州茅台"}
👁  Obs:     贵州茅台 的股票代码为 600519
...
✅ Final Answer:
茅台2023年毛利率91.96%，高于五粮液的75.79%，差16.17个百分点。
```

---

### 2.2 react_manual.py — 手写版独立运行

```bash
python react_manual.py
python react_manual.py --question "贵州茅台过去一年股价涨跌幅是多少？"
```

**内部流程**：

1. 构造含工具描述的 System Prompt
2. 调用 LLM，`stop=["Observation:"]` 让模型在工具调用前停止
3. 正则解析 Thought / Action / Action Input
4. 执行工具，获取 Observation
5. 将 Observation 追加到对话历史，继续下一步
6. 检测到 `Final Answer:` 时终止

---

### 2.3 react_function_calling.py — Function Calling 版独立运行

```bash
python react_function_calling.py
python react_function_calling.py --question "海康威视2023年年报的主要风险有哪些？"
```

**与手写版的关键区别**：

- 工具通过 JSON Schema（`TOOLS_SCHEMA`）注册给模型
- 模型通过 `tool_calls` 字段返回工具调用请求，格式由 API 保证
- Thought 过程在模型内部，CLI/UI 中显示为灰色提示

---

### 2.4 evaluate.py — 对比评估

```bash
python evaluate.py
python evaluate.py --output ../evaluation/compare_result.json
```

**评估问题集**（5题）：

- Q1: 跨公司财务对比（茅台 vs 五粮液毛利率）
- Q2: 跨年度趋势 + CAGR 计算（宁德时代营收）
- Q3: 单次年报检索（海康威视风险因素）
- Q4: 股价查询 + 涨跌幅计算（茅台2023全年）
- Q5: 边界拒绝（预测A股走势，应拒绝）

**输出示例**：

```
ID   Mode     Steps   Time(s)   Success  ParseErr
──────────────────────────────────────────────────
Q1   manual   5       68.0      True     0
Q1   fc       5       55.0      True     0
Q2   manual   7       95.0      True     0
Q2   fc       6       80.0      True     0
...
[manual] 平均步数:4.6  平均耗时:72.0s  成功率:80%  解析错误总数:0
[fc]     平均步数:4.4  平均耗时:60.0s  成功率:80%  解析错误总数:0
```

---

### 2.5 serve.py — Web 服务启动

```bash
cd react_financial_agent/src
uvicorn serve:app --host 0.0.0.0 --port 8000
```

启动后访问：http://localhost:8000

**接口列表**：

- `GET  /health` — 健康检查
- `POST /query/manual` — 手写版，流式 SSE 返回每步
- `POST /query/fc` — Function Calling 版，流式 SSE 返回每步

**SSE 事件格式**：

```json
{
  "type": "start",
  "question": "...",
  "mode": "manual"
}
{
  "type": "action",
  "step": 1,
  "thought": "...",
  "action": "company_lookup",
  "action_input": {
    ...
  },
  "observation": "..."
}
{
  "type": "final",
  "step": 6,
  "thought": "...",
  "answer": "..."
}
{
  "type": "done"
}
```

---

## 3. 作为模块调用

```python
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# 添加 src 到路径
import sys

sys.path.insert(0, "react_financial_agent/src")

# 手写版
from react_manual import run as manual_run

for step in manual_run("茅台2023年毛利率是多少？", max_steps=10):
    if step["type"] == "action":
        print(f"Step {step['step']}: {step['action']}({step['action_input']})")
        print(f"  => {step['observation'][:100]}")
    elif step["type"] == "final":
        print(f"Final: {step['answer']}")

# Function Calling 版（接口完全一致）
from react_function_calling import run as fc_run

for step in fc_run("宁德时代2023年营收？"):
    ...
```

---

## 4. 工具单独调用

```python
from tools import TOOLS_MAP

# 公司名转代码
print(TOOLS_MAP["company_lookup"](name="贵州茅台"))
# => 贵州茅台 的股票代码为 600519

# 计算器
print(TOOLS_MAP["calculator"](expr="91.96 - 75.79"))
# => 16.17

# 年报检索
print(TOOLS_MAP["rag_search"](query="茅台2023年毛利率", top_k=3))
# => [1] 来源：600519 2023年报 第X页 ...

# 财务指标
print(TOOLS_MAP["financial_indicator"](symbol="600519"))
# => 毛利率: 2025年: 91.18 | 2024年: 91.93 | 2023年: 91.96 ...

# 历史股价
print(TOOLS_MAP["stock_price"](symbol="600519", start_date="20230101", end_date="20231231"))
# => 区间涨跌幅: +2.60%
```

---

## 5. 调试与常见问题

**Q: `rag_search` 报错 `assert d == self.d`**
A: Embedding 维度不匹配。确认 `DASHSCOPE_API_KEY` 已设置，`rag_search` 使用 DashScope API 编码查询，与建索引时保持 1024
维一致。

**Q: `financial_indicator` 速度很慢（20秒以上）**
A: 正常，AkShare `stock_financial_abstract` 需拉取历史财务数据。第一次调用较慢，Agent 循环中如多次调用同一公司建议加缓存（可选优化）。

**Q: Web UI 显示"请求失败"**
A: 检查 `uvicorn` 是否正常启动，访问 `http://localhost:8000/health` 确认服务状态。

**Q: 手写版 Thought 为空**
A: 正常现象。qwen-max 在部分中间步骤会省略 Thought 直接输出 Action，解析器容忍此情况，不影响工具执行。

**Q: 想用 DeepSeek-V3**
A: 修改环境变量：

```bash
export DASHSCOPE_API_KEY="sk-xxx"   # DashScope key 同样可用于 DeepSeek
export AGENT_MODEL="deepseek-v3"
```

或者改用 DeepSeek 官方接口，修改 `tools.py` 和 `react_manual.py` 中的 `base_url`：

```python
base_url = "https://api.deepseek.com/v1"
```
