# USAGE_GUIDE.md — 代码调用与测试指南

## 1. 环境准备

### 1.1 依赖安装

```bash
cd market_research_subagents
pip install -r requirements.txt
```

> 仅 openai + fastapi + uvicorn，无 SDK。Tavily 用标准库 urllib 调用，ReAct/拓扑全手写。

### 1.2 API Key

```bash
export DEEPSEEK_API_KEY="sk-xxx"     # 主/subagent 的 LLM 推理
export TAVILY_API_KEY="tvly-xxx"     # 联网搜索
```

## 2. 各步骤流程

### Step 1：CLI 跑一次调研

```bash
python src/agents.py
```

内置自测跑一个调研问题，打印主 agent 动作序列 + subagent 数 + 并行统计。

或直接调 `run_research`：

```python
import sys; sys.path.insert(0, "src")
from agents import run_research
r = run_research("2024年中国新能源汽车市场调研：销量规模、主要厂商竞争格局、政策趋势")
print(r["final_answer"])
print("并行:", r["parallel_stats"])
```

### Step 2：HTTP 服务 + 可视化

```bash
uvicorn src.serve:app --host 0.0.0.0 --port 8002
# 浏览器开 http://localhost:8002
```

- `GET /health` → `{tavily, llm}` 就绪状态
- `POST /query {question}` → SSE 流，逐事件推：
  `start` → `main_step`(主 agent 每步) → `dispatch`(派发，拓扑加节点) → `subagent_step`(各子 agent 步骤) →
  `subagent_done` → `final`(报告+并行统计) → `done`
- Web 页：左侧拓扑（节点随派发动态出现、实时高亮），右侧点节点看其 ReAct 过程（可切换），下方最终报告。

### Step 3：Parallel vs Serial 对比

```bash
python src/eval_compare.py --limit 2
```

4 题（--limit 可缩），每题 parallel (ThreadPool) vs serial (for 循环) 各跑一次，输出墙钟/加速对比表 +
`outputs/eval_compare.json`。

## 3. 作为模块调用

```python
import sys; sys.path.insert(0, "src")
from agents import run_research

# 带 trace 回调（接 SSE / 日志 / 可视化）
def on_main(step): print(f"[main] {step['action']}")
def on_sub(sid, step): print(f"[{sid}] {step['action']}")
def on_dispatch(info): print(f"派发: {info['subtopics']}")
def on_done(sid, dur, topic): print(f"[{sid}] done {dur}s")

r = run_research("中国咖啡市场调研：规模、品牌、趋势",
    on_main_step=on_main, on_subagent_step=on_sub,
    on_subagent_done=on_done, on_dispatch=on_dispatch)

# 单独用 ReAct loop
from react_loop import ReActLoop
from tavily_search import tavily_search, format_search_result
loop = ReActLoop("my", tools={"web_search": (
    lambda q, **_: format_search_result(tavily_search(q)), "联网搜索")},
    max_steps=4)
print(loop.run("2024年比亚迪销量")["final_answer"])
```

## 4. 调试与常见问题

**Q: 主 agent 不派发 subagent，自己串行搜索？**
A: 现已通过 `MAIN_SYSTEM` + worked example 引导，多侧面调研会派发。若仍偶发，确认 `system_prompt=MAIN_SYSTEM` 传给了主
agent 的 ReActLoop（之前踩过此坑）。

**Q: 主 agent 卡在空 action？**
A: 已用 `_parse` 兜底（无 Action 有文本 → 当 Final Answer）。若仍出现，检查 LLM 输出是否被 `stop=["Observation:"]` 误截。

**Q: Tavily 报错 / 无结果？**
A: 确认 `TAVILY_API_KEY` 设置且网络可达 `api.tavily.com`。失败时 `tavily_search` 返回 `{error}`，ReAct 把它当 Observation
继续或换关键词。

**Q: 总墙钟加速比 dispatch 加速小？**
A: 正常。dispatch 加速是子任务并行的纯收益（2.5×），总墙钟包含主 agent 自身串行段（规划+综合的 LLM 调用），不并行化，拉低总加速比（Amdahl
定律）。优化见 ARCHITECTURE 第 5 节。

**Q: 想看某 subagent 的 ReAct 全过程？**
A: Web 页左侧点该节点，右侧切换显示其全部 Thought/Action/Observation 步骤。或读 `r["subagents"][sid]["trace"]`。

**Q: serial 模式怎么跑？**
A: `run_research(question, serial=True)`，subagent 改 for 循环顺序执行（eval 基线）。
