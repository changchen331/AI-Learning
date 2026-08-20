# ARCHITECTURE.md — 市场调研 Subagent 并行调研系统

## 1. 项目定位

**场景**：用户提一个市场调研问题，主 agent 自主决定是否派发多个 subagent 并行联网收集不同侧面的资料，聚合成结构化调研报告。直接落地「subagent
并行」范式，凸显并行加速优势。

**核心设计**（对应你的需求）：

- 主 agent 自己是 ReAct 循环， **有 2 个工具**：`web_search`（单次搜索，简单问题用）和 `dispatch_subagents`（派发多个 subagent
  并行调研，多侧面问题用）。主 agent **根据 query 自主决定**用哪个——不是固定拓扑，是 LLM 自主路由。
- subagent 也是 ReAct 循环（只有 `web_search`），多个 subagent 用 `ThreadPoolExecutor` 并行执行 → 并行优势。
- 可视化： **左侧拓扑图**（主节点 + 派发时动态加入的 subagent 节点）， **右侧点任意节点看其 ReAct 过程**
  （Thought/Action/Observation），可不断切换。

**范式归属**：动态 Orchestrator-Workers（PPT 6.3）——主 agent 决定派几个、派什么，拓扑在运行时生长。

## 2. 整体流水线

```
用户问题
   ↓
主 agent ReAct 循环（工具: web_search + dispatch_subagents）
   ├─ 简单事实 → 直接 web_search → Final Answer
   └─ 多侧面调研 → dispatch_subagents("课题1|课题2|课题3")
                       ↓
              ┌─ subagent1 ReAct(web_search) ─┐
              ├─ subagent2 ReAct(web_search) ─┤ 并行(ThreadPool)
              └─ subagent3 ReAct(web_search) ─┘
                       ↓ 汇总（含并行加速统计）
              主 agent 综合成报告 → Final Answer
```

脚本对应：`tavily_search.py`(搜索) → `react_loop.py`(通用 ReAct) → `agents.py`(主agent+派发) → `serve.py`(SSE) →
`eval_compare.py`(A/B)。

## 3. 各环节技术选型

### 3.1 联网搜索：Tavily（零 SDK 依赖）

用标准库 `urllib` 直接调 Tavily REST API，不引 `tavily-python` SDK（CLAUDE.md 少依赖原则）。Tavily 为 LLM 优化：返回 `answer`
(摘要) + `results`(带 url/content 的来源)。失败降级返回错误字符串（ReAct 兜底）。

### 3.2 通用 ReAct 引擎（react_loop.py）

主 agent 和 subagent 共用同一个 `ReActLoop` 类，区别只在 `tools` 字典：

- 主 agent：`{web_search, dispatch_subagents}`
- subagent：`{web_search}`

经典 ReAct：LLM 输出 `Thought/Action/Action Input`，用 `stop=["Observation:"]` 在 Action Input 后截断，runner 执行工具得
Observation 续写。 **完整 trace 捕获**（每步 Thought/Action/ActionInput/Observation）存下来供可视化「点节点看 ReAct 过程」。

解析兜底：LLM 拿到长结果后常直接写报告不带 `Final Answer:` 前缀，`_parse` 检测到无 Action 但有实质文本时当作 Final
Answer（避免空 action 死循环）。

### 3.3 主 agent 自主决策（agents.py）

系统提示 `MAIN_SYSTEM` 给出明确决策原则 + worked example：

- 2 个及以上侧面（调研/分析/概况）→ **必须** `dispatch_subagents`
- 单一事实 → 直接 `web_search`

`dispatch_subagents` 工具输入是 `课题1|课题2|课题3`（管道分隔），主 agent 自主拆分。派发后 N 个 subagent 并行，主 agent 收齐汇总
Observation 综合成报告。

### 3.4 并行执行（凸显优势的核心）

`ThreadPoolExecutor(max_workers=N)` 并行跑 N 个 subagent ReAct。量化：`wall_clock`（并行墙钟）vs `serial_sum`（各 subagent
时长之和 = 串行基线）。`serial=True` 模式退化为 for 循环（eval A/B 真实对比基线）。

### 3.5 可视化（static/index.html + viz/topology.js）

- **可视化代码隔离**：SVG 拓扑动画在 `static/viz/topology.js`（vanilla JS 无依赖），主流程 UI 在 index.html。学生关注
  ReAct/编排逻辑，viz 仅辅助。
- **深色科技主题**：渐变背景、玻璃卡片 (backdrop-blur)、霓虹标题 (渐变文字)、发光节点 (SVG filter glow)、运行节点脉冲
  (半径周期变化)、流光虚线边、monospace observation。
- 左侧拓扑：主节点先画，`dispatch` 事件到达时动态加 subagent 节点 + 主→子边，节点按 `subagent_step` 实时脉冲、`done` 变绿。
- 右侧过程流（核心 UX）： **默认"全部实时流"**——所有节点 (main + 各 subagent)的每一步按到达顺序实时滚动展示，带节点
  badge，subagent 运行过程全程可见。点左侧节点按钮 → 只看该节点；点"全部实时流"回到全部。`autoFollow` 自动滚到底。
- 切换问题：`TopoViz` 构造时 `host.innerHTML=''` 整体换图，不堆叠。

## 4. 实验结果（真实跑出）

### 4.1 端到端调研

调研「2024 中国新能源汽车市场」：主 agent 2 步（`dispatch_subagents` → `Final Answer`），派发 3 个 subagent（销量规模/竞争格局/政策趋势），各
subagent ReAct 2~5 步。报告分维度组织带来源。

### 4.2 Parallel vs Serial A/B（2 题，真实数字）

| 问题           | 并行墙钟   | 串行墙钟   | dispatch 加速 |
|----------------|------------|------------|---------------|
| 新能源汽车调研 | 35.88s     | 44.78s     | 2.32×         |
| 咖啡市场调研   | 30.07s     | 56.16s     | 2.71×         |
| **平均**       | **32.98s** | **50.47s** | **2.51×**     |

**结果解读**：

- dispatch 加速 **2.51×**：3 个独立子任务并行，墙钟从 sum 压到 ≈max。
- 总墙钟加速（1.4×）小于 dispatch 加速（2.51×）： **主 agent 自身的串行开销**（规划 + 综合的 LLM
  调用）不并行化，拉低了总加速比——这是诚实的教学点：并行收益只在可并行的子任务部分，串行编排段是瓶颈（Amdahl 定律）。
- subagent 数由主 agent 自主决定（本例都拆 3 个），非硬编码。

### 4.3 与 PPT 6.3/6.4 对应

- 拓扑：动态 Orchestrator-Workers（主 agent 派发，节点运行时生长）
- 用图理由（6.4）：多异构节点协作 ✓、可并行分支 ✓、需独立验证 ✓
- 并行 vs 顺序：本项目的 serial 基线正是 6.4「顺序任务」对照，量化并行收益

## 5. 优化方向

| 层面     | 方向                                                                |
|----------|---------------------------------------------------------------------|
| 并行收益 | 主 agent 规划/综合也异步化，或用更便宜快模型做规划降串行段占比      |
| subagent | 加 max_results 调参、失败重试、子任务结果去重                       |
| 决策     | 主 agent 决策不稳定时加规则兜底（query 含"调研/分析"强制 dispatch） |
| 可视化   | 拓扑加边动画（dispatch 时主→子流光）、trace 自动滚动跟随            |
| 工程     | subagent 数上限保护、Tavily 限流、trace 持久化回放                  |

## 6. 关键工程决策与踩坑

| 问题                                      | 根因                                                | 解法                                                                       |
|-------------------------------------------|-----------------------------------------------------|----------------------------------------------------------------------------|
| 主 agent 不派发、自己串行 web_search 8 次 | `MAIN_SYSTEM` 定义了但没传给 ReActLoop              | ReActLoop 加 `system_prompt` 参数，主 agent 传 `MAIN_SYSTEM`               |
| prompt 光说"必须 dispatch"无效            | ReAct 需 worked example 教格式                      | MAIN_SYSTEM 加 `dispatch_subagents` 的完整示例                             |
| 拿到长结果后输出空 action 撞 max_steps    | LLM 直接写报告不带 `Final Answer:` 前缀，正则没匹配 | `_parse` 兜底：无 Action 但有文本 → 当 Final Answer                        |
| 重复派发（dispatch 两次）                 | 第一步空 action 后误判重派                          | 同上兜底解决，主 agent 现 2 步收尾                                         |
| dispatch observation 过长撑爆 context     | 3 个 subagent 全文回灌                              | 每个子结果截短到 500 字喂回主 agent（完整 trace 仍在 shared_state 供 viz） |
| SSE 跨线程                                | run_research 在线程跑，StreamingResponse 在主线程   | queue 桥接：回调 push 队列，SSE 主循环 get+yield                           |
| Tavily 网络偶发失败                       | 教学环境                                            | tavily_search 返回 {error} 不抛，ReAct 兜底                                |

## 7. 目录结构

```
market_research_subagents/
├── src/
│   ├── tavily_search.py     # Tavily 搜索（urllib 零依赖）
│   ├── react_loop.py        # 通用 ReAct 引擎（主/subagent 共用）
│   ├── agents.py            # 主 agent + dispatch_subagents 并行派发
│   ├── serve.py             # FastAPI + SSE 流式
│   ├── llm_client.py        # 极简 DeepSeek 客户端
│   └── eval_compare.py     # parallel vs serial A/B
├── static/
│   ├── index.html          # 左拓扑右trace切换 主流程 UI
│   └── viz/topology.js     # SVG 拓扑动画（隔离，非教学重点）
├── outputs/
│   └── eval_compare.json
├── requirements.txt
├── ARCHITECTURE.md
├── USAGE_GUIDE.md
└── RESUME_GUIDE.md
```
