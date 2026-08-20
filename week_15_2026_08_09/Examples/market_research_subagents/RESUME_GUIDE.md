# RESUME_GUIDE.md — 求职简历指导

## 1. 可量化数据

| 指标             | 数值                                                   | 简历可用 |
|------------------|--------------------------------------------------------|----------|
| 主 agent 工具    | 2 个（web_search + dispatch_subagents）                | ✅       |
| 并行加速         | dispatch 平均 2.51×（2.32×/2.71×）                     | ✅       |
| 并行 vs 串行墙钟 | 32.98s vs 50.47s（2 题平均）                           | ✅       |
| subagent 数      | 主 agent 自主决定（3 个/调研）                         | ✅       |
| ReAct trace 粒度 | 主+各 subagent 每步 Thought/Action/Observation         | ✅       |
| 拓扑可视化       | 左动态拓扑 + 右点节点切 trace                          | ✅       |
| 技术栈           | 手写 ReAct / DeepSeek / Tavily(urllib) / FastAPI / SSE | ✅       |
| 依赖数           | 仅 openai + fastapi + uvicorn                          | ✅       |

## 2. 项目名称怎么写

✅ **好**：市场调研 Subagent 并行调研系统（主 agent 自主派发 + 左拓扑右 trace 可视化） ❌ **差**：搜索 agent / 多 agent demo

## 3. 按岗位方向写法

### 算法工程师（Agent / 编排方向）

> **市场调研 Subagent 并行调研系统** ｜ 手写 ReAct · DeepSeek · Tavily · FastAPI
> 设计并实现主 agent 自主决策派发 subagent 的并行调研系统：主 agent 是 ReAct 循环，持 web_search + dispatch_subagents
> 两工具，按 query 自主路由——单事实直接搜、多侧面调研派发 N 个 subagent 并行收集。subagent 也是 ReAct 循环（只 web_search），用
> ThreadPoolExecutor 并行执行，wall-clock 从 sum 压到 ≈max。完整 trace 捕获（每步 Thought/Action/Observation）。A/B 对比：3
> subagent 并行 vs 串行，dispatch 平均加速 2.51×。解决 ReAct LLM 不按格式输出（空 action）、长 observation 撑爆 context、主
> agent 不派发（system prompt 未注入）等工程问题。

### 后端工程师（AI 应用工程方向）

> **并行调研服务 + 可视化** ｜ FastAPI · SSE · vanillar JS
> 基于 FastAPI 构建调研服务，SSE 流式推送主 agent + 各 subagent 的节点级事件（main_step/dispatch/subagent_step/done），前端左侧
> SVG 拓扑随派发动态加节点、实时高亮，右侧点任意节点切换查看其 ReAct 全过程。用 queue 桥接 ThreadPoolExecutor 调研线程与
> StreamingResponse 主线程的跨线程流式。可视化代码与主流程隔离（独立 vanilla JS，零依赖）。Tavily 用标准库 urllib 调用，无 SDK
> 依赖。

## 4. 按经验层级写法

### 应届

> 课程项目：实现主 agent + subagent 并行调研，主 agent 持 web_search + dispatch 两工具自主决策派发，3 subagent 用
> ThreadPool 并行，对比串行加速 2.51×。理解 ReAct 循环、并行墙钟 vs 串行 sum、Amdahl 定律对总加速比的限制。

### 1~3 年

> 主导市场调研 subagent 系统设计：手写通用 ReAct 引擎（主/subagent 共用，区别在工具集），dispatch_subagents 用
> ThreadPoolExecutor 并行 N 个 subagent，A/B 量化 dispatch 加速 2.51×。解决 ReAct 格式解析兜底、长 observation
> 截短、system_prompt 注入等 6 条工程坑。FastAPI + SSE 节点级事件流 + 左拓扑右 trace 可视化。

### 3 年以上

> 设计自主决策的并行调研架构，主 agent ReAct 自主路由（web_search vs dispatch_subagents），subagent 并行执行 + trace
> 全程捕获。量化并行收益与 Amdahl 瓶颈：dispatch 加速 2.51×，总墙钟受主 agent 串行段限制（已识别规划/综合异步化为优化方向）。落地
> SSE 流式 + 拓扑/trace 双面板可视化，可迁移至竞品分析、行业研究等需多源并行采集的场景。

## 5. 好句 vs 差句

❌ 做了个多 agent 搜索系统，比单个快。 ✅ 主 agent 持 web_search + dispatch_subagents 两工具自主路由，3 subagent 用
ThreadPoolExecutor 并行，A/B 对比串行 dispatch 加速 2.51×（32.98s vs 50.47s），完整 ReAct trace 节点级可观测。

❌ 实现了并行和可视化。 ✅ SSE 流式推送主+各 subagent 节点级事件，左侧 SVG 拓扑随派发动态加节点、右侧点节点切换看 ReAct
全过程；可视化代码隔离在独立 vanilla JS（零依赖）。

❌ 主 agent 能决定是否派发。 ✅ 主 agent 是 ReAct 循环，按 query 自主路由——单事实走 web_search、多侧面调研走
dispatch_subagents，LLM 通过 Thought 推理决策而非硬编码规则，system prompt + worked example 引导稳定派发。

## 6. 面试常见问题

**Q: 主 agent 怎么决定派不派 subagent？**
A: 主 agent 自己是 ReAct 循环，system prompt 给决策原则（2+ 侧面必须 dispatch）+ worked example 教格式。LLM 在 Thought 里推理后选
Action: web_search 或 dispatch_subagents。不是硬编码规则，是 LLM 自主路由。

**Q: 并行加速 2.51× 怎么来的？**
A: dispatch_subagents 用 ThreadPoolExecutor 并行 N 个 subagent，wall_clock ≈ max (子时长)，serial_sum = sum (子时长)。A/B 用
serial=True（for 循环顺序跑）做真实基线，对比得 2.51×。

**Q: 为什么总墙钟加速只有 1.4×，比 dispatch 2.51× 小？**
A: Amdahl 定律。总墙钟 = 主 agent 串行段（规划+综合的 LLM 调用）+ dispatch
并行段。只有并行段加速，串行段不变，拉低总加速比。优化方向：规划/综合也异步，或用便宜快模型降串行段占比。

**Q: ReAct 的 stop=["Observation:"] 是干啥的？**
A: 让 LLM 生成到 Action Input 后停下（不自己编 Observation），runner 执行工具得真实 Observation 续写。这是 ReAct 经典实现，保证
Observation 是真实工具结果而非幻觉。

**Q: 主 agent 和 subagent 用同一个 ReAct 引擎，区别在哪？**
A: 只在 `tools` 字典。主 agent 有 {web_search, dispatch_subagents}，subagent 只有 {web_search}。同一 ReActLoop
类，不同工具集——体现「agent 的能力由工具集定义」。

**Q: 为什么不用 LangGraph/AutoGen？**
A: 教学优先。手写 ReAct + ThreadPool 让学生看清 ReAct 循环、并行调度、trace 捕获的原理，零框架依赖。生产可换 LangGraph（PPT
6.2 也提到它早已实践），但手写更透明。

**Q: Tavily 为什么不用 SDK？**
A: 少依赖原则。Tavily REST API 用标准库 urllib 直接调，返回 JSON 解析即可，省一个 SDK 依赖。失败降级返回 error 字符串让
ReAct 兜底。
