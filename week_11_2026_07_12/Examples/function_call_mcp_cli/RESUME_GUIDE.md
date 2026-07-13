# RESUME_GUIDE.md — 求职简历指导

## 1. 可量化数据

| 指标    | 数值                                                       | 说明                                      |
|-------|----------------------------------------------------------|-----------------------------------------|
| 知识库规模 | 15 份 A 股年报 / 10353 个语义分块                                 | 5 家公司 × 3 年                             |
| 向量索引  | FAISS IndexFlatIP，10353 × 1024 维                         | L2 归一化后内积 = 余弦相似度                       |
| 工具数   | 3 个（search_annual_report / list_companies / get_weather） | 三方式共用同一份后端                              |
| 对比方式数 | 4 种（Function Call / MCP / CLI-named / CLI-bash）          | 同一问题集横向对比                               |
| 单轮延迟  | Function Call ~5s / MCP ~6s / CLI ~6-13s                 | FC 进程内直调最快；MCP/CLI 有子进程或 IPC 开销，随问题规模上升 |
| 幻觉控制  | 4/4 方式正确拒绝不在库的公司（比亚迪）                                    | 部分方式 0 工具调用即拒绝                          |
| 沙箱规则  | 13 条危险模式正则 + 7 个命令头白名单 + 超时 15s                          | CLI(bash) 形态安全兜底                        |
| 代码复用  | 三方式共享 `src/` 后端，零逻辑重复                                    | 公平对比前提                                  |

## 2. 项目名称怎么写

**好写法**：

- 「Function Call / MCP / CLI 三种大模型工具调用方式对比教学项目」
- 「大模型工具调用抽象层对比实践（Function Call vs MCP vs CLI）」

**差写法**：

- 「一个 MCP 项目」（太泛，看不出对比视角）
- 「用 MCP 查年报」（只点名一种方式，丢掉对比卖点）
- 「LLM 工具调用 demo」（没具体度）

## 3. 按岗位方向写法

### 3.1 算法工程师方向

> **大模型工具调用三方式对比实践（Function Call / MCP / CLI）**
> 针对"大模型如何动手操作外部世界"这一主题，设计同业务后端（A股年报 RAG 检索 + 天气查询，FAISS 10353 分块）×
> 三种接入方式的横向对比框架。Function Call 层手写 JSON Schema 注册工具并实现单轮 tool_call 闭环（支持并行多工具）；MCP 层用
> FastMCP 实现自写 Server，Host 端经 stdio JSON-RPC 完成 initialize/list_tools/call_tool 全流程，并将 MCP inputSchema 适配为
> OpenAI tools schema 喂给 LLM；CLI 层提供具名白名单（run_cli）与通用沙箱（run_bash，13 条危险模式正则 + 命令头白名单 +
> 超时）两种形态。自建 compare.py 对 4 类问题 × 4 方式实跑，量化对比接入成本/延迟/跨模型复用性/幻觉控制行为差异，得出"
> 四方式能力一致、Function Call 进程内直调最快而 MCP/CLI 有子进程或 IPC 开销、CLI(bash) 需沙箱兜底"等结论。LLM 默认
> DeepSeek（function calling + 并行调用），Embedding 用 DashScope text-embedding-v3。

### 3.2 后端工程师方向

> **大模型工具调用协议接入与 CLI 沙箱实践**
> 围绕"同一份业务能力如何被大模型安全调用"设计三套接入方案并对比。MCP 方案：用 FastMCP 实现两个 stdio Server（年报 RAG /
> 天气），Host 端用 AsyncExitStack 统一管理多 Server 生命周期，实现 initialize 握手、list_tools 工具发现、call_tool 路由的全链路；并完成
> MCP inputSchema → OpenAI tools schema 的协议层适配。CLI 方案：argparse 统一入口 + pyproject `[project.scripts]` 注册为
> PATH 上的真实命令 `fincli`（`pip install -e .`，与 git/ls 同形态，非 `python xxx.py`）+ 两种触发形态——具名 run_cli（白名单
> enum，host 拼 `fincli` 子命令执行，安全可控）与通用 run_bash（shell=True + 危险命令正则黑名单 + 命令头白名单 + 15s 超时 +
> 工作目录锁定）；未安装 fincli 时 shutil.which 探测自动降级。用 subprocess + JSON 协议实现 compare.py
> 跨方式对比运行器，统一收集四方式工具调用与延迟数据。解决了 MCP tool 函数与后端函数同名递归、stdio 通道 stdout 污染、地名
> geocoding 歧义、pyproject packages 须用点号包名等工程问题。

## 4. 按经验层级写法

### 4.1 应届

> 课程项目：搭建 Function Call / MCP / CLI 三方式对比框架，理解大模型工具调用的三层关系（意图生成层 / 协议接入层 /
> 执行实现层）。实现 MCP Host 连接多 Server、CLI 沙箱拦截危险命令，跑通 4 类问题横向对比。

### 4.2 1~3 年

> 设计并实现大模型工具调用三方式对比教学项目。MCP 侧完成 Host + 自写 Server 全链路（stdio JSON-RPC、AsyncExitStack
> 多连接管理、schema 适配）；CLI 侧设计白名单 + 沙箱双层安全模型。量化对比四方式延迟与幻觉控制行为，沉淀踩坑文档（同名递归、stdout
> 协议污染、geocoding 歧义等）。

### 4.3 3 年以上

> 主导大模型工具调用接入方案选型与对比验证。定义同业务后端 × 三方式对比框架，量化接入成本/延迟/跨模型复用/安全四维度，输出选型建议（快速原型用
> Function Call、多工具生态用 MCP、工程师场景用 CLI+沙箱）。设计 CLI(bash) 沙箱安全模型（黑名单正则 + 白名单 + 资源限制），解决
> MCP 多 Server 生命周期管理与协议层 schema 适配问题。

## 5. 好句 vs 差句对比

| 差句           | 好句                                                                                                              |
|--------------|-----------------------------------------------------------------------------------------------------------------|
| 用 MCP 做了工具调用 | 用 FastMCP 实现 stdio Server，Host 端走 initialize→list_tools→call_tool 全流程，并适配 MCP inputSchema 为 OpenAI tools schema |
| 做了 CLI 工具    | 实现 CLI 两种形态：具名 run_cli（白名单 enum）与通用 run_bash（13 条危险模式正则 + 命令头白名单 + 超时 + cwd 锁定）                                 |
| 对比了三种方式      | 同一业务后端 × 三方式横向对比 4 类问题，量化延迟（FC ~5s / MCP ~6s / CLI ~6-13s）与幻觉控制行为差异                                             |
| 防止了危险命令      | 沙箱拦截 rm/del/format/sudo/curl\|sh/nc 等 13 类模式，命令头白名单仅放行 7 个可执行文件                                                 |
| 解决了递归 bug    | MCP tool 函数与导入后端函数同名遮蔽导致 RecursionError，用 as 别名导入隔离命名空间修复                                                       |

## 6. 面试常见问题

1. **Function Call、MCP、CLI 三者是什么关系？**
   分层协作而非互斥：Function Call 是模型能力层（生成结构化调用意图）；MCP 是协议标准层（统一工具注册/发现/通信）；CLI
   是工具实现层（用命令行执行操作）。CLI 可作为 Function Call 的工具被触发，MCP Server 内部也可调 CLI。

2. **MCP 解决了什么问题？**
   M×N 碎片化：M 个模型 × N 个工具曾要两两对接。MCP 把工具侧封装成 Server、模型侧封装成 Client，各只对接协议一次，降为
   M+N。本项目 rag_server 写一次，可被任意支持 MCP 的 Host 复用。

3. **MCP 的 list_tools 返回的 schema 为什么还要转成 OpenAI tools 格式？**
   MCP 让工具与模型解耦，但具体喂给某个 LLM 时仍要变成该模型 API 认识的格式（OpenAI 的 `function.parameters`）。MCP 的
   `inputSchema` 本身就是 JSON Schema，直接塞进 `parameters` 字段即可。

4. **CLI(bash) 形态有哪些安全风险，怎么防？**
   模型可能生成 `rm -rf`、`curl|sh`、反弹 shell 等。本项目用四层防御：危险模式正则黑名单、命令头白名单（只放行
   python/git/ls/cat/echo 等）、15s 超时、工作目录锁定。形态 A（具名 enum）天然更安全，是优先选择。

5. **为什么三种方式要共享同一份后端？**
   对比公平性：差异应只来自"接入方式"，不能混入业务逻辑差异。共享 `src/rag_backend`
   后，四方式跑同一问题的检索结果完全一致，对比才能隔离出接入成本/延迟/安全差异。

6. **单轮闭环的局限是什么？怎么扩展？**
   单轮：模型一次输出 tool_calls（可并行）→ 执行 → 回填 → 最终回答，不再调工具。无法处理"先查 A 再根据 A 的结果查 B"
   的链式依赖。对比实验中"茅台 vs 五粮液"部分方式模型想再查一次但做不到。扩展为 Agent 多轮循环（ReAct）即可，属后续课程。

7. **MCP tool 函数为什么会递归调用自己？**
   `from src.rag_backend import search_annual_report` 后又用 `@mcp.tool()` 定义同名 `def search_annual_report`
   ，后者遮蔽了前者，函数体内 `return search_annual_report(...)` 解析到自身。用 `as` 别名导入隔离命名空间即可。

8. **如何选型？**
   快速原型/单模型 → Function Call（接入成本最低）；多工具生态/跨产品复用 → MCP（写一次到处接）；工程师场景/现成命令多 →
   CLI+沙箱（零封装、与模型无关）。
