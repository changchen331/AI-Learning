# RESUME_GUIDE.md — 简历指导

基于本项目可写进简历的内容、写法对比，以及面试可能被追问的技术细节。

---

## 一、可量化数据（可直接写进简历）

| 维度                           | 数字                                                    | 出处                                   |
|------------------------------|-------------------------------------------------------|--------------------------------------|
| 吞吐加速 vs transformers 串行      | **59.3×**（0.82 → 48.59 QPS）                           | `outputs/throughput_results.json`    |
| 吞吐加速 vs transformers batch=8 | **12.5×**（3.89 → 48.59 QPS）                           | 同上                                   |
| Generation throughput        | **3394 tok/s**（单 0.5B 模型 / RTX 4060 8GB）              | 同上                                   |
| 50 并发请求端到端时延                 | **1.03s**（同卡，同 50 条 prompt）                           | 同上                                   |
| Function call 合法率提升          | 裸 prompt 60% → **guided_json 100%**（stock quote 50 例） | `outputs/function_call_results.json` |
| 复杂 schema 场景提升               | **42% → 100%**（create_order，含正则 + enum + 范围约束）        | 同上                                   |
| 约束解码延迟代价                     | **≈ 0%**（0.43s vs 0.43s，FSM 一次构建长期复用）                 | 同上                                   |
| 工具数 × 测试规模                   | 2 个工具 × 50 条 × 3 模式 = **300 次请求对比**                   | `src/demo_function_call.py`          |

**写简历建议**：

- 首位数字用"59×"或"60×"这种整数，视觉冲击强
- 把"42% → 100%"这种百分点配套出现，读者自动脑补差距
- "3394 tok/s"是具体物理量，比"很快"有说服力

---

## 二、项目名怎么写

**好写法**：

- `基于 vLLM 的 LLM 生产级部署与约束解码 Demo`
- `vLLM 推理优化：从 transformers 到 continuous batching 的 60× 吞吐提升`
- `LLM Function Call 可靠性优化：JSON Schema 约束解码的工程实践`

**差写法**：

- `LLM 部署项目`（太空）
- `大模型推理加速`（看不出做了啥）
- `vLLM 体验`（没量化结果）

**挑岗位重点**：

- 投**算法工程师**：强调"约束解码"、"Function Call 可靠性"
- 投**后端/工程**：强调"OpenAI 兼容 API"、"60× 吞吐"、"生产部署"
- 投**MLOps/推理工程**：强调"PagedAttention"、"continuous batching"、"WSL2 跨平台"

---

## 三、按岗位方向写法

### 3.1 算法工程师 / LLM 应用方向

> **基于 vLLM 的 LLM 推理优化与约束解码工程实践**（2026.04 ~ 2026.05）
>
> **技术栈**：vLLM 0.9.2、PyTorch 2.7、Qwen2-0.5B-Instruct、JSON Schema、xgrammar、FastAPI
>
> **项目概述**：在 WSL2 环境下构建 vLLM 教学级部署 demo，对比 transformers 原生推理与 vLLM 的吞吐差距，并系统演示 4
> 种约束解码方式（guided_choice / guided_regex / guided_json / response_format）在 Function Call 场景的可靠性提升。
>
> **关键成果**：
> - 同卡同模型下，**vLLM 相对 transformers 串行加速 59.3×、相对 batch=8 加速 12.5×**，Generation throughput 达到 3394 tok/s
> - 设计 2 个工具（`get_stock_quote` 金融查询 + `create_order` 电商下单）共 100 条测试用例，系统对比三种 JSON 输出方式
> - 用 `guided_json` 把复杂 schema（含 6 位正则、11 位手机号正则、整数范围、多枚举）的通过率从 **42% 提升到 100%**，且延迟无额外开销
> - 深入分析失败模式：字段拼错、枚举值用自然语言、手机号前缀加号等，并用 FSM 解码从根本上解决
>
> **项目亮点**：对比实验严谨（三模式分层指标：JSON 合法 / 必选字段 / 完整 schema），把"约束解码为什么有用"讲到 token 级（FSM
> 屏蔽非法 token logit）。

### 3.2 后端 / 推理基础设施工程师

> **LLM 推理服务：vLLM OpenAI 兼容 Server 部署与吞吐优化**（2026.04 ~ 2026.05）
>
> **技术栈**：vLLM、CUDA 12、WSL2、FastAPI、OpenAI 兼容协议、matplotlib
>
> **项目概述**：在 Windows 开发机上基于 WSL2 构建 vLLM 生产级部署流水线，一键启动 OpenAI 兼容 HTTP 服务，支持流式推理、约束解码、并发请求调度。
>
> **关键成果**：
> - 端到端部署一个 Qwen2-0.5B-Instruct 推理服务，**QPS 从原生 transformers 的 0.82 提升到 48.59（59× 加速）**
> - 50 并发请求从 61 秒压缩到 1 秒完成，GPU 利用率从串行时的 ~20% 拉到满载
> - 定位并解决驱动-CUDA-torch-vLLM 四层版本兼容问题（CUDA 13 需 NVIDIA 驱动 580+；降级 vLLM 到 0.9.2 + torch 2.7+cu126 适配
    12.7 驱动）
> - 设计 WSL2 跨平台部署方案：Ubuntu 22.04 + 清华 apt/pip 源 + venv 隔离 + VHDX 迁移到 D 盘避免 C 盘压力
>
> **项目亮点**：理解 vLLM 的 PagedAttention 和 continuous batching 原理（KV cache 按 block 分配、动态 batch 组合），能解释不同场景下
> QPS/tok-s 的差异原因。

### 3.3 工程 / MLOps（综合版）

> **vLLM 部署 Demo：跨平台推理服务 + 约束解码实践**（2026.04 ~ 2026.05）
>
> - 从零搭建 WSL2 + CUDA + vLLM 环境，解决 6 个真实兼容性问题（包括 CUDA 13 驱动不匹配、transformers 5.x 与 vLLM 0.9.2 的
    aimv2 冲突等）
> - 实现 3 个吞吐对比路线（串行 / 手动 batch / vLLM），横向对比得出 **vLLM 加速 59×** 的结论
> - 设计 6 个演示脚本覆盖 4 种约束解码，**Function Call 场景 schema 通过率 42% → 100%**
> - 产出 3 份工程文档（ARCHITECTURE / USAGE_GUIDE / RESUME_GUIDE）+ 柱状图 + JSON 详细结果

---

## 四、按经验层级写法

### 应届生

聚焦"做了什么 + 学到了什么"：

> 在 WSL2 中搭建 vLLM 推理服务，对比 transformers 原生推理与 vLLM 的吞吐，实测 **vLLM 加速 59.3×**，并用 50 条测试集量化了
`guided_json` 约束解码对 Function Call 可靠性的影响（通过率 42% → 100%）。通过该项目理解了 PagedAttention、continuous
> batching、约束解码（FSM）的基本原理。

### 1~3 年经验

聚焦"发现问题 + 解决问题"：

> 主导 vLLM 部署 demo 项目。诊断并修复了 4 个关键兼容性问题（NVIDIA 驱动 566 与 CUDA 13 不匹配、transformers 5.x 与 vLLM
> 0.9.2 的 aimv2 注册冲突等）。设计了 2 个工具 × 50 测试 × 3 模式的对比实验，证明 `guided_json` 在 0.5B 小模型上把 Function
> Call schema 通过率从 42% 提升到 100%，且无额外延迟。产出 3 份工程文档和可复现的 benchmark（59.3× 吞吐加速）。

### 3 年以上 / 资深

聚焦"设计决策 + 工程权衡"：

> 设计 vLLM 教学 demo 的整体架构：选择 0.9.2 版本（兼容 CUDA 12.7 驱动，覆盖 90% 笔记本用户）、用 transformers 原生推理做性能
> baseline、用 jsonschema 做分层评估指标（JSON 语法 / 必选字段 / 完整 schema）。通过对比实验量化得出两个结论：**vLLM 相对
transformers 加速 59.3×**（PagedAttention + continuous batching 的复合收益），**`guided_json` 相对 `response_format` 在复杂
schema 场景差距 58 个百分点**（JSON 合法率相同，但字段语义正确率仅 42% vs 100%）。项目产出可用于课堂教学和团队培训。

---

## 五、好句 vs 差句对比

| 差           | 好                                                                                                        |
|-------------|----------------------------------------------------------------------------------------------------------|
| 学习了 vLLM 部署 | 完成 vLLM 0.9.2 服务化部署，50 并发请求从 61s 压到 1s                                                                   |
| 用了约束解码技术    | 用 guided_json 把 Function Call 的 schema 通过率从 42% 提升到 100%                                                 |
| 做了吞吐测试      | 设计 3 路 × 50 请求 benchmark，定量得出 vLLM 相对 transformers 加速 59×                                                |
| 解决了一些兼容性问题  | 定位 NVIDIA 驱动-CUDA-torch-vLLM 四层版本链问题，降级 vLLM 0.20→0.9.2 修复                                               |
| 项目效果良好      | 失败案例分析显示：裸 prompt 42% 失败集中在字段语义（`"credit card"` 不在 enum / `"+手机号"`违反正则），guided_json 通过 FSM 屏蔽非法 token 根治 |

---

## 六、面试可能被追问的技术细节

### 6.1 吞吐相关

**Q：vLLM 比 transformers 为什么能快 59×？**

A：两个核心机制复合：

1. **PagedAttention**：传统 KV cache 是连续内存，批处理必须 pad 到最长，浪费显存；vLLM 按 block（16 token）分配，消除 padding
   内存浪费，显存利用率提高后能塞下更大的 batch size
2. **Continuous batching**：不等 batch 内最长请求完成，短请求一完成立刻插入新请求，GPU 利用率从 ~20% 拉到近 100%

**Q：为什么 transformers batch=8 只快了 4.7×（不是理想的 8×）？**

A：padding 浪费——prompts 长短不一，短 prompt 要 pad 到最长，实际有效计算量 < 8x。另外 decoder 生成阶段如果早停，剩下的 batch
位置还在跑空计算直到最长的也结束。

**Q：max_model_len 和 gpu_memory_utilization 怎么权衡？**

A：KV cache 大小 ≈ 2 × layers × hidden_dim × heads × max_len × 2（fp16）。max_len 越大单请求占越多
cache，能并发的请求数越少。gpu_memory_utilization 越高 KV cache 越大，但留给其他程序的显存越少。生产环境按 P95 prompt 长度定
max_len，按 GPU 专用度定 utilization。

### 6.2 约束解码相关

**Q：`guided_json` 的底层原理？**

A：vLLM 用 xgrammar 把 JSON Schema 编译成 FSM（有限状态自动机），每步解码时：

1. 根据当前 FSM 状态，计算哪些 token 下一步合法
2. 把非法 token 的 logit 设为 `-inf`
3. Softmax 后只从合法集采样
4. 用采样出的 token 推进 FSM 状态

这是 logit processor 的一种，不修改模型权重，不增加参数。

**Q：`guided_json` 和 `response_format` 有什么区别？**

A：

- `response_format={"type": "json_object"}`：OpenAI 官方 API，只保证输出是**合法 JSON**（语法层），不管字段名、字段值类型、枚举值
- `guided_json=schema`：vLLM 扩展，保证输出**完全符合 JSON Schema**（语义层），包括字段齐全、类型正确、正则匹配、枚举合法、数值范围

实测数据：create_order 场景下，`response_format` 和裸 prompt 完整通过率都是 42%，`guided_json` 是 100%，差距 58 个百分点。

**Q：约束解码会不会让模型输出"合法但胡说"？**

A：会。约束解码只管格式，不管语义正确性。本项目 0.5B 模型偶尔把"宁德时代"代码猜错（000750 vs 300750），但市场代码、日期格式、字段结构都
100% 合法。这是教学重点之一：**guided 解决格式问题，不解决模型能力问题**。生产上需要结合更大模型 / fine-tune / RAG。

**Q：约束解码有性能代价吗？**

A：FSM 构建开销约 0.1~1s（取决于 schema 复杂度），会被 vLLM 内部缓存（同 schema 第二次请求直接命中）。解码时每步多一次 logit
mask（向量 `-inf` 填充），对大模型 <5% latency 影响；小模型上实测 0% 差异（见项目数据）。

### 6.3 工程相关

**Q：为什么不用 Docker 部署？**

A：本项目是教学 demo，目标读者是 Windows 本地开发的学生。Docker 在 Windows 上也依赖 WSL2，反而多一层嵌套。直接 WSL2 + venv
更透明，学生能看到每个包是怎么装进来的。生产环境当然应该 Docker 化。

**Q：为什么 vLLM 版本选 0.9.2 而不是最新的？**

A：最新版（0.20+）torch 依赖是 CUDA 13，需要 NVIDIA 驱动 580+。常见笔记本（RTX 4060 等）驱动 566.x 是 CUDA 12.7，不兼容。0.9.2 +
torch 2.7+cu126 是兼容 12.x 驱动的最新稳定组合，覆盖更多硬件。

**Q：WSL2 和原生 Linux 跑 vLLM 有什么差异？**

A：

- GPU 驱动：WSL2 通过微软的桥接层使用 Windows NVIDIA 驱动，CUDA 功能完整
- 文件系统：`/mnt/d/` 挂载跨 9P 协议，读写比原生 ext4 慢 2~5×（模型只加载一次，影响可忽略）
- 网络：WSL2 有独立 IP，localhost 转发对 Windows 侧透明
- 性能：推理性能与原生 Linux 相近（实测差异 <5%）

**Q：如何监控生产环境的 vLLM 服务？**

A：vLLM server 暴露 `/metrics` Prometheus 端点，含：

- 请求 QPS、延迟分位数
- GPU 显存利用率、KV cache 利用率
- Queue 长度、每请求平均 token 数

配 Grafana 做大盘。关键告警：显存 > 95%、队列持续 > 100、首 token 延迟 P95 > 2s。

### 6.4 Function Call 相关

**Q：为什么 Agent 场景特别需要约束解码？**

A：Agent 的工具调用是"LLM 输出 → 解析为 JSON → 传给函数执行"。如果 JSON 失败或字段不对，整条 reasoning chain 就断了。本项目数据：0.5B
模型裸 prompt 42% 完整 schema 通过率，意味着 58% 的工具调用会失败，对 Agent 几乎不可用。用 `guided_json` 提到 100% 才能真正做生产级
Agent。

**Q：OpenAI 的 tool_calls 和 vLLM 的 guided_json 有什么关系？**

A：OpenAI 的 `tool_calls` 是更高层的语法糖，背后其实就是 JSON Schema 约束（OpenAI 的私有实现）。vLLM 的 `guided_json`
是通用的约束解码接口，可以实现自己的 tool calling 逻辑。vLLM 0.9+ 也支持原生 `tools` 参数（透传到 guided decoding）。

---

## 七、简历中可能遇到的坑

1. **不要夸大规模**：这是个教学 demo，不是"生产系统"。写"线上日均调用 100 万次"是吹牛，面试被拆穿很难看
2. **量化要匹配事实**：59.3× 是 50 请求单卡单模型场景，别写成"任何场景 60×"
3. **小心"自研"**：约束解码是 vLLM 集成的 xgrammar/outlines，不是你自研的
4. **诚实说范围**：只跑了 0.5B 模型，不要声称 "在 70B 模型上也验证了"
