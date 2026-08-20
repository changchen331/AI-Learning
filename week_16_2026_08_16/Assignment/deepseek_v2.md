# DeepSeek-V2 模型结构特点分析
---

## 1. 总体结构

该实现的 DeepSeek-V2 是一个 **自回归 Transformer Decoder-only 语言模型**。整体可以概括为：

```text
Input IDs
   │
   ▼
Token Embedding
   │
   ▼
┌─────────────────────────────────────┐
│       DeepseekV2 Decoder Layers     │ × N
│                                     │
│  RMSNorm                            │
│     ↓                               │
│  Multi-head Latent Attention (MLA)  │
│     ↓                               │
│  Residual Connection                │
│     ↓                               │
│  RMSNorm                            │
│     ↓                               │
│  Dense MLP / MoE                   │
│     ↓                               │
│  Residual Connection                │
└─────────────────────────────────────┘
   │
   ▼
Final RMSNorm
   │
   ▼
Linear LM Head
   │
   ▼
Vocabulary Logits
```

代码中 `DeepseekV2Model` 使用 `config.num_hidden_layers` 个 `DeepseekV2DecoderLayer` 堆叠，并在最后使用 RMSNorm；
`DeepseekV2ForCausalLM` 再接一个无 bias 的线性 `lm_head` 映射到词表空间。

因此，它的整体骨架仍然是典型的：

**Embedding → Transformer Decoder Blocks → Norm → LM Head**

真正具有 DeepSeek-V2 代表性的部分主要集中在两个地方：

1. **MLA（Multi-head Latent Attention）**
2. **MoE（Mixture-of-Experts）**

此外还结合了 RMSNorm、RoPE/扩展 RoPE、Flash Attention、KV Cache 等工程设计。

---

# 2. Decoder Layer：标准骨架 + DeepSeek 特化模块

每个 `DeepseekV2DecoderLayer` 的计算顺序非常清晰：

```text
x
│
├── RMSNorm
│
├── Self Attention / MLA
│
└── + Residual
     │
     ├── RMSNorm
     │
     ├── MLP / MoE
     │
     └── + Residual
```

代码中首先进行 `input_layernorm`，然后进入 self-attention；attention 输出与原始输入做残差相加。之后再次进行
`post_attention_layernorm`，进入 MLP 或 MoE，再做一次残差连接。

这种结构属于 **Pre-Norm Transformer** 风格。

### 2.1 RMSNorm

模型没有使用传统 LayerNorm，而是实现了 `DeepseekV2RMSNorm`。

其核心计算为：

```text
variance = mean(x²)
x = x / sqrt(variance + eps)
y = weight * x
```

实现中先将 hidden states 转换为 FP32 完成归一化，再转换回输入 dtype。fileciteturn2file6L607-L624

特点：

- 不计算均值，只基于二阶矩进行归一化；
- 参数量比 LayerNorm 更简单；
- 计算相对直接；
- 对深层 Transformer 的训练稳定性较友好。

---

# 3. 核心特点一：MLA（Multi-head Latent Attention）

DeepSeek-V2 与普通 Multi-Head Attention 最大的结构差异之一，就是它并不是简单地对 hidden states 做：

```text
Q = XWq
K = XWk
V = XWv
```

而是采用了 **低秩潜变量压缩**思路。

代码中的 attention 明确包含：

- `q_a_proj`
- `q_a_layernorm`
- `q_b_proj`
- `kv_a_proj_with_mqa`
- `kv_a_layernorm`
- `kv_b_proj`

这些模块共同构成了 MLA 的核心。

---

## 3.1 Query 的低秩投影

Query 路径为：

```text
hidden_states
      │
      ▼
   q_a_proj
      │
      ▼
q_a_layernorm
      │
      ▼
   q_b_proj
      │
      ▼
   Query
```

代码：

```python
q = self.q_b_proj(
    self.q_a_layernorm(
        self.q_a_proj(hidden_states)
    )
)
```

随后 Query 被拆分成两部分：

```text
Q
├── q_nope
└── q_pe
```

其中 `q_pe` 是用于位置编码的部分，而 `q_nope` 不直接施加 RoPE。

---

# 4. Query / Key 的“位置相关”和“内容相关”解耦

这是该实现中非常值得注意的一个结构特点。

代码显式定义：

```text
q_head_dim
    =
qk_nope_head_dim
    +
qk_rope_head_dim
```

也就是说，一个 attention head 的 Query / Key 表示被拆成：

```text
┌───────────────────────────────┐
│ Query / Key                   │
├───────────────┬───────────────┤
│ NoPE 部分     │ RoPE 部分     │
│ 内容信息      │ 位置信息      │
└───────────────┴───────────────┘
```

代码中先得到 `q_nope` 和 `q_pe`，然后只对 `q_pe`、`k_pe` 应用 Rotary Position Embedding。

这说明 DeepSeek-V2 并不是简单地把 RoPE 作用到完整的 Q/K 上，而是：

> **将内容表示与位置信息分开处理。**

这对于理解 MLA 非常关键。

---

# 5. KV 路径：压缩 KV 表示

KV 路径更加体现 MLA 的特点。

首先：

```text
hidden_states
      │
      ▼
kv_a_proj_with_mqa
      │
      ├───────────────┐
      ▼               ▼
compressed_kv        k_pe
      │
      ▼
kv_a_layernorm
      │
      ▼
kv_b_proj
      │
      ├───────────────┐
      ▼               ▼
    k_nope           Value
```

代码首先把 hidden states 映射到：

```text
kv_lora_rank + qk_rope_head_dim
```

随后拆成：

- `compressed_kv`
- `k_pe`

然后仅对压缩 KV 表示做 LayerNorm，再通过 `kv_b_proj` 恢复多头的 Key/Value 表示。

因此，KV 并不是直接保存完整的每头 K/V 表示，而是先经过一个 **低秩 latent representation**。

---

# 6. MLA 最重要的优势：降低 KV Cache 压力

从代码结构可以看出，模型显式引入：

```text
kv_lora_rank
```

来压缩 KV 表示。

这类设计的核心意义并不是简单地“让网络更小”，而是：

> **降低自回归生成过程中 KV Cache 的存储和访问成本。**

传统 Transformer 在生成阶段，需要缓存过去 token 的 K/V：

```text
Token 1 → K1,V1
Token 2 → K2,V2
Token 3 → K3,V3
...
Token t → Kt,Vt
```

上下文越长，KV Cache 越大。

MLA 则通过 latent representation 对 KV 进行压缩，使缓存阶段可以更多地依赖低维潜表示，从而降低长上下文推理时的内存压力。

代码同时支持 `past_key_values`，并在 attention 中将缓存长度加入当前 KV sequence length。

因此可以把 MLA 的设计目标概括为：

**保持多头注意力表达能力 + 压缩 KV 表示 + 降低推理 KV Cache 成本。**

---

# 7. RoPE：位置编码设计

代码实现了多个 RoPE 版本：

```text
DeepseekV2RotaryEmbedding
DeepseekV2LinearScalingRotaryEmbedding
DeepseekV2DynamicNTKScalingRotaryEmbedding
DeepseekV2YarnRotaryEmbedding
```

并根据 `config.rope_scaling["type"]` 选择：

```text
linear
dynamic
yarn
```

或者使用普通 RoPE。 这说明代码设计上并不是把 RoPE 写死，而是为 **长上下文位置编码扩展**留下了接口。

尤其是 YaRN 实现中，会对不同频率区间进行不同程度的插值/缩放。

---

# 8. 核心特点二：MoE

DeepSeek-V2 的第二个关键结构就是 MoE。

代码中的 `DeepseekV2MoE` 被定义为：

> A mixed expert module containing shared experts.

即：

**路由专家（Routed Experts） + 共享专家（Shared Experts）**

的混合结构。

可以表示为：

```text
                     Hidden States
                           │
                           ▼
                         Gate
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
           Expert 1     Expert 2      ... Expert N
              │            │
              └──────┬─────┘
                     ▼
              Weighted Sum
                     │
                     ├──────────────┐
                     ▼              ▼
                Routed Output   Shared Experts
                     │              │
                     └──────┬───────┘
                            ▼
                          Output
```

---

# 9. Sparse MoE：每个 Token 不经过所有专家

Gate 首先对每个 token 计算所有 routed experts 的分数：

```python
logits = F.linear(hidden_states, self.weight)
scores = logits.softmax(...)
```

随后执行 Top-K expert selection。

也就是说：

```text
Token
  │
  ▼
Gate
  │
  ├── Expert 3
  └── Expert 17
```

而不是：

```text
Token
  │
  ├── Expert 1
  ├── Expert 2
  ├── Expert 3
  ├── ...
  └── Expert N
```

因此 MoE 的核心思想是：

> **参数规模可以非常大，但每个 token 实际激活的参数只是其中的一小部分。**

这形成了：

**Large Total Parameters + Sparse Activated Parameters**

的结构特点。

---

# 10. Top-K 路由机制

代码支持两种主要的 Top-K 方法：

```text
gready
group_limited_greedy
```

普通 greedy 直接从全部 expert 中选择 Top-K。

`group_limited_greedy` 则先：

1. 将 experts 划分成多个 group；
2. 计算 group-level score；
3. 先选择若干 group；
4. 再在这些 group 中选择 Top-K experts。

这种设计可以在大量专家的情况下，对 expert selection 进行结构化约束。

---

# 11. Expert 权重归一化

当一个 token 激活多个专家时，并不是简单把专家输出相加。

代码中：

```python
topk_weight = topk_weight / denominator
```

将 Top-K 权重归一化，使其和为 1。

因此可以表示为：

\[ y = \sum_{i \in TopK (x)} \alpha_i E_i (x)
\]

其中：

- \(E_i (x)\)：第 i 个专家的输出
- \(\alpha_i\)：Gate 给出的归一化权重
- \(y\)：最终 routed expert 输出

---

# 12. Shared Experts：共享专家

除了 routed experts，DeepSeek-V2 还支持 shared experts。

代码中：

```python
self.shared_experts = DeepseekV2MLP(...)
```

其输入直接来自原始 hidden states。

最终：

```python
y = y + self.shared_experts(identity)
```

即：

```text
Routed Experts Output
          │
          ├──────────┐
          │          │
          ▼          ▼
       routed      shared
       experts     experts
          │          │
          └────┬─────┘
               ▼
              Add
```

因此，它并不是纯粹的“专家二选一/多选一”结构，而是：

> **稀疏路由专家负责输入相关的专门化能力，共享专家负责所有 token 都可以获得的通用能力。**

这是一种很有代表性的 **混合专家结构**。

---

# 13. MoE 的负载均衡辅助损失

MoE 最大的问题之一是：

> Gate 可能长期偏爱少数专家。

如果发生：

```text
Expert 1  ████████████████
Expert 2  ███████████
Expert 3  ██
Expert 4  ▏
...
```

那么就会造成：

- 某些专家计算过载；
- 其他专家没有得到充分训练；
- 专家容量利用率低；
- 分布式训练中的通信/计算不均衡。

因此代码实现了 auxiliary loss。

Gate 会统计 expert 的选择情况和平均 routing probability，并计算辅助损失。

此外，通过 `AddAuxiliaryLoss` 自定义 autograd function，把辅助损失的梯度传回训练过程。

所以 MoE 部分不仅仅是：

**Gate → Top-K → Expert**

还包括：

**Gate → Top-K → Load Balancing Auxiliary Loss**

这一训练机制。

---

# 14. Dense MLP 与 MoE MLP 的统一性

有意思的一点是，MoE 中每一个 expert 本身并不是一个特殊的新型网络，而是一个 `DeepseekV2MLP`。

Dense MLP 的形式为：

```text
x
│
├── gate_proj ──→ Activation ──┐
│                              ×
└── up_proj ───────────────────┘
                               │
                               ▼
                           down_proj
                               │
                               ▼
                               y
```

代码为：

```python
down_proj(
    act_fn(gate_proj(x)) * up_proj(x)
)
```

这属于 **Gated MLP** 结构，而不是传统的单路：

```text
Linear → Activation → Linear
```

因此 DeepSeek-V2 的 FFN 部分实际上是：

```text
Dense Layer:
    Gated MLP

MoE Layer:
    Many Gated MLP Experts
          +
    Shared Gated MLP
```

---

# 15. 并不是所有 Decoder Layer 都一定使用 MoE

代码中有一个非常重要的判断：

```python
config.n_routed_experts is not None
and layer_idx >= config.first_k_dense_replace
and layer_idx % config.moe_layer_freq == 0
```

满足条件时使用：

```text
DeepseekV2MoE
```

否则使用：

```text
DeepseekV2MLP
```

因此模型结构可以是：

```text
Layer 0   → Dense MLP
Layer 1   → Dense MLP
Layer 2   → MoE
Layer 3   → Dense MLP
Layer 4   → MoE
...
```

具体哪些层使用 MoE，需要由 `DeepseekV2Config` 中的：

- `first_k_dense_replace`
- `moe_layer_freq`
- `n_routed_experts`

等配置决定。

这是一种 **稀疏化深度方向上的设计**：不是简单地把每一个 Transformer Block 都替换成 MoE。

---

# 16. Attention + MoE 的分工

从整个 Decoder Layer 来看：

```text
                Transformer Block
                       │
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
   MLA Attention                  MoE / MLP
         │                           │
         ▼                           ▼
  建模 Token 间关系             建模 Token 内部变换
         │                           │
         └─────────────┬─────────────┘
                       ▼
                  Residual
```

可以从功能角度理解：

### MLA

负责：

- token-token interaction
- 上下文信息聚合
- 长距离依赖
- 位置信息建模

### MoE

负责：

- 非线性特征变换
- 知识/能力的参数化
- token-dependent expert specialization

所以 DeepSeek-V2 的核心思路可以概括成：

> **Attention 负责“看谁”，MoE 负责“怎么处理”。**

---

# 17. Flash Attention 支持

代码提供：

```python
DeepseekV2Attention
DeepseekV2FlashAttention2
```

两种 attention implementation。

Flash Attention 版本会：

1. 根据 attention mask 去除 padding；
2. 调用 `flash_attn_varlen_func` 或 `flash_attn_func`；
3. 得到结果后再恢复 padding 布局。

因此这里需要区分：

**MLA 是模型结构设计；Flash Attention 是高效计算实现。**

两者不是同一层面的优化。

---

# 18. KV Cache 与自回归生成

模型支持：

```text
past_key_values
```

并且可以在生成过程中只输入尚未处理的新 token。

代码的 generation preparation 会根据 cache length 和 input length 截取真正需要计算的 token。

这使模型可以采用：

```text
第一次：
[token1 token2 token3 token4]

第二次：
                    [token5]

第三次：
                           [token6]
```

而不是每一步都重新计算整个历史序列。

结合 MLA 后，KV Cache 的内存效率成为 DeepSeek-V2 架构中的一个关键设计目标。

---

# 19. 最终语言模型 Head

经过所有 Decoder Layers 后：

```text
hidden_states
      │
      ▼
Final RMSNorm
      │
      ▼
   lm_head
      │
      ▼
Vocabulary logits
```

代码中：

```python
self.lm_head = nn.Linear(
    config.hidden_size,
    config.vocab_size,
    bias=False
)
```

最终 logits 的形状可以理解为：

```text
[batch_size, sequence_length, vocab_size]
```

训练时采用标准 causal language modeling：

```text
x1 → predict x2
x2 → predict x3
x3 → predict x4
...
```

代码明确将 logits 与 labels 做 shift，然后使用 `CrossEntropyLoss`。

---

# 20. 结构上的核心创新总结

如果只抓 DeepSeek-V2 最重要的结构特点，可以浓缩成下面这张表：

| 模块             | 传统 Transformer | DeepSeek-V2 实现       | 主要作用                       |
|------------------|------------------|------------------------|--------------------------------|
| Norm             | LayerNorm 常见   | RMSNorm                | 稳定训练、简化归一化           |
| Attention        | MHA/GQA 等       | MLA                    | 压缩 KV、降低 KV Cache         |
| Position         | RoPE             | RoPE + 多种 scaling    | 支持更灵活的位置扩展           |
| FFN              | Dense MLP        | Dense MLP + MoE        | 提升参数容量                   |
| Expert           | 无               | Routed Experts         | token-dependent specialization |
| Shared Expert    | 无               | Shared Experts         | 提供共享通用能力               |
| Routing          | 无               | Top-K Gate             | 稀疏激活                       |
| Load Balance     | 无               | Auxiliary Loss         | 防止专家负载失衡               |
| Attention Kernel | 普通 attention   | 可选 Flash Attention 2 | 提升计算效率                   |
| Generation       | KV Cache         | 支持 KV Cache          | 加速自回归生成                 |

---

# 21. DeepSeek-V2 最核心的“组合拳”

从架构设计角度，DeepSeek-V2 并不是单独依赖某一个技巧，而是把多个优化组合起来：

```text
                         DeepSeek-V2
                              │
          ┌───────────────────┴───────────────────┐
          │                                       │
       Attention                                FFN
          │                                       │
          ▼                                       ▼
         MLA                                    MoE
          │                                       │
    ┌─────┴─────┐                       ┌─────────┴─────────┐
    │           │                       │                   │
  Latent     RoPE                     Routed             Shared
   KV                               Experts             Experts
    │           │                       │                   │
    └─────┬─────┘                       └─────────┬─────────┘
          │                                       │
          ▼                                       ▼
      KV Cache                           Sparse Activation
      Efficiency                         Parameter Efficiency
```

因此可以把它理解成两条主要优化路线：

### 路线 A：降低推理成本

**MLA → 压缩 KV 表示 → 降低 KV Cache 开销**

### 路线 B：提高参数效率

**MoE → 大量专家参数 → 每个 token 只激活少量专家**

最终形成：

> **更大的参数容量 + 更低的单 token 激活成本 + 更低的 KV Cache 成本**

---

# 22. 与普通 Transformer 的本质区别

如果把普通 Transformer 简化为：

```text
Attention
   +
Dense FFN
```

那么 DeepSeek-V2 可以简化为：

```text
                Transformer
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
     MLA Attention              MoE
          │                     │
    KV latent compression   Sparse Experts
          │                     │
          └──────────┬──────────┘
                     ▼
                  Output
```

所以 DeepSeek-V2 的关键并不是“重新发明 Transformer”，而是在 Transformer 的两个主要计算瓶颈上分别做了针对性优化：

### Attention 侧

传统：

```text
完整 K/V
↓
KV Cache 很大
```

DeepSeek-V2：

```text
Latent KV
↓
更紧凑的 KV 表示
```

### FFN 侧

传统：

```text
一个巨大 Dense FFN
↓
每个 token 都计算全部参数
```

DeepSeek-V2：

```text
Gate
↓
Top-K Experts
↓
只激活部分专家
```

---

# 23. 工程层面的特点

除了架构创新，这份实现还体现出较明显的工程优化：

## 23.1 Flash Attention

支持 `flash_attention_2`，并针对 padding token 做 unpadding。

## 23.2 Distributed Expert Parallelism

MoE 中存在 `ep_size`、`ep_rank`、`experts_per_rank` 等逻辑，并通过 `dist.all_to_all` 在不同设备之间交换 token。

这说明 MoE 不只是算法层面的稀疏化，同时考虑了 **多 GPU / 多节点专家并行**。

## 23.3 Gradient Checkpointing

`DeepseekV2PreTrainedModel` 声明支持 gradient checkpointing。

## 23.4 KV Cache

模型完整支持 `past_key_values` 和 generation cache。

---

# 24. 一句话总结

如果需要在论文、汇报或者面试中用一句话介绍 DeepSeek-V2 的结构，可以说：

> **DeepSeek-V2 是一种基于 Decoder-only Transformer 的大语言模型，其核心架构特点是在 Attention 中采用 MLA，通过低秩潜表示压缩
KV 以降低 KV Cache 开销；在 FFN 部分采用带共享专家的稀疏 MoE，通过 Top-K 路由实现大参数容量与低激活计算成本之间的平衡，同时结合
RMSNorm、RoPE、Flash Attention 和 KV Cache 等设计提升训练与推理效率。**

---

# 25. 阅读代码时建议重点关注的类

如果后续要继续深入这份实现，建议按照下面顺序阅读：

```text
DeepseekV2Model
      │
      ▼
DeepseekV2DecoderLayer
      │
      ├───────────────┐
      ▼               ▼
DeepseekV2Attention  DeepseekV2MoE
      │               │
      ▼               ▼
   MLA / RoPE      MoEGate
                      │
                      ▼
                 DeepseekV2MLP
```

其中最值得重点研究的是：

1. `DeepseekV2Attention` —— 理解 MLA；
2. `MoEGate` —— 理解 Top-K routing；
3. `DeepseekV2MoE` —— 理解 routed + shared experts；
4. `DeepseekV2DecoderLayer` —— 理解完整 Block 数据流；
5. `DeepseekV2ForCausalLM` —— 理解最终语言模型输出与训练目标。

---

## 结论

从这份实现本身来看，DeepSeek-V2 的结构特点可以归纳为四个关键词：

**MLA + MoE + Sparse Activation + KV Compression**

其中：

- **MLA** 主要解决 Attention 尤其是长上下文推理中的 KV Cache 成本；
- **MoE** 主要解决模型参数规模与计算成本之间的矛盾；
- **Sparse Activation** 让模型拥有大量参数，但单个 token 只使用少部分专家；
- **Shared Experts** 保留跨 token 的通用能力；
- **RMSNorm + RoPE + Residual** 构成稳定的 Transformer 基础结构；
- **Flash Attention + KV Cache + Expert Parallelism** 则进一步把上述结构落到高效训练和推理实现上。
