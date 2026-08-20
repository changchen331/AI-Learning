# Qwen-V1 模型结构特点分析
---

## 1. 总体结构

这份 Qwen-V1 实现属于典型的 **Decoder-only Transformer 自回归语言模型**：

```text
Input IDs
   │
   ▼
Token Embedding
   │
   ▼
Embedding Dropout
   │
   ▼
┌───────────────────────────────┐
│        QWenBlock × N          │
│                               │
│  RMSNorm                      │
│     ↓                         │
│  Self Attention               │
│     ↓                         │
│  Residual Connection          │
│     ↓                         │
│  RMSNorm                      │
│     ↓                         │
│  Gated MLP                    │
│     ↓                         │
│  Residual Connection          │
└───────────────────────────────┘
   │
   ▼
Final RMSNorm
   │
   ▼
LM Head
   │
   ▼
Vocabulary Logits
```

代码中的 `QWenModel` 创建 `config.num_hidden_layers` 个 `QWenBlock`，输入首先进入 `wte` embedding 和 dropout，所有 Block
完成后进入最终 `ln_f`；`QWenLMHeadModel` 再通过一个线性 `lm_head` 映射到词表。

所以它的基本骨架可以概括为：

**Embedding → Transformer Decoder Blocks → RMSNorm → LM Head**

与后来的 DeepSeek-V2 / V3 / Kimi K2 类模型相比，这一代 Qwen 的结构明显更加传统： **Dense Transformer + MHA + Gated MLP**
，没有 MLA，也没有 MoE。

---

# 2. QWenBlock：标准 Pre-Norm Transformer

每一个 `QWenBlock` 的数据流是：

```text
x
│
├── RMSNorm
│
├── Self Attention
│
└── + Residual
     │
     ├── RMSNorm
     │
     ├── Gated MLP
     │
     └── + Residual
```

代码：

```python
layernorm_output = self.ln_1(hidden_states)

attn_outputs = self.attn(layernorm_output, ...)
attn_output = attn_outputs[0]

layernorm_input = attn_output + residual

layernorm_output = self.ln_2(layernorm_input)

mlp_output = self.mlp(layernorm_output)
hidden_states = residual + mlp_output
```

因此 Qwen-V1 使用的是典型的 **Pre-Norm Transformer**。

与 Post-Norm 结构：

```text
Attention → Add → Norm
```

不同，它采用：

```text
Norm → Attention → Add
```

这样做通常有利于深层 Transformer 的训练稳定性。

---

# 3. RMSNorm

Qwen-V1 使用 `RMSNorm`，而不是传统 LayerNorm。

在 `QWenBlock` 中可以看到：

```python
self.ln_1 = RMSNorm(...)
self.ln_2 = RMSNorm(...)
```

模型最后也使用：

```python
self.ln_f = RMSNorm(...)
```

RMSNorm 的核心思想是只对 hidden states 的均方根进行归一化：

\[ \mathrm{RMS} (x)=\sqrt{\frac{1}{d}\sum_i x_i^2} \]

然后：

\[ y=\frac{x}{\mathrm{RMS} (x)+\epsilon}\odot w \]

它不需要计算均值，因此比 LayerNorm 更简单。

---

# 4. 核心特点一：标准 Multi-Head Self-Attention

Qwen-V1 的 Attention 是比较传统的 **Multi-Head Self-Attention（MHA）**。

代码中：

```python
self.num_heads = config.num_attention_heads
self.head_dim = self.hidden_size // self.num_heads
```

并且：

```python
self.c_attn = nn.Linear(
    config.hidden_size,
    3 * self.projection_size
)
```

因此输入：

```text
X
│
▼
c_attn
│
├── Query
├── Key
└── Value
```

一次线性投影同时生成 Q/K/V。

---

# 5. Q/K/V 的计算方式

代码：

```python
mixed_x_layer = self.c_attn(hidden_states)

query, key, value = mixed_x_layer.split(
    self.split_size,
    dim=2
)
```

然后分别拆成多个 attention heads：

```python
query = self._split_heads(...)
key = self._split_heads(...)
value = self._split_heads(...)
```

因此标准的数据流为：

```text
Hidden State
     │
     ▼
Linear
     │
 ┌───┼───┐
 ▼   ▼   ▼
 Q   K   V
 │   │   │
 └───┴───┘
      │
 Multi-Head Attention
      │
      ▼
   c_proj
```

与 DeepSeek-V2 的 MLA 相比，这里没有低秩 KV latent compression。

---

# 6. Attention 输出投影

Attention 得到多个 head 的结果之后：

```python
context_layer = self._merge_heads(
    attn_output,
    self.num_heads,
    self.head_dim
)
```

然后：

```python
attn_output = self.c_proj(context_layer)
```

所以完整 Attention 可以写成：

\[ \mathrm{Attention} (X)
= \mathrm{Concat} (head_1,\ldots,head_h)W_O \]

这就是经典 Transformer Attention 的基本形式。

---

# 7. RoPE：Qwen-V1 的位置编码特点

Qwen-V1 使用 Rotary Position Embedding（RoPE）。

代码中：

```python
self.rotary_emb = RotaryEmbedding(
    dim,
    base=config.rotary_emb_base
)
```

模型还支持：

```python
rotary_pct
```

这意味着 RoPE 不一定作用于整个 head dimension。

如果：

```text
rotary_pct = 1.0
```

则整个 KV channel 都使用 rotary embedding。

否则：

```text
rotary_ndims
=
kv_channels × rotary_pct
```

只对其中一部分维度应用 RoPE。

---

# 8. RoPE 的实际作用位置

在 Attention 中：

```python
query = apply_rotary_pos_emb(query, q_pos_emb)
key = apply_rotary_pos_emb(key, k_pos_emb)
```

即：

```text
Q ──→ RoPE ──┐
              │
              ▼
          Attention
              ▲
              │
K ──→ RoPE ──┘

V ───────────────→ Attention
```

只对 Q/K 进行位置编码，Value 不进行 RoPE。

这是 Transformer 中使用 RoPE 的典型方式。

---

# 9. Qwen-V1 的一个重要特点：Dynamic NTK

Qwen-V1 的代码专门实现了：

```python
self.use_dynamic_ntk = config.use_dynamic_ntk
```

并根据实际 KV sequence length 计算：

```python
ntk_alpha = self.get_ntk_alpha(true_seq_len)
```

然后：

```python
self.rotary_emb(
    kv_seq_len,
    ntk_alpha=ntk_alpha
)
```

其逻辑可以概括为：

```text
实际上下文长度
       │
       ▼
是否超过基础 seq_length？
       │
       ▼
Dynamic NTK scaling
       │
       ▼
调整 RoPE
```

因此 Qwen-V1 并不是简单地把最大上下文长度写死，而是尝试通过 Dynamic NTK 对更长上下文进行扩展。

这是这份早期 Qwen 实现中非常值得注意的地方。

---

# 10. LogN Attention

除了 Dynamic NTK，这份代码还支持：

```python
self.use_logn_attn = config.use_logn_attn
```

如果推理时序列长度超过基础 `seq_length`：

```python
query = query * logn_tensor
```

也就是说，它在长上下文场景下不仅调整 RoPE，还可以对 Query 做 LogN scaling：

```text
Long Context
     │
     ├── Dynamic NTK → 调整位置编码
     │
     └── LogN Attention → 调整 Query scale
```

这体现了 Qwen-V1 对长上下文问题的早期探索。

---

# 11. 核心特点二：Gated MLP

Qwen-V1 的 FFN 并不是最传统的：

```text
Linear
 ↓
Activation
 ↓
Linear
```

而是一个双分支 Gated MLP。

代码：

```python
a1 = self.w1(hidden_states)
a2 = self.w2(hidden_states)

intermediate_parallel = a1 * F.silu(a2)

output = self.c_proj(intermediate_parallel)
```

对应：

```text
                   x
                 /   \
                ▼     ▼
              w1(x)  w2(x)
                │     │
                │    SiLU
                │     │
                └── × ┘
                    │
                    ▼
                  c_proj
                    │
                    ▼
                    y
```

数学形式：

\[ \mathrm{MLP} (x)
= W_3\left (W_1x\odot \mathrm{SiLU} (W_2x)\right)
\]

这和后来的 SwiGLU 类结构非常接近。

---

# 12. 为什么 Gated MLP 要用两个投影？

传统 FFN：

\[ W_2\sigma (W_1x)
\]

只有一个中间激活路径。

Qwen-V1：

\[ W_3 (W_1x\odot\sigma (W_2x))
\]

多了一个门控路径：

```text
W1(x) ────────────┐
                   × → W3 → output
W2(x) → SiLU ─────┘
```

可以理解为：

> 一个分支产生特征，另一个分支决定这些特征应该被保留多少。

因此表达能力通常比简单的单路 FFN 更强。

---

# 13. Qwen-V1 没有 MoE

这是和前面两个模型最明显的区别之一。

当前代码中的 `QWenBlock` 直接：

```python
self.mlp = QWenMLP(config)
```

不存在：

```text
MoEGate
Experts
Shared Experts
Top-K Routing
Expert Parallelism
```

因此 Qwen-V1 是：

> **Dense Transformer**

而不是：

> Sparse MoE Transformer

这意味着每个 token 都会经过同一个 MLP 参数集合。

---

# 14. Dense Transformer 与 MoE 的区别

把你前面分析的 DeepSeek-V2 / Kimi K2 和 Qwen-V1 放在一起：

```text
Qwen-V1

Token
 │
 ▼
Attention
 │
 ▼
一个 Dense MLP
 │
 ▼
Output
```

而 MoE：

```text
Token
 │
 ▼
Gate
 │
 ├── Expert 1
 ├── Expert 2
 ├── ...
 └── Expert N
       │
     Top-K
       │
       ▼
    Output
```

因此 Qwen-V1 的参数和计算关系更加直接：

> **参数规模增加，通常意味着每个 token 的计算量也增加。**

而 MoE 可以做到：

> **总参数规模很大，但每个 token 只激活少量专家。**

---

# 15. 核心特点三：KV Cache

Qwen-V1 支持：

```python
past_key_values
```

在第一次计算后缓存历史 K/V：

```text
Prompt
 │
 ▼
Q/K/V
 │
 ▼
Cache K/V
```

后续生成：

```text
New Token
    │
    ▼
New Q/K/V
    │
    ├── New K/V → Cache
    │
    └── Q × Historical K/V
```

代码中：

```python
if layer_past is not None:
    past_key, past_value = layer_past
    key = torch.cat((past_key, key), dim=1)
    value = torch.cat((past_value, value), dim=1)
```

因此它支持标准的自回归 KV Cache。

---

# 16. Qwen-V1 一个很有意思的设计：KV Cache Quantization

这份代码不仅仅支持普通 KV Cache，还实现了：

```python
use_cache_quantization
```

并提供：

```python
quantize_cache_v(...)
dequantize_cache_torch(...)
```

核心思路是：

```text
FP16/BF16 K/V
       │
       ▼
      INT8
       │
       ▼
  KV Cache
```

需要使用时：

```text
INT8 Cache
   │
   ▼
Dequantization
   │
   ▼
Attention
```

这是一种非常直接的 **推理显存优化**。

---

# 17. KV Cache Quantization 的具体方式

代码通过当前数据的：

```text
fmax
fmin
```

计算：

```text
scale
zero point
```

然后：

```python
qdata = torch.clamp(
    fdata / scale + zero,
    qmin,
    qmax
).to(torch.uint8)
```

因此它本质上是一个类似：

\[ q = \mathrm{clip} (x/s+z)
\]

的线性量化。

反量化：

\[ x=s (q-z)
\]

对应：

```python
data = scale * (qdata - zero)
```

---

# 18. KV Cache Quantization 的意义

KV Cache 的规模随着上下文长度增长：

\[ O (L)
\]

其中：

- \(L\)：上下文长度。

对于大模型和长上下文：

```text
KV Cache
   ↓
显存占用越来越大
```

如果使用 8-bit KV Cache：

```text
FP16
 ↓
8-bit
 ↓
大约减少一半存储空间
```

当然，实际收益还要考虑 scale / zero-point 和 kernel 等额外开销。

所以 Qwen-V1 的这一设计体现出一个很明确的工程目标：

> **不仅优化模型本身，还直接优化推理阶段最容易膨胀的 KV Cache。**

---

# 19. KV Cache Kernel

代码还支持：

```python
use_cache_kernel
```

如果相应 CUDA/C++ kernel 存在：

```python
cache_autogptq_cuda_256
```

则可以直接使用自定义 kernel 完成量化 KV Cache 的矩阵计算。

在 Attention 中可以看到：

```python
vecquant8matmul_batched_faster_old(...)
```

以及：

```python
vecquant8matmul_batched_column_compression_faster_old(...)
```

所以这里不仅是：

**KV Cache Quantization**

而且进一步尝试：

**Quantized KV Cache + Custom CUDA Kernel**

从而避免频繁地：

```text
INT8 → FP16 → MatMul
```

---

# 20. Flash Attention

Qwen-V1 也支持 Flash Attention。

代码首先定义：

```python
FlashSelfAttention
```

并根据：

```python
config.use_flash_attn
```

选择是否使用。

Flash Attention 路径会：

```text
Q/K/V
 │
 ▼
Unpadding
 │
 ▼
Flash Attention Kernel
 │
 ▼
Padding Recovery
 │
 ▼
Output
```

对于 batch 中存在 padding 的情况，会先删除无效 token，再调用 Flash Attention。

---

# 21. PyTorch 2.x Scaled Dot-Product Attention

这份代码还具有一个很明显的兼容性设计：

如果：

```python
SUPPORT_TORCH2
```

并且没有使用 KV Cache Quantization，就可以调用：

```python
F.scaled_dot_product_attention(...)
```

所以 Attention 实现实际上有多条路径：

```text
                 Qwen Attention
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
     Flash Attention   SDPA       手写 Attention
          │            │            │
       高性能        PyTorch 2.x    兼容路径
```

这说明这份实现具有很明显的 **工程兼容性和硬件适配意识**。

---

# 22. Dynamic Precision

`QWenLMHeadModel` 会根据硬件自动决定：

```text
BF16
FP16
FP32
```

代码检查：

```python
SUPPORT_BF16
SUPPORT_FP16
SUPPORT_CUDA
```

如果用户没有显式指定精度，就根据硬件自动选择。

大致逻辑：

```text
没有指定 precision
        │
        ▼
   GPU 支持 BF16？
      /     \
    Yes      No
     │        │
    BF16   支持 FP16？
              /   \
            Yes    No
             │      │
            FP16   FP32
```

这主要服务于推理部署。

---

# 23. Embedding 与输出层

模型输入 embedding：

```python
self.wte = nn.Embedding(
    self.vocab_size,
    self.embed_dim
)
```

最终：

```python
self.lm_head = nn.Linear(
    config.hidden_size,
    config.vocab_size,
    bias=False
)
```

因此：

```text
Token ID
   ↓
Embedding
   ↓
Hidden State
   ↓
Transformer
   ↓
Hidden State
   ↓
LM Head
   ↓
Vocabulary Logits
```

需要注意，代码中 `wte` 和 `lm_head` 是两个独立的模块，从当前文件不能直接认为进行了 embedding weight tying。

---

# 24. Gradient Checkpointing

Qwen-V1 支持 gradient checkpointing：

```python
supports_gradient_checkpointing = True
```

训练时：

```python
torch.utils.checkpoint.checkpoint(...)
```

会重新计算部分中间激活，从而降低显存使用。

因此它采用：

> **计算换显存**

的经典训练优化策略。

同时代码明确处理了：

```text
gradient checkpointing
+
use_cache
```

的冲突：

```python
use_cache = True
```

时会关闭 cache。

---

# 25. Qwen-V1 的整体数据流

完整地画出来：

```text
                    Input IDs
                        │
                        ▼
                  Token Embedding
                        │
                        ▼
                  Embedding Dropout
                        │
                        ▼
              ┌─────────────────────┐
              │      QWenBlock       │ × N
              │                     │
              │    RMSNorm          │
              │       ↓             │
              │   Q/K/V Projection  │
              │       ↓             │
              │      RoPE           │
              │       ↓             │
              │   Multi-Head Attn   │
              │       ↓             │
              │   Residual Add      │
              │       ↓             │
              │    RMSNorm          │
              │       ↓             │
              │   Gated MLP         │
              │       ↓             │
              │   Residual Add      │
              └─────────────────────┘
                        │
                        ▼
                    Final RMSNorm
                        │
                        ▼
                     LM Head
                        │
                        ▼
                  Vocabulary Logits
```

---

# 26. 与 DeepSeek-V2 / Kimi K2 的对比

你前面让我分析了两个更现代的模型，现在把三者放在一起会非常直观。

| 模块                  | Qwen-V1      | DeepSeek-V2                | Kimi K2 对应实现           |
|-----------------------|--------------|----------------------------|----------------------------|
| Backbone              | Decoder-only | Decoder-only               | Decoder-only               |
| Norm                  | RMSNorm      | RMSNorm                    | RMSNorm                    |
| Attention             | **MHA**      | **MLA**                    | **MLA**                    |
| KV Compression        | 无           | 有                         | 有                         |
| RoPE                  | 有           | 有                         | 有                         |
| Gated MLP             | 有           | 有                         | 有                         |
| MoE                   | **无**       | 有                         | 有                         |
| Shared Expert         | 无           | 有                         | 有                         |
| Top-K Routing         | 无           | 有                         | 有                         |
| Expert Parallel       | 无           | 有                         | 有                         |
| Flash Attention       | 有           | 有                         | 有                         |
| KV Cache              | 有           | 有                         | 有                         |
| KV Cache Quantization | **有**       | 代码侧重点不同             | 代码侧重点不同             |
| Dynamic NTK           | **有**       | 采用其他 RoPE scaling 机制 | 采用其他 RoPE scaling 机制 |
| LogN Attention        | **有**       | 无明显对应设计             | 无明显对应设计             |

这张表其实非常能说明模型架构的演进方向：

```text
Qwen-V1
   │
   │  传统 Dense Transformer
   │
   ├── MHA
   ├── Gated MLP
   ├── RoPE
   └── KV Cache 优化
   │
   ▼
DeepSeek-V2 / 后续大规模 MoE 模型
   │
   ├── MLA
   ├── MoE
   ├── Shared Experts
   ├── Sparse Routing
   └── Expert Parallelism
```

---

# 27. Qwen-V1 与 DeepSeek-V2 最大的结构差异

可以重点看 Attention 和 FFN：

### Qwen-V1

```text
Attention:
X → Q/K/V → MHA

FFN:
X → Gated MLP
```

### DeepSeek-V2

```text
Attention:
X → Low-rank Q/KV → MLA

FFN:
X → Gate → Top-K Experts
              +
          Shared Expert
```

所以两者的主要区别可以总结成：

> **Qwen-V1 是 Dense Transformer 的工程优化版本，而 DeepSeek-V2 则进一步对 Attention 和 FFN 都进行了结构级改造。**

其中：

```text
Qwen-V1
   └── 优化“怎么更高效地运行 Dense Transformer”

DeepSeek-V2
   ├── MLA → 改造 Attention
   └── MoE → 改造 FFN
```

---

# 28. Qwen-V1 最值得关注的设计思想

虽然 Qwen-V1 没有 MLA 和 MoE 这些后来的“大杀器”，但从代码看，它已经非常明显地在解决三个问题：

## 28.1 长上下文

通过：

```text
RoPE
+
Dynamic NTK
+
LogN Attention
```

解决更长上下文下的位置表示和 Attention scaling 问题。

## 28.2 推理显存

通过：

```text
KV Cache
+
KV Cache Quantization
+
Custom CUDA Kernel
```

降低生成阶段 KV Cache 的内存压力。

## 28.3 计算效率

通过：

```text
Flash Attention
+
PyTorch SDPA
+
BF16 / FP16
```

提高 Attention 和整体推理速度。

所以 Qwen-V1 的核心思想可以总结为：

> **Dense Transformer 本体相对传统，但在长上下文、KV Cache 和计算 kernel 层面进行了大量工程优化。**

---

# 29. 从架构演进角度理解 Qwen-V1

如果把你目前分析的三个模型放在一条时间/技术演进线上：

```text
                 Transformer
                      │
                      ▼
                  Qwen-V1
                      │
          ┌───────────┴───────────┐
          │                       │
       Dense MHA              Gated MLP
          │                       │
          ▼                       ▼
  长上下文 / Cache 优化       Dense 参数计算
          │                       │
          └───────────┬───────────┘
                      ▼
                 DeepSeek-V2
                      │
              ┌───────┴───────┐
              ▼               ▼
             MLA             MoE
              │               │
          KV 压缩          Sparse FFN
              │               │
              └───────┬───────┘
                      ▼
                更大规模模型
```

这个角度非常适合你后面做“大模型架构演进”的学习。

---

# 30. 一句话介绍

如果面试或者汇报时需要快速介绍 Qwen-V1，可以说：

> **Qwen-V1 是一个基于 Decoder-only Transformer 的 Dense 自回归语言模型，采用 RMSNorm、RoPE、Multi-Head Self-Attention 和
Gated MLP 组成基本 Transformer Block；相比单纯的标准 Transformer，它重点在 Dynamic NTK、LogN Attention、Flash Attention 以及
KV Cache Quantization 等方面进行长上下文和推理效率优化，但本身没有采用 MLA 或 MoE。**

---

# 31. 最终总结

从这份代码来看，Qwen-V1 可以浓缩成：

**Dense Transformer + MHA + RoPE + Gated MLP + KV Cache Optimization**

其中：

- **Dense Transformer**：每个 token 都经过同一套 MLP 参数；
- **MHA**：标准多头自注意力；
- **RMSNorm**：用于每个 Transformer Block 前的归一化；
- **RoPE**：提供相对位置信息；
- **Dynamic NTK**：尝试扩展上下文长度；
- **LogN Attention**：长上下文时对 Query scaling；
- **Gated MLP**：提高 FFN 表达能力；
- **KV Cache**：加速自回归生成；
- **KV Cache Quantization**：进一步降低推理显存；
- **Flash Attention / SDPA**：提高 Attention 计算效率；
- **Gradient Checkpointing**：降低训练显存。

如果把它和你前面的 DeepSeek-V2、Kimi K2 放在一起，最核心的区别就是：

```text
Qwen-V1
    ↓
Dense + MHA
    ↓
重点优化“工程效率”

DeepSeek-V2 / Kimi K2
    ↓
MoE + MLA
    ↓
进一步优化“模型结构本身”
    ↓
更大的参数容量
+
更低的激活计算
+
更低的 KV Cache 成本
```

因此，Qwen-V1 很适合作为理解现代大模型架构的一个 **基线模型**：先从它理解完整的 Dense Decoder Transformer、MHA、RoPE、Gated
MLP、KV Cache，再去看 DeepSeek-V2 的 MLA/MoE，会非常顺。
