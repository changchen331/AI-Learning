# Qwen3.8 源码拉取与模型结构对比

> 对比对象：Qwen3.8（27B / 2.4T-A95B）vs 目录内现有开源模型
> —— DeepSeek-V3、DeepSeek-V4（Flash / Pro）、Kimi-K3、GLM-5.2、Qwen3.6（27B / 35B-A3B）
>
> 所有数值均来自各模型 config.json 与建模源码（已逐项核对），代码级机制说明来自 transformers 官方实现及仓库内建模文件。

---

## 0. Qwen3.8 源码来源（本次拉取）

Qwen3.8 的官方 GitHub 仓库 `QwenLM/Qwen3.8` **只包含 README / LICENSE / Issue 模板（公告仓库）**，不含建模代码。实际权重与配置发布在
Hugging Face，建模实现位于 transformers 库。本次拉取/下载到本目录的文件：

| 文件                                                          | 来源                                                             | 说明                                                         |
|---------------------------------------------------------------|------------------------------------------------------------------|--------------------------------------------------------------|
| `Qwen3.8/`（git 克隆）                                        | `https://github.com/QwenLM/Qwen3.8.git`（main 分支，2026-08-17） | 官方公告仓库：README、LICENSE                                |
| `Qwen3.8-27B_config.json`                                     | HF `Qwen/Qwen3.8-27B`                                            | 27B 多模态 dense 配置                                        |
| `Qwen3.8-2.4T-A95B_config.json`                               | HF `Qwen/Qwen3.8-2.4T-A95B`                                      | 2.4T MoE 文本配置                                            |
| `Qwen3.8-27B_modelcard.md` / `Qwen3.8-2.4T-A95B_modelcard.md` | HF 对应仓库 README                                               | 官方模型卡（架构 + 基准）                                    |
| `modeling_qwen3_5.py`                                         | transformers `models/qwen3_5/modeling_qwen3_5.py`                | 27B 实际建模源码（`Qwen3_5ForConditionalGeneration`）        |
| `modeling_qwen3_5_moe.py`（目录已有）                         | transformers `models/qwen3_5_moe`                                | 2.4T-A95B 与 Qwen3.6-35B-A3B 共用（`Qwen3_5MoeForCausalLM`） |

---

## 1. Qwen3.8 结构总览

### 1.1 Qwen3.8-27B（dense 多模态，`Qwen3_5ForConditionalGeneration`）

- 总参数 **27B**（dense，无 MoE）；视觉编码器独立
- 文本： **64 层**，hidden **5120**，词表 **248,320**
- **混合层布局：`16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))`**
    - 48 层 **Gated DeltaNet**（线性注意力）+ 16 层 **Gated Attention**（全注意力，每 4 层 1 个）
- **Gated DeltaNet（线性注意力）**：QK 头 16 / V 头 48，head_dim 128，短卷积核 4，A_log 指数衰减 + sigmoid 写入门控（beta）+
  QK-L2 归一化 + 门控 RMSNorm 输出（SiLU (z)）
- **Gated Attention（全注意力）**：Q 头 24 / KV 头 4（GQA 6:1），head_dim **256**，Q/K 各自 RMSNorm，输出经 sigmoid 门控（
  `attn_output_gate`）
- **FFN**：dense SwiGLU，intermediate **17,408**
- **位置编码**：RoPE theta **10M**，partial rotary 0.25 → 旋转维度 64， **交错 mRoPE**（[T11,H11,W10] 三维网格，适配图文视频）
- **MTP**：`mtp_num_hidden_layers: 1`（多步训练；transformers 代码中无模块，仅忽略权重，推理走 vLLM `qwen3_next_mtp` 投机解码）
- 上下文 **262,144** 原生，可扩展至 **1,000,000**
- 视觉编码器：27 层 ViT，hidden 1152，16 头，patch 16×16、temporal patch 2（3D Conv 支持视频），spatial merge 2×2 → 5120

### 1.2 Qwen3.8-2.4T-A95B（MoE 文本，`Qwen3_5MoeForCausalLM`）

- 总参数 **2.4T / 激活 95B**
- **92 层**，hidden **8192**，词表 **248,320**
- **混合层布局：`23 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))`**
    - 69 层 Gated DeltaNet + 23 层 Gated Attention
- **Gated DeltaNet**：QK 头 16 / **V 头 128**，head_dim 128，其余同 27B
- **Gated Attention**：Q 头 64 / KV 头 4，head_dim 256
- **MoE（每层）**： **512 专家 / 每 token 10 个** + 1 共享专家，expert intermediate **2048**（共享 2048），TopK
  router（softmax + topk 归一化，无 aux loss）
- **位置编码**：RoPE theta 10M，partial rotary 0.25（64 维），纯文本无 mRoPE
- **MTP**：1 层；上下文 262,144（可扩展至 ~1,010,000）

> 与 Qwen3.6 共享同一套 transformers 代码（`qwen3_5` / `qwen3_5_moe`），差异全部体现在 config 与训练上。

---

## 2. 七款模型结构速览表

| 维度            | **Qwen3.8-27B**                                          | **Qwen3.8-2.4T-A95B**         | Qwen3.6-27B                     | Qwen3.6-35B-A3B                    | DeepSeek-V3                                          | DeepSeek-V4 Pro / Flash                                            | Kimi-K3                                                              | GLM-5.2                                                                  |
|-----------------|----------------------------------------------------------|-------------------------------|---------------------------------|------------------------------------|------------------------------------------------------|--------------------------------------------------------------------|----------------------------------------------------------------------|--------------------------------------------------------------------------|
| 架构类          | Qwen3_5ForConditionalGeneration                          | Qwen3_5MoeForCausalLM         | Qwen3_5ForConditionalGeneration | Qwen3_5MoeForConditionalGeneration | DeepseekV3ForCausalLM                                | DeepseekV4ForCausalLM                                              | KimiK3ForConditionalGeneration                                       | GlmMoeDsaForCausalLM                                                     |
| 总参数 / 激活   | 27B / 27B                                                | 2.4T / 95B                    | ~27B / dense                    | ~35B / ~3B                         | ~671B* / ~37B*                                       | 1.6T / 49B · 284B / 13B                                            | 2.8T / 104B                                                          | 未声明                                                                   |
| 层数            | 64                                                       | 92                            | 64                              | 40                                 | 61                                                   | 61 / 43                                                            | 93                                                                   | 78                                                                       |
| hidden_size     | 5120                                                     | 8192                          | 5120                            | 2048                               | 7168                                                 | 7168 / 4096                                                        | 7168                                                                 | 6144                                                                     |
| 注意力类型      | **混合**：3 线性 + 1 全注意力                            | **混合**：3 线性 + 1 全注意力 | 同 Qwen3.8-27B                  | 混合：3 线性 + 1 全注意力          | **纯全注意力**（MLA，softmax 全部历史）              | **混合稀疏**：滑窗 128 + 压缩 KV top-k                             | **混合**：69 KDA + 24 Gated MLA                                      | **全层 DSA 稀疏 top-k**（无全注意力层）                                  |
| 线性/稀疏注意力 | Gated DeltaNet（卷积 4、beta 门控、QK-L2）               | Gated DeltaNet（V 头 128）    | Gated DeltaNet                  | Gated DeltaNet（V 头 32）          | 无                                                   | 可学习门控池化 Compressor（ratio 4/128）+ learned Indexer          | **KDA**（DeltaNet 变体，短卷积 4、full-rank gate、AttnRes 每 12 层） | **DSA Indexer**（轻量投影 + ReLU + top-k 2048，IndexShare 每 4 层 1 次） |
| 全注意力        | Gated MLA 风格（24Q/4KV，head 256，Q/K norm + 输出门控） | 64Q/4KV head 256              | 24Q/4KV head 256                | 16Q/2KV head 256                   | MLA：128 头、q_lora 1536、kv_lora 512、qk 192+rope64 | MLA-512：KV 头=1、head_dim 512、q_lora 1024/1536、o_lora 1024 分组 | Gated MLA：96 头、q_lora 1536、kv_lora 512、输出门控                 | MLA：64 头、q_lora 2048、kv_lora 512、qk 256（nope 192+rope 64）         |
| 位置编码        | RoPE 10M，partial 0.25，**交错 mRoPE**                   | RoPE 10M，partial 0.25        | RoPE 10M，partial 0.25，mRoPE   | RoPE 10M，partial 0.25，mRoPE      | RoPE 10k + **YaRN ×40**（160K）                      | RoPE 10k + YaRN ×16 + compress_theta 160k（1M）                    | 文本无显式 RoPE 应用；视觉 2D RoPE                                   | RoPE **8M**，interleaved，partial 64/256                                 |
| FFN / MoE       | dense 17408                                              | MoE 512/10 + 1 shared（2048） | dense 17408                     | MoE 256/8 + 1 shared（512）        | 前 3 层 dense 18432 + MoE 256/8 + 1 shared           | 全 MoE：384/6 · 256/6 + 1 shared；前 3 层 hash 路由                | 1 dense + MoE **896/16 + 2 shared**（Latent 3584）                   | 前 3 层 dense 12288 + MoE 256/8 + 1 shared                               |
| 路由方式        | —                                                        | TopK softmax，无 aux          | —                               | TopK softmax                       | sigmoid + 分组 topk（8 组×4）+ noaux_tc              | sqrtsoftplus + noaux_tc；hash 前 3 层                              | sigmoid + correction bias + LatentMoE + noaux_tc                     | sigmoid + correction bias + noaux_tc                                     |
| 归一化          | RMSNorm 1e-6                                             | RMSNorm 1e-6                  | RMSNorm 1e-6                    | RMSNorm 1e-6                       | RMSNorm 1e-6                                         | RMSNorm 1e-6 + **mHC 超连接**（Sinkhorn×4）                        | RMSNorm 1e-5                                                         | RMSNorm 1e-5                                                             |
| 上下文          | 262K（→1M）                                              | 262K（→1.01M）                | 262K（→1M）                     | 262K（→1.01M）                     | 160K                                                 | **1M**                                                             | **1M**                                                               | **1M**                                                                   |
| MTP             | 1 层（多步）                                             | 1 层（多步）                  | 1 层                            | 1 层                               | 1 层                                                 | 1 层（代码已实现）                                                 | **无**（num_nextn=0）                                                | 1 层                                                                     |
| 多模态          | 视觉（图+视频）                                          | 纯文本                        | 视觉                            | 视觉                               | 纯文本                                               | 纯文本                                                             | 视觉（MoonViT-V2）+ 视频                                             | 纯文本                                                                   |
| vocab           | 248320                                                   | 248320                        | 248320                          | 248320                             | 129280                                               | 129280                                                             | 163840                                                               | 154880                                                                   |
| 量化            | bf16                                                     | bf16                          | bf16                            | bf16                               | FP8（e4m3）                                          | FP8 + **FP4 专家**（e4m3/ue8m0）                                   | **MXFP4 权重 / MXFP8 激活**（QAT）                                   | bf16                                                                     |

*V3 参数量为目录内 readme 中 V3.2 基准值（671B/37B），V3 自身 config 未声明。

---

## 3. 逐模型差异分析

### 3.1 vs Qwen3.6（同门：结构完全同源，Qwen3.8 是"训练升级"而非"结构升级"）

对 config 逐字段 diff 的结果：

- **Qwen3.8-27B 与 Qwen3.6-27B：差异仅 1 个字段（`transformers_version` 4.57.1 → 5.8.0.dev0）**。层数、hidden、注意力头、Gated
  DeltaNet 参数、mRoPE、MTP、视觉编码器 **全部一致**。也就是说 27B 这一档 Qwen3.8 与 Qwen3.6 **结构完全相同**，官方口径也是
  "Built on the architectural foundation of Qwen3.5"——提升主要来自预训练/后训练数据与 RL。
- **Qwen3.8-2.4T-A95B 相对 Qwen3.6-35B-A3B 是"同一架构放大 + 去多模态"**：
    - 共用 `Qwen3_5Moe` 建模代码（TopK router + shared expert sigmoid gate + Gated DeltaNet）；
    - 2.4T：92 层 / hidden 8192 / 64Q-4KV / 512 专家·10 topk / 线性 V 头 128（35B-A3B 为 40 层 / 2048 / 16Q-2KV / 256 专家·8
      topk / V 头 32）；
    - 2.4T 为纯文本（无 vision_config），35B-A3B 带视觉编码器且启用 mRoPE。

**结论：Qwen3.8 家族 = Qwen3.5/3.6 的 Gated DeltaNet + Gated Attention 混合架构的延续，27B 与 Qwen3.6-27B 零结构差异。**

### 3.2 vs DeepSeek-V3 / V4

| 对比点       | Qwen3.8                                                           | DeepSeek-V3                           | DeepSeek-V4                                                            |
|--------------|-------------------------------------------------------------------|---------------------------------------|------------------------------------------------------------------------|
| 注意力流派   | **线性注意力（Gated DeltaNet）**，全注意力仅 1/4 层               | **纯全注意力**（MLA，全序列 softmax） | **稀疏注意力**（滑窗 + 压缩 KV top-k），非线性注意力                   |
| KV 处理      | 线性层无 KV cache（recurrent state）；全注意力层标准 GQA KV cache | MLA 低秩潜变量（kv_lora 512）压缩缓存 | 单 KV 头 MLA-512 + **逐层门控池化压缩**（ratio 4/128）+ Indexer 选位置 |
| 长上下文手段 | 线性注意力把复杂度降到线性，配合 partial rotary                   | YaRN 拉伸（160K）                     | 压缩 KV + 稀疏 top-k（1M），KV cache 仅为 V3.2 的 10%                  |
| MoE          | 512/10 topk，TopK softmax                                         | 256/8 + 分组路由（8 组×4）            | 256–384/6 + hash 路由前 3 层 + sqrtsoftplus                            |
| 残差/其他    | 标准 Pre-Norm                                                     | 标准 Pre-Norm                         | **mHC 流形超连接**（每层 4 份状态 + Sinkhorn）                         |

核心差异： **Qwen3.8 用"混合线性注意力"解决长上下文，DeepSeek-V4 用"稀疏注意力 + KV 压缩"解决，DeepSeek-V3 则完全不解决（160K
封顶）**。Qwen3.8 的 Gated DeltaNet 属于 DeltaNet 家族（可学习衰减 + 写入门控），与 DeepSeek 的 MLA/DSA 是完全不同的两条技术路线。

### 3.3 vs Kimi-K3

两者是 **最接近的"同类"**（都是混合线性注意力 + 超大 MoE），但实现细节差异明显：

| 对比点     | Qwen3.8-2.4T-A95B                               | Kimi-K3                                                                       |
|------------|-------------------------------------------------|-------------------------------------------------------------------------------|
| 线性注意力 | Gated DeltaNet（QK 16 头 / V 128 头，head 128） | **KDA**（DeltaNet 变体，96 头 head 128，short conv 4，full-rank gate，QK-L2） |
| 层排布     | 3 线性 + 1 全注意力（每 4 层）                  | 3 线性（KDA）+ 1 全注意力（Gated MLA）**+ 尾部两层连续 MLA**                  |
| 附加机制   | 无                                              | **AttnRes**：每 12 层一个跨层注意力残差块（可学习加权历史前缀和）             |
| 全注意力   | GQA 64Q/4KV + Q/K norm + 输出门控               | Gated MLA（96 头、kv_lora 512、输出门控，**无 GQA 分组**）                    |
| MoE        | 512/10 topk + 1 shared                          | **896/16 + 2 shared + Stable LatentMoE**（专家计算在 3584 维 latent 空间）    |
| 激活       | SwiGLU                                          | **SiTU-GLU**（beta·tanh(gate/beta)·sigmoid(gate)·up）                         |
| 上下文     | 262K                                            | 1M                                                                            |
| MTP        | 1 层                                            | 无                                                                            |
| 多模态     | 纯文本                                          | 原生视觉 + 视频                                                               |

要点：Kimi-K3 把"稀疏化"推到更极致（896 专家只激活 16 个 + LatentMoE 降维），并多了 AttnRes 跨层残差与 SiTU 激活；Qwen3.8
则保持每层 MoE 512 专家/10 激活，靠 2.4T 规模与 MTP 提效。

### 3.4 vs GLM-5.2

| 对比点     | Qwen3.8                          | GLM-5.2                                                                                                                 |
|------------|----------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| 注意力流派 | 线性注意力（Gated DeltaNet）为主 | **全层 DSA top-k 稀疏注意力**（softmax 但只对 top-2048 个 key）——即 DeepSeek Sparse Attention 路线，与 DeepSeek-V4 同源 |
| 长上下文   | 线性复杂度 + 262K                | 稀疏 top-k + 1M；IndexShare 每 4 层跑 1 次完整索引器（21 full / 57 shared），1M 下 FLOPs 降 2.9×                        |
| 全注意力   | 1/4 层 GQA                       | 无独立全注意力层（全部稀疏）                                                                                            |
| MoE        | 512/10 + 1 shared                | 256/8 + 1 shared，前 3 层 dense                                                                                         |
| 位置编码   | partial rotary 0.25 + mRoPE      | partial rotary 0.25（interleaved），theta 8M                                                                            |
| MTP        | 1 层（投机解码）                 | 1 层 + **索引跨 MTP 迭代共享**（投机接受率 +20%）                                                                       |

要点：GLM-5.2 代表"稀疏注意力"阵营（DeepSeek 系），Qwen3.8 代表"线性注意力"阵营；两者都在追求 1M 级上下文下的计算/缓存效率，但机制正交。

---

## 4. 五大技术流派归纳

目录内 7 款模型可归为 4 条技术路线：

1. **Gated DeltaNet 混合线性注意力流（Qwen 系）**：Qwen3.6 / Qwen3.8 全部变体。3:1 线性:全注意力层排布，Gated DeltaNet
   负责线性复杂度，每 4 层 1 个 Gated Attention 保强检索能力；MTP 提速。上下文 262K 起步。
2. **MLA 全注意力流（DeepSeek-V3）**：纯 softmax 全序列注意力 + 低秩 KV 潜变量压缩，靠 YaRN 撑到 160K。结构最"保守"。
3. **稀疏注意力 + KV 压缩流（DeepSeek-V4 / GLM-5.2）**：滑窗 + 压缩 KV + learned Indexer top-k 稀疏化，1M 上下文；GLM-5.2
   进一步做跨层索引共享。
4. **KDA 线性注意力 + 极致稀疏 MoE 流（Kimi-K3）**：DeltaNet 变体 KDA + AttnRes，896 专家仅激活 16 个 + Stable LatentMoE +
   SiTU-GLU，2.8T 规模，1M 上下文。

**共性趋势**：① 全部采用 RMSNorm + Pre-Norm；② 全部走向"混合/稀疏"而非纯 dense 注意力，以支撑超长上下文；③ MoE 成为大模型标配（仅
27B 档 dense）；④ MTP 普遍存在（仅 Kimi-K3 未启用）；⑤ RoPE 都做部分旋转（64 维左右）；⑥ 规模 2.4T–2.8T 成为旗舰档，激活参数
95–104B。

---

## 5. 关键结论

1. **Qwen3.8 的结构基因 = Qwen3.5/3.6**：`Qwen3.8-27B` 与 `Qwen3.6-27B` 的 config **逐字段一致**（仅 transformers 版本号不同）；
   `2.4T-A95B` 是同一 MoE 架构的放大版（92 层/512 专家/95B 激活），并去掉了视觉。 **Qwen3.8
   的进步主要在训练/数据/RL，而非网络结构。**
2. **与目录内其他模型相比，Qwen3.8 属于"线性注意力"路线**：Gated DeltaNet（卷积 4 + 可学习衰减 + 写入门控 + QK-L2 norm + 门控
   RMSNorm）是其核心创新点，全注意力仅占 1/4 层；这与 DeepSeek-V3 的纯 MLA 全注意力、DeepSeek-V4 / GLM-5.2 的 DSA
   稀疏注意力形成鲜明对照。
3. **长上下文方案分阵营**：Qwen3.8（线性注意力，262K 原生）、DeepSeek-V4 / GLM-5.2 / Kimi-K3（稀疏/压缩，1M）、DeepSeek-V3（仅
   160K）。
4. **效率取向差异**：Kimi-K3 稀疏化最激进（896 专家/16 激活 + LatentMoE + 原生 MXFP4），DeepSeek-V4 引入 mHC 超连接与 FP4
   专家，Qwen3.8 靠 MTP + Gated DeltaNet 低 KV 开销。
5. **MTP 已成为 Qwen/DeepSeek/GLM 的标配提速件**，仅 Kimi-K3 未启用（其用 AttnRes 补偿）。
