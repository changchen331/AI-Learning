# ARCHITECTURE.md — bert_text_matching 技术方案文档

## 一、项目定位

**场景：** 中文语义文本匹配（Semantic Text Matching）

> 给定两句话，判断它们是否表达相同意思——这是搜索引擎问句去重、智能客服意图匹配、RAG 向量检索的核心子问题。

**教学目标：** 对比"表示型"与"交互型"两种 BERT 文本匹配范式，以及 LLM zero-shot 和 LLM SFT（LoRA 指令微调），形成判别式 vs
生成式的完整对比视角。

### 三套实现对比

| 维度          | BiEncoder（表示型）               | CrossEncoder（交互型）  | LLM SFT（生成式）          |
|---------------|-----------------------------------|-------------------------|----------------------------|
| 核心思路      | 两句分别编码 → 余弦相似度         | 两句拼接全层交互 → 分类 | 生成【相似】/【不相似】    |
| 是否可预计算  | ✓ 向量可离线，适合大规模检索     | ✗ 每对重新计算         | ✗                         |
| AFQMC 实测 F1 | **0.6765**（cosine, 3 epoch）     | 0.6750（3 epoch）       | 0.5556（5K 平衡, 3 epoch） |
| 推理速度      | 毫秒级（预计算后点积）            | 数十毫秒                | ~1 秒/条（generate）       |
| 可训练参数    | 全量（45.6M，4层）                | 全量（45.6M，4层）      | LoRA 0.22%（1.08M）        |
| 训练 Loss     | CosineEmbeddingLoss / TripletLoss | CrossEntropyLoss        | 生成 loss（loss masking）  |
| 典型应用      | RAG Recall、向量数据库            | Reranker 精排           | 灵活扩展新类型、本地离线   |

---

## 二、整体流水线

```
原始数据（AFQMC JSONL）
        │
        ▼
  explore_data.py          ← 数据分布、长度统计、length-bias 检测
        │
        ▼
  [表示型]                  [交互型]
  PairDataset               CrossEncoderDataset
  TripletDataset            [CLS] s1 [SEP] s2 [SEP]
        │                           │
        ▼                           ▼
  BiEncoder                   CrossEncoder
  shared BERT(4层)             BERT(4层)
  池化 → L2 归一化              CLS → Linear(2)
  encode(s1), encode(s2)               │
        │                             │
  CosineEmbeddingLoss         CrossEntropyLoss
  TripletLoss                          │
        │                             │
        ▼                             ▼
  train_biencoder.py          train_crossencoder.py
        │                             │
        ▼                             ▼
  evaluate.py（阈值搜索）      evaluate.py（argmax）
        │                             │
        └──────────┬──────────────────┘
                   ▼
  src_llm/llm_compare.py   ← DashScope API zero-shot 对比
                   │
                   ▼
  src_llm/train_sft.py     ← LoRA 指令微调（chat格式 + loss masking + --full_ft）
                   │
                   ▼
  src_llm/evaluate_sft.py  ← 加载 checkpoint，Accuracy/F1 多方对比
```

---

## 三、各环节技术选型

### 3.1 数据集：AFQMC

**选型理由：**

- 蚂蚁金融问句匹配，真实业务数据，场景明确
- 句子极短（均值 13.4 字），max_length=64 覆盖 99.9%，训练快
- 正负比 31:69，接近真实业务分布，有类别不均衡教学价值
- LCQMC / BQ Corpus 保留为学生自主练习数据集

**已知限制：**

- CLUE 竞赛格式：test 集标签未公开（-1 占位），实际评估用 validation 集

### 3.2 BiEncoder：Sentence-BERT 架构

**池化策略：** 默认 `mean`（Sentence-BERT 论文结论：mean pooling 在语义相似度任务优于 CLS）

**L2 归一化：** `encode()` 输出已归一化，余弦相似度 = 点积，可直接用 FAISS IndexFlatIP 做向量检索

**层数限制：** 默认 4 层（课堂快速跑通），全量 12 层留给学生自行对比

```
BiEncoder 4层  ≈  45.6M 参数  ≈  全量的 38%
```

### 3.3 损失函数对比

| Loss                | 输入格式           | 核心公式                                | margin 作用                         |
|---------------------|--------------------|-----------------------------------------|-------------------------------------|
| CosineEmbeddingLoss | (s1, s2, ±1)       | 正例拉近到 sim≥0，负例推远到 sim≤margin | 负例的"安全距离"，超出后梯度为 0    |
| TripletLoss         | (anchor, pos, neg) | sim(a,p) - sim(a,n) > margin            | 要求正例与负例的相似度差超过 margin |

**TripletDataset 构建：**

- 扫描所有正样本对（10,573 对），每个 anchor 配一个负样本
- 优先找 anchor 自身的负样本；若无，从全局池随机选
- 进阶（学生练习）：Online Hard Negative Mining，同 batch 内选最难负样本

### 3.4 CrossEncoder

- 输入：`[CLS] s1 [SEP] s2 [SEP]`，token_type_ids 区分两句
- 头部：`Linear(768, 2)` + CrossEntropyLoss
- 与 rag_annual_report 中的 CrossEncoder Reranker 是同一架构

### 3.5 评估策略（全部方案统一口径）

所有方案最终均输出二元预测（相似=1 / 不相似=0），使用相同的 Accuracy + F1（weighted）评估，数字可直接横向比较，不存在 NER
项目那样的评估方法论差异。

| 方案                            | 输出形式                | 转为 label 的方式                         |
|---------------------------------|-------------------------|-------------------------------------------|
| BiEncoder                       | 余弦相似度（0~1 浮点）  | 阈值搜索，val 上取 F1 最高阈值            |
| CrossEncoder                    | logits（2维）           | argmax → 0/1                              |
| LLM zero-shot（llm_compare.py） | 文本"是"/"否"           | 含"是"→1，含"否"→0，其余→parse_fail       |
| LLM SFT（evaluate_sft.py）      | 文本【相似】/【不相似】 | 含"不相似"→0，含"相似"→1，其余→parse_fail |

### 3.6 LLM SFT：LoRA 指令微调

#### 文本匹配的 chat 格式

```
system: 你是语义匹配助手，判断两句话语义是否相同，只输出【相似】或【不相似】。
user:   句子A：{sentence1}\n句子B：{sentence2}\n是否相似：
TARGET: 【相似】 / 【不相似】   ← 仅 3~5 token，是三类任务中最短的
```

**与分类和 NER 任务的对比**：

| 维度            | 文本分类            | 文本匹配（本任务）  | NER                      |
|-----------------|---------------------|---------------------|--------------------------|
| TARGET token 数 | 1~2（类别名）       | 3~5（相似/不相似）  | 20~150（JSON）           |
| max_length 建议 | 128                 | 128                 | 256                      |
| loss 集中程度   | 极高                | 极高                | 分散                     |
| 评估标准差异    | 无（直接 Accuracy） | 无（直接 Accuracy） | 有（seqeval vs span F1） |

#### 类别不均衡与平衡采样（重要）

AFQMC 训练集正负比为 31:69。若直接随机采样训练，模型会快速学到"一直输出【不相似】"以最小化 loss（负例多，此策略的 token-level
loss 最低）。表现为 Accuracy ≈ 0.69，F1 (pos) ≈ 0，这与 CrossEncoder 1-epoch 的退化现象完全一致。

**解决方案**：`train_sft.py` 默认开启正负平衡采样，从正负样本中各取 `num_train//2` 条。

```python
n_each = min(args.num_train // 2, len(pos), len(neg))
train_raw = random.sample(pos, n_each) + random.sample(neg, n_each)
```

#### LoRA 配置与实测结果（平衡采样 5000 条，3 epoch，RTX 4060 Laptop）

配置与分类/NER 任务相同：Qwen2-0.5B，r=8，target `q/k/v/o_proj`，可训练参数 ~1.08M / 495M（0.22%）。

| 指标                        | 数值                       |
|-----------------------------|----------------------------|
| Accuracy（200 条 val 子集） | 0.6400                     |
| F1 (weighted)               | 0.6535                     |
| **F1 (正例)**               | **0.5556**                 |
| parse_fail                  | **0 条（0%）**             |
| 训练时间                    | ~55 min（3×1110s）         |
| 推理速度                    | ~1.0 s/条（GPU，generate） |

**train/val loss 曲线**（每 epoch）：

| epoch | train_loss |               val_loss |
|------:|-----------:|-----------------------:|
| **1** |      0.162 | **0.122**（最优 ckpt） |
|     2 |      0.129 |                  0.123 |
|     3 |      0.106 |    0.129（轻微过拟合） |

**关键观察**：val_loss 在 epoch 1 已到底部，2/3 epoch train_loss 继续下降但 val_loss 反弹。文本匹配 TARGET 只有 3~5
token，学习信号高度集中， **1 epoch 已足够**，多跑反而过拟合。后续若复现，可直接 `--epochs 1` 节省 ~37 min。

---

## 四、实验结果与方法对比

> 实验条件：AFQMC validation 集（4,316 条），4 层 BERT，3 epoch，batch=32，RTX 4060 Laptop GPU。SFT 部分另在 200 条 val 子集上评估。

### 4.1 量化指标

| 方法                                     |   Accuracy | F1(weighted) | F1(正例) | 决策方式                | 训练时长 |
|------------------------------------------|-----------:|-------------:|---------:|-------------------------|----------|
| BiEncoder + CosineEmbeddingLoss          |     0.6735 |   **0.6765** |   0.6765 | threshold=0.51（val）   | ~12 min  |
| BiEncoder + TripletLoss                  |     0.6664 |       0.6599 |   0.6599 | threshold=0.81（val）   | ~5.6 min |
| CrossEncoder + CrossEntropyLoss          | **0.6905** |       0.6750 |   0.6750 | argmax                  | ~9.5 min |
| Qwen2-0.5B SFT（LoRA，5K 平衡，3 epoch） |     0.6400 |       0.6535 |   0.5556 | 生成【相似】/【不相似】 | ~55 min  |

> 这套数字是 3 epoch 默认配置一次跑通的结果，可直接复现：`python train_biencoder.py --loss cosine` 等等。

### 4.2 结果解读

**CrossEncoder Acc 最高（0.6905），但 F1 最高的是 BiEncoder Cosine（0.6765）**

3 epoch 训练后，CrossEncoder 的 F1 已经基本追上 BiEncoder（0.6750 vs 0.6765）， 不再是 1 epoch 时"Acc 虚高 / F1 极低"的退化状态。但
Accuracy 仍领先 BiEncoder Cosine 约 1.7 个点， 说明它在多数类（负类）上预测更稳健——当类别不均衡时，CrossEncoder
的全层交互更容易把握"明显的不相似"。

**对比 1-epoch 旧实验**（教学反差点保留参考）：

- 1 epoch CrossEncoder：Acc=0.6921 / F1=0.5703，正类 recall 极低，模型倾向全预测负类
- 3 epoch CrossEncoder：Acc=0.6905 / F1=0.6750，正类被学到，F1 跳升 0.10

这条对比直接给出"训练不足 → Accuracy 虚高 / F1 偏低"的教学样本， **accuracy 并不总能反映模型真实能力**，尤其在类别不均衡 +
训练不足时。

**CosineEmbeddingLoss 优于 TripletLoss**（F1 差 0.0166）

AFQMC 正样本仅 10K 条，TripletDataset 构造的三元组同样只有 10K 个， 训练信号偏少。Triplet 的 train_loss 从 0.068 在 epoch 1
末就降到 0.015，epoch 2/3 几乎不动（margin 容易"打满"）， val_f1 也在 epoch 2 见顶（0.6599）后微跌。数据量更大的 LCQMC（238K 对）上
Triplet 的优势会更明显。

**TripletLoss 阈值偏高（0.81）**

TripletLoss 只约束正例相对负例更近，并不直接要求绝对相似度的量级， 导致所有嵌入在相似度轴上整体向高值偏移，最优阈值因此明显高于
CosineEmbeddingLoss（0.51）。

**判别式 vs 生成式：BERT 在文本匹配上仍然占优**

同样 ~10 min 左右训练时间，BiEncoder Cosine F1 (正例) = 0.6765，比 LoRA SFT 的 0.5556 高 12 个点， 而 SFT 还多花了 5 倍时间（55
min）。文本匹配 TARGET 极短（3~5 token）且标签集封闭（二分类）， 判别式架构能把全部容量用于"两句话是否相似"这一信号，不需要承担生成任意中文的负担。
SFT 的真正价值不在精度，而在于： (1) 灵活扩展新类别（如三分类）； (2) 离线本地推理无需 API 成本。

### 4.3 消融方向（学生练习建议）

| 变量       | 候选值              | 实测/预期结论                                                     |
|------------|---------------------|-------------------------------------------------------------------|
| BERT 层数  | 4 / 8 / 12          | 层数越多精度越高，速度线性下降（本次仅跑 4 层）                   |
| 训练 epoch | 1 / 3 / 5           | CrossEncoder 1→3 epoch F1 跳升 0.10，需 3 epoch 才能体现真实水平  |
| 池化策略   | cls / mean / max    | mean 通常最优（Sentence-BERT 结论）                               |
| margin     | 0.1 / 0.3 / 0.5     | 见下方 bad case 分析                                              |
| 负采样     | 随机 vs Online Hard | LCQMC 上对比效果更明显                                            |
| SFT epoch  | 1 / 3 / 5           | val_loss 在 epoch 1 见底，本次 3 epoch 已轻微过拟合，建议 1 epoch |

---

## 五、Bad Case 分析

> 基于 BiEncoder + CosineEmbeddingLoss（threshold=0.51，3 epoch）在 validation 集上的错误分析。
> 总错误 1409 条，错误率 32.6%（acc 0.6735）。

### 5.1 错误分布

| 错误类型                         | 数量 | 占总错误 | 说明                       |
|----------------------------------|-----:|---------:|----------------------------|
| FP 假阳性（预测相似，实为不同）  |  752 |      53% | 模型过度自信"相似"         |
| └─ 高置信度错误（Δscore > 0.15） |  155 |      11% | 问题最严重，离阈值远       |
| └─ 临界错误（Δscore ≤ 0.15）     |  597 |      42% | 接近阈值，调整阈值可改善   |
| FN 假阴性（预测不同，实为相似）  |  657 |      47% | 模型错过真实相似对         |
| └─ 高置信度错误（Δscore > 0.15） |  132 |       9% | 需改进模型表示能力         |
| └─ 临界错误（Δscore ≤ 0.15）     |  525 |      37% | 接近阈值，调阈值可部分改善 |

**FP 数量略多于 FN**，与之前 1-epoch 实验"FN 远多于 FP"的局面相反。3 epoch 训练后阈值从 0.55 降到 0.51，
模型对"相似"的判定更激进，吃掉了一部分原本判错的 FN，代价是产生更多 FP。 本质是同一组分数 + 不同切点的权衡。

> 79% 的错误（11%+9% = 20% 高置信度 + 42%+37% = 79% 临界）分布在阈值附近 ±0.15 区间，
> 阈值微调能改善这部分；高置信度错误才是真正需要靠模型/数据来解决的。

### 5.2 语言特征对比

| 特征              | FP 假阳性 | FN 假阴性 | 含义                       |
|-------------------|----------:|----------:|----------------------------|
| 字符 Jaccard 均值 | **0.484** |     0.380 | FP：词汇高度重叠但语义不同 |
| 句子长度差均值    |    3.4 字 |    4.8 字 | FN 句对表述方式差异略大    |

这两个数字直接指向不同的根因和对应优化方向。即使错误数量相对发生了反转， 两类错误的"语言指纹"仍然清晰： **FP 是"换字不换意"
陷阱，FN 是"换说法"挑战**。

### 5.3 典型 Bad Case

**FP 高置信度（score 极高，但标签为不相似）**

```
score=0.967  "【蚂蚁借呗】你支付宝 ******@qq.com 提交的蚂蚁借呗申请经综合评估暂未通过..."
             "【蚂蚁借呗】你支付宝 hys***@***.com 提交的蚂蚁借呗申请经综合评估暂未通过..."
→ 仅账号不同，模板内容完全一样。数据质量问题：这两条标注为"不相似"值得存疑。

score=0.900  "花呗能参加优惠活动吗"  ||  "花呗能参加购物优惠活动吗"
→ 只多了"购物"二字，语义接近，但标注为不相似。边界模糊的标注难例。
```

**FP 临界（在阈值 0.51 附近）**

```
score=0.586  "支付宝里找不到花呗"      ||  "怎么找不到花呗页面"
score=0.614  "怎么用蚂蚁借呗"          ||  "怎么样才可以用借呗"
score=0.568  "怎么查看花呗的消费明细"  ||  "已自动还款的花呗明细怎么查看"
→ 这类样本主观看是相似的，但标注为不相似。3 epoch 模型把它们拉到阈值之上反而被算成 FP，
   暴露 AFQMC 标注本身在边界上的不一致。
```

**FN 高置信度（score 极低，但标签为相似）**

```
score=0.082  "我现在的花呗是否全部还款完毕"      ||  "我在花呗有***百元，消费***元，还款是否***元"
→ 表达差异极大，脱离了"还款情况查询"这个共同语义。

score=0.157  "花呗使用不了"                     ||  "花呗被冻结"
→ 因果同源但词汇完全不重叠，模型未能从"使用不了 = 被冻结"建立等价关系。

score=0.197  "花呗可以用来支付车费吗"            ||  "我一直滴滴打车都可以用花呗"
→ 一个是问句，一个是陈述句，且"车费"与"滴滴打车"需要常识链接。
```

**FN 临界**

```
score=0.405  "使用花呗有风险吗"  ||  "花呗安全吗"
→ 完全同义，但词汇只共享"花呗"。对完全换词的改写句不鲁棒。

score=0.379  "之前有逾期账单，什么时候才能用花呗"  ||  "花呗逾期后多久能用"
→ 同一问题的不同表述，长度差大、连接词不同，模型还差一点拉过阈值。
```

### 5.4 数据质量观察

部分高 FP 案例（如 score=0.967 的两条仅账号不同的系统消息）标注为"不相似"存在争议，
临界区也有不少主观判断会标"相似"但金标记是"不相似"的样本。 说明 AFQMC 训练集存在一定比例的标注噪声。这是真实业务数据的常态，
也是为什么模型在 val 集上 F1 有天花板（~0.68）的原因之一——继续优化时要分清是模型问题还是标注问题。

---

## 六、优化方向

### 6.1 数据层面

| 方向         | 针对问题               | 实施方式                                        |
|--------------|------------------------|-------------------------------------------------|
| 难负样本挖掘 | FP：词汇重叠高的负例   | 用训练好的 BiEncoder 找 top-K 相似但标签=0 的对 |
| 正样本扩充   | TripletLoss 三元组不足 | LLM API 对正例做同义改写，扩充到 3x             |
| 跨数据集预热 | 模型语义泛化不足       | 先在 LCQMC（238K）训练，再迁移到 AFQMC          |

### 6.2 模型层面

| 方向                                    | 针对问题                       | 预期收益                                                                       |
|-----------------------------------------|--------------------------------|--------------------------------------------------------------------------------|
| 增加层数（4→12）                        | FN：深层语义理解不足           | val F1 预计提升 3~8 个点（4 层 3 epoch 已到 0.6765，全量 12 层进一步空间有限） |
| 加大 margin（0.3→0.5）                  | FP：词汇重叠造成的高置信假阳性 | FP 数量减少，但可能增加 FN                                                     |
| 金融领域预训练模型（FinBERT / MacBERT） | AFQMC 领域术语理解             | 初始化更好，收敛更快                                                           |

### 6.3 训练策略层面

| 方向                        | 描述                                            | 适用场景                      |
|-----------------------------|-------------------------------------------------|-------------------------------|
| SimCSE 对比学习             | 同一句 dropout 两次为正例，batch 内其他句为负例 | FN 多（同义异词）时效果最明显 |
| Online Hard Negative Mining | 同 batch 内找最难负例，替代随机负采样           | TripletLoss 数据量大时        |
| 调整 epoch                  | BERT 系列 3 epoch 已接近收敛；SFT 1 epoch 即可  | 见 4.3 消融                   |

### 6.4 工程与部署层面

| 方向                        | 描述                                     | 立即可做                                                                 |
|-----------------------------|------------------------------------------|--------------------------------------------------------------------------|
| 两阶段级联（Recall→Rerank） | BiEncoder 召回 Top-K → CrossEncoder 精排 | 当前两个 checkpoint 直接组合，无需重训                                   |
| 阈值校准                    | val 集正负比（31:69）≠ 线上分布          | 按实际流量分布重新搜最优阈值；本次 cosine 阈值 0.51 接近 0.5，迁移成本低 |
| 学生消融实验                | 4/12 层 × 1/5 epoch 的 2×2 对比          | 代码无需改动，只改命令行参数                                             |

---

## 七、目录结构

```
bert_text_matching/
├── src/                         # BERT 判别式实现
│   ├── download_data.py         # 三个数据集下载（AFQMC/LCQMC/BQ）
│   ├── explore_data.py          # 数据探索与可视化（4 张图）
│   ├── dataset.py               # PairDataset / TripletDataset / CrossEncoderDataset
│   ├── model.py                 # BiEncoder / CrossEncoder + 工厂函数
│   ├── train_biencoder.py       # 表示型训练（--loss cosine/triplet）
│   ├── train_crossencoder.py    # 交互型训练
│   ├── evaluate.py              # 评估工具 + 独立评估脚本
│   ├── compare_methods.py       # 方法效果对比 + 可视化
│   └── analyze_badcases.py      # FP/FN 分析 + 优化方向输出
│
├── src_llm/                     # LLM 生成式实现
│   ├── llm_compare.py           # DashScope API zero-shot 对比（结果保存到 logs/）
│   ├── train_sft.py             # LoRA 指令微调（--full_ft 开关，chat格式 + loss masking）
│   └── evaluate_sft.py          # 加载 checkpoint，Accuracy/F1 多方对比
│
├── data/
│   ├── afqmc/                  # 训练用数据集（train/validation/test.jsonl）
│   ├── lcqmc/                  # 学生自主练习
│   └── bq_corpus/              # 学生自主练习
│
├── outputs/
│   ├── checkpoints/            # 最优 checkpoint（biencoder_cosine_best.pt 等）
│   ├── figures/                # 数据探索图 + 相似度分布图
│   ├── logs/                   # 训练日志 JSON + llm_compare_results.json + sft_results.json
│   ├── sft_adapter/            # LoRA adapter（adapter_model.safetensors + adapter_config.json）
│   └── sft_full_ckpt/          # 全量微调完整模型（--full_ft 时生成）
│
├── requirements.txt
├── ARCHITECTURE.md
├── USAGE_GUIDE.md
└── RESUME_GUIDE.md
```

---

## 八、关键工程决策与踩坑

| 问题                                                        | 根因                                                                                     | 解法                                                                                                          |
|-------------------------------------------------------------|------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| AFQMC test 集标签全为 -1                                    | CLUE 竞赛格式，test 标签未公开                                                           | 评估统一使用 validation 集（4,316 条）                                                                        |
| Windows OpenMP 冲突                                         | torch 与 numpy 各自链接了一份 libiomp5md.dll                                             | 脚本顶部加 `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"`                                                      |
| transformers 不接受 Windows 相对路径                        | 路径含 `..` 被当成 HuggingFace repo ID 验证                                              | 用 `Path(__file__).parent.parent` 构造绝对路径                                                                |
| TripletLoss 使用欧氏距离                                    | `F.triplet_margin_loss` 默认 L2 距离                                                     | encode() 输出已 L2 归一化，欧氏距离与余弦距离单调相关，margin 值仍有效                                        |
| BQ/LCQMC HuggingFace 仓库无 train split                     | FinanceMTEB/bq_corpus 只有 test split                                                    | 手动按 8:1:1 切分 86K 样本；LCQMC 使用 C-MTEB/LCQMC                                                           |
| DashScope BERT API 来源                                     | 老版 `shibing624/nli_zh` 仓库使用 dataset script                                         | datasets 4.0 移除了 dataset script 支持，改用 `clue/afqmc` 等 parquet 仓库                                    |
| SFT `apply_chat_template(tokenize=True)` 返回 BatchEncoding | transformers 5.x 改变返回类型                                                            | 改用 `tokenize=False` + 单独 `tokenizer.encode()`                                                             |
| SFT `from_pretrained` 的 `torch_dtype` 废弃                 | transformers 5.x 将参数改名为 `dtype`                                                    | 改用 `dtype=torch.float32`                                                                                    |
| SYSTEM_PROMPT 中的引号冲突                                  | `"相似"` 里的 ASCII 双引号与 Python 字符串分隔符冲突                                     | 改用【相似】/【不相似】（角括号格式），同步更新 LABEL_MAP 和解析逻辑                                          |
| SFT 训练后 F1(pos)=0，Accuracy≈69%                          | AFQMC 正负比 31:69，不平衡训练使模型退化为全预测负例（与 CrossEncoder 1-epoch 同一问题） | `train_sft.py` 默认正负平衡采样：各取 `num_train//2` 条                                                       |
| `pretrain_models/` 路径在 ROOT.parent 找不到                | 项目从 `pretrain_models/` 与本仓库平级移到了再上一层                                     | 全部脚本将 `BERT_PATH = ROOT.parent / "pretrain_models" / ...` 改为 `ROOT.parent.parent / ...`，影响 8 个文件 |
| SFT 多跑几个 epoch 反而 val_loss 上升                       | TARGET 仅 3~5 token，1 epoch 已收敛，更多 epoch 在标签过拟合                             | 默认 3 epoch 但保存"最优 val_loss" checkpoint（实测 epoch 1 即最优）；想节省时间可直接 `--epochs 1`           |