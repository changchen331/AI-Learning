# 文本匹配 — LCQMC / BQ Corpus 复现实验总结

> 在 AFQMC 已有实验基础上，把 BERT 三方法（BiEncoder + Cosine、BiEncoder + Triplet、CrossEncoder）原样复用到 LCQMC 与 BQ Corpus 上，得到 **3 个数据集 × 3 种方法 = 9 组结果**，并对跨数据集规律做横向解读。
>
> 实验代码全部复用 `src/`（一行未改），LCQMC / BQ 的入口与产物完全独立隔离。

---

## 一、实验设置（三套实验完全对齐，确保可比）

| 项 | 值 |
|----|----|
| 预训练模型 | bert-base-chinese |
| BERT 层数 | 4（限层加速） |
| epoch | 3 |
| batch_size | 32 |
| max_length（biencoder / cross）| 64 / 128 |
| learning rate（BERT 主干 / head）| 2e-5 / 1e-4 |
| margin（cosine / triplet 同名参数）| 0.3 |
| 池化 | mean |
| 评估指标 | Accuracy / weighted F1（biencoder 在 val 上做 101 阈值搜索）|
| 评估 split | validation |
| 硬件 | RTX 4060 Laptop GPU |

> **只换数据集，不换配方**，所有数字直接横向可比。

---

## 二、数据集对比

| 数据集 | train | val | test | 领域 | 来源 |
|------|------:|----:|----:|------|------|
| AFQMC | 34,334 | 4,316 | 3,861（无标签）| 蚂蚁金融问句 | CLUE / 蚂蚁集团 |
| LCQMC | 238,766 | 8,802 | 12,500 | 开放域问句 | 哈工大 |
| BQ Corpus | 68,960 | 8,620 | 8,620 | 微众银行问句 | 哈工大 + 微众银行 |

> 数据量比例 **LCQMC : BQ : AFQMC ≈ 7 : 2 : 1**。AFQMC 是三者中最小且标注噪声最高的。

---

## 三、量化结果（自动汇总）

完整自动汇总表见 [SUMMARY_TABLE.md](./SUMMARY_TABLE.md)，由 `aggregate.py` 读取三个 `outputs/logs/` 目录生成。

### 3.1 最终对比（按数据集排）

| 数据集 | 训练量 | BiEncoder Cosine | BiEncoder Triplet | CrossEncoder | 该数据集冠军 |
|------|------:|--------------:|---------------:|-----------:|---------|
| AFQMC | 34K | **0.6765** | 0.6599 | 0.6750 | **Cosine** |
| LCQMC | 238K | 0.7894 | 0.8173 | **0.8562** | **CrossEncoder** |
| BQ Corpus | 69K | 0.8649 | 0.8545 | **0.8848** | **CrossEncoder** |

> 表中数字均为 weighted F1（best epoch）。

### 3.2 训练耗时（秒）

| 数据集 | Cosine | Triplet | Cross | 总耗时 |
|------|------:|------:|------:|------:|
| AFQMC | 697 | 338 | 571 | **~26 min** |
| BQ Corpus | 858 | 622 | 861 | **~39 min** |
| LCQMC | 2,888 | 3,156 | 3,600 | **~161 min** |

> Triplet 因为只用正例对构造三元组（约等于正例数），**训练样本量是 Cosine/Cross 的一半**——所以 Triplet 在 AFQMC、BQ 上比 Cosine 显著快，而在 LCQMC 上反而更慢（LCQMC 正例多，三元组也多，配上每对要过 3 次 BERT，整体翻倍）。

### 3.3 各 epoch 验证集 F1 趋势

| 数据集 | 方法 | epoch1 | epoch2 | epoch3 |
|------|------|------:|------:|------:|
| AFQMC | Cosine | (n/a) | (n/a) | 0.6765 |
|  | Triplet |  | **0.6599** ↘ |  |
|  | Cross | (低) |  | 0.6750 |
| LCQMC | Cosine | 0.7589 | 0.7832 | **0.7894** |
|  | Triplet | 0.7953 | 0.8134 | **0.8173** |
|  | Cross | 0.8227 | 0.8472 | **0.8562** |
| BQ Corpus | Cosine | 0.8230 | 0.8580 | **0.8649** |
|  | Triplet | 0.8276 | 0.8462 | **0.8545** |
|  | Cross | 0.8298 | 0.8714 | **0.8848** |

> AFQMC 的 epoch 趋势详见原 `outputs/logs/`，简化为最终值。LCQMC、BQ 的所有方法 **3 个 epoch 都在单调上升**——和 AFQMC 上 "epoch 3 已轻微震荡" 不同，更大的数据集需要的训练充分性更高。

### 3.4 阈值（BiEncoder 在 val 上的最优分类阈值）

| 数据集 | Cosine 阈值 | Triplet 阈值 |
|------|----------:|------------:|
| AFQMC | 0.51 | 0.81 |
| LCQMC | 0.72 | 0.76 |
| BQ Corpus | 0.69 | 0.56 |

---

## 四、跨数据集解读

### 4.1 验证了 AFQMC 实验中的核心预测：数据量足够时 Triplet 反超 Cosine

| 数据集 | 训练量 | Cosine | Triplet | Δ(Triplet − Cosine) | 趋势 |
|------|------:|------:|------:|------:|------|
| AFQMC | 34K | 0.6765 | 0.6599 | **−0.0166** | Cosine 领先 |
| BQ Corpus | 69K | 0.8649 | 0.8545 | **−0.0104** | Cosine 仍领先，但差距缩半 |
| LCQMC | 238K | 0.7894 | 0.8173 | **+0.0279** | **Triplet 反超** |

这正是 ARCHITECTURE.md 4.2 节给出的预言："AFQMC 正样本仅 10K 条，TripletDataset 构造的三元组同样只有 10K，训练信号偏少；**数据量更大的 LCQMC（238K 对）上 Triplet 的优势会更明显**。"

LCQMC 上 Triplet 已经显著优于 Cosine，而且 Triplet 仅训练 1 epoch 就达到 0.7953，**已超过 Cosine 跑满 3 epoch 的 0.7894** ——三元组训练在大数据下 sample efficiency 更高。

> 数据量临界点位于 **70K 到 240K 之间**：BQ 69K Triplet 还未反超，LCQMC 238K 已经反超 +0.028。直观上正例对数量级达到 10⁵ 时 Triplet 才开始显示优势。

### 4.2 CrossEncoder 在所有"数据量足够"的数据集上稳居第一

- AFQMC（34K）：Cross F1 = 0.6750，**比 Cosine 0.6765 略低**——但 Acc=0.6905 反而最高，这是 ARCHITECTURE.md 已分析的"类别不均衡 + 训练不足时 Cross 倾向多数类"。
- BQ Corpus（69K）：Cross F1 = 0.8848，比 Cosine **+0.020**。
- LCQMC（238K）：Cross F1 = 0.8562，比 Triplet（次优）**+0.039**。

**结论**：当数据量超过 ~70K 时，CrossEncoder 全层交互的表达优势就稳定显现；AFQMC 数据量不足是 Cross 没拉开差距的主因，不是架构问题。

### 4.3 数据集"难度"与正负比的影响

| 数据集 | 各方法最高 F1 | 推测难度因素 |
|------|------------:|------------|
| **BQ Corpus** | 0.8848 | 域窄（银行业务）+ 标注一致 → 最易 |
| **LCQMC** | 0.8562 | 开放域 + 数据量大 → 数据量补偿了域宽度 |
| **AFQMC** | 0.6765 | 域窄但 **数据少 + 标注噪声**（见 ARCHITECTURE 5.4 节）→ 最难 |

> 反直觉点：开放域 LCQMC 上限（0.8562）比单一金融域 AFQMC（0.6765）高 19 个点。**数据量 + 标注质量 > 领域宽度** 在这组实验里完全成立。

### 4.4 阈值规律：训练分布会动 Cosine 与 Triplet 对应阈值不同

- **Cosine 阈值**随数据规模/复杂度变化（AFQMC 0.51 → BQ 0.69 → LCQMC 0.72），数据集复杂度越高，正负分布越往中间堆叠，最优切点越靠右；
- **Triplet 阈值**没有同样的单调规律（0.81 / 0.56 / 0.76）——TripletLoss 只约束相对距离，绝对相似度量级会因数据分布漂移；BQ 上 Triplet 阈值反而最低（0.56），这是 BQ 正负样本被 Triplet "拉到不同极端" 的结果，与 LCQMC、AFQMC 拉开的方式不同。
- **教学要点**：**Triplet 训练完得在自己数据集上重新搜阈值，不能套 Cosine 的经验值**。

### 4.5 收敛性

LCQMC 和 BQ 的所有方法 3 个 epoch 都还在上升，AFQMC 的 Triplet 在 epoch 2 已见顶。这说明：

| 现象 | 解释 |
|------|------|
| AFQMC Triplet 早收敛 | 三元组只有 ~10K，1 epoch 就把信号学完，2 epoch 后 train_loss 几乎为 0 |
| LCQMC / BQ 仍在上升 | 数据量大，3 epoch 不算训练充分 → **建议 5 epoch 才能榨干** |

---

## 五、结论与教学要点

1. **数据量决定 Loss 选型**：
   - <50K 正样本对：用 **CosineEmbeddingLoss**（Triplet 三元组不够）
   - >100K 正样本对：用 **TripletLoss**（更高 sample efficiency）
   - 不知道选哪个：**CrossEncoder** 几乎永远不输（除非小到 AFQMC 的 34K）

2. **数据量大于一切**（在三个对比里）：
   - LCQMC 用最复杂的开放域数据 + 7× AFQMC 训练量，所有方法都比 AFQMC 高 10+ 个点；
   - 同领域的 AFQMC（34K）和 BQ（69K），仅 2× 数据差，最高 F1 差了 21 个点（0.6765 vs 0.8848）——AFQMC 的数据噪声把上限锁住了。

3. **AFQMC 上的"反差教学点"在大数据集上消失**：
   - "Cross 1 epoch Acc 虚高 / F1 极低" 在 LCQMC、BQ 上不复存在（epoch 1 F1 已经合理），原因是大数据下少数类信号也足够，Cross 不退化；
   - "Triplet 不如 Cosine" 在 LCQMC 上完全反转。
   - **教学价值**：把这两个反差作为"小数据 vs 大数据" 的对比例子讲，比单数据集结论更有说服力。

4. **阈值规律是 BiEncoder 部署时必须重做的步骤**：
   - 不要套用其他数据集的阈值，特别是 Triplet 训练时；
   - 阈值搜索代码（`evaluate.py:_find_best_threshold`）每次评估都跑一次成本极低，建议作为 BiEncoder 评估的标准步骤，已在本实验中默认做。

5. **复用代码时的工程经验**：
   - 通过 `sys.path` + 模块级路径变量赋值实现"零修改原代码"的多数据集复用，比拷贝 src/ 副本更易维护；
   - 三个数据集独立 outputs 目录，权重文件 / 日志 / 图互不污染；
   - `aggregate.py` 仅读 logs JSON，无需重新加载模型即可生成对比表。

---

## 六、可继续的实验

- **更长训练**：LCQMC、BQ 上跑 5 epoch，看是否突破 0.86 / 0.89；
- **难负样本挖掘**：用本次训出的 BiEncoder 在 LCQMC 上找 top-K 相似但 label=0 的对，看 Triplet 是否还能进一步提升；
- **跨数据集迁移**：用 LCQMC 预训练 → AFQMC 微调，看是否突破 AFQMC 0.68 的天花板；
- **LLM SFT**：当前未做（按"BERT 三方法"范围），若做应在 LCQMC / BQ 上验证 5K 平衡采样的"小数据 LoRA 假设"在大数据集上是否仍成立。

---

## 附录：复现命令

```bash
# 切到项目根
cd "D:/badou/八斗课程/week8 文本匹配问题/文本匹配项目"

# BQ Corpus（约 39 min）
cd experiments_bq && python run_all.py --epochs 3

# LCQMC（约 161 min）
cd experiments_lcqmc && python run_all.py --epochs 3

# 汇总（秒级，读 logs JSON）
cd experiments_summary && python aggregate.py
```

`run_all.py` 也支持 `--epochs N --batch_size B --num_hidden_layers L --skip cosine|triplet|cross` 参数。详见各 `experiments_*/README.md`。
