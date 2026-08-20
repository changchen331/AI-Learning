# 实验结果总览（自动生成）

| 数据集    | 方法                | best epoch | val_acc |     val_f1 |   阈值 | 总耗时(s) | 状态 |
|-----------|---------------------|-----------:|--------:|-----------:|-------:|----------:|------|
| afqmc     | BiEncoder + Cosine  |          3 |  0.6735 | **0.6765** |   0.51 |       697 | ok   |
| afqmc     | BiEncoder + Triplet |          2 |  0.6664 | **0.6599** |   0.81 |       338 | ok   |
| afqmc     | CrossEncoder        |          3 |  0.6905 | **0.6750** | argmax |       571 | ok   |
| lcqmc     | BiEncoder + Cosine  |          3 |  0.7902 | **0.7894** |   0.72 |      2888 | ok   |
| lcqmc     | BiEncoder + Triplet |          3 |  0.8175 | **0.8173** |   0.76 |      3156 | ok   |
| lcqmc     | CrossEncoder        |          3 |  0.8563 | **0.8562** | argmax |      3600 | ok   |
| bq_corpus | BiEncoder + Cosine  |          3 |  0.8651 | **0.8649** |   0.69 |       858 | ok   |
| bq_corpus | BiEncoder + Triplet |          3 |  0.8545 | **0.8545** |   0.56 |       622 | ok   |
| bq_corpus | CrossEncoder        |          3 |  0.8848 | **0.8848** | argmax |       861 | ok   |
