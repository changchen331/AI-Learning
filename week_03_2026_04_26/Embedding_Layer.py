import numpy as np
import torch
from torch import nn

length = 30

vocabulary = {"[pad]": 0, "哈": 1, "基": 2, "米": 3,
              "南": 4, "北": 5, "绿": 6, "豆": 7, "[unk]": 8}


# 编码
def encode(words, voca, max_len):
    ids = [voca.get(ch, voca["[unk]"]) for ch in words][:max_len]
    ids += [voca["[pad]"]] * (max_len - len(ids))
    return ids


if __name__ == '__main__':
    # 构建 Embedding 表
    voca_size = len(vocabulary)
    embed_size = np.random.randint(3, 7)
    Embedding = nn.Embedding(voca_size, embed_size)  # 初始化 Embedding 矩阵
    print("=" * length,
          f" Embedding 权重矩阵 (shape={voca_size}×{embed_size}) ",
          "=" * length)
    print(f"Embedding matrix:\n{Embedding.weight.detach().numpy()}\n")

    for word, i in vocabulary.items():
        print(f"\"{word}\" 的向量: {Embedding.weight[i].detach().numpy().tolist()}")

    # Padding 的作用（长度不够会用 [pad] 进行填充）
    MAX_LEN = 5
    sentences = ["哈基米南北绿豆", "南北绿豆", "阿希噶阿西"]
    token_ids = [encode(sentence, vocabulary, MAX_LEN) for sentence in sentences]
    print("=" * length, " Padding 后的 Token ID ", "=" * length)
    for sen, IDS in zip(sentences, token_ids):
        print(f"{sen}: {IDS}")

    # padding_idx 参数
    Embedding_with_pad = nn.Embedding(voca_size, embed_size, padding_idx=0)
    print("=" * length, " 设置 padding_idx=0 后的权重矩阵 ", "=" * length)
    print(f"Embedding matrix:\n{Embedding_with_pad.weight.detach().numpy()}\n")

    # 完整 Batch 送入 Embedding
    xs = torch.LongTensor(token_ids)
    ys = Embedding_with_pad(xs)
    print("=" * length,
          f" Embedding 输出 (token={len(sentences)}, seq_len={MAX_LEN}, embed_dim={embed_size}) ",
          "=" * length)
    print(f"Embedding matrix:\n{ys.detach().numpy()}\n")
