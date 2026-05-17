# coding:utf8
import torch
import torch.nn as nn

'''
embedding层的处理
'''

# 构造字符表
vocab = {
    "[pad]": 0,
    "你": 1,
    "你好": 2,
    "中国": 3,
    "好": 4,
    "[cls]": 5,
    "[sep]": 6,
    "[unk]": 7
}
vocab_size = len(vocab)
embedding_dim = 768
token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
segment_embedding = nn.Embedding(2, embedding_dim)
position_embedding = nn.Embedding(512, embedding_dim)

# 构造输入
#       [cls] 你 您好 中国 [sep] 中国 好 [sep]
token = [5, 1, 2, 3, 6, 3, 4, 6]
seg = [0, 0, 0, 0, 0, 1, 1, 1]
pos = [0, 1, 2, 3, 4, 5, 6, 7]

tensor_token = torch.LongTensor(token)
tensor_seg = torch.LongTensor(seg)
tensor_pos = torch.LongTensor(pos)

# 计算embedding
token_emb = token_embedding(tensor_token)
seg_emb = segment_embedding(tensor_seg)
pos_emb = position_embedding(tensor_pos)

# 加和输出
output = token_emb + seg_emb + pos_emb
print(output)
print(output.size())
