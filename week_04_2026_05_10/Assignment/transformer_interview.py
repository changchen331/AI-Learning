"""
面试版 Transformer Encoder
核心：Multi-Head Self-Attention / FFN / 残差 + LN / 堆叠
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden, n_head):
        super().__init__()
        assert hidden % n_head == 0
        self.n_head = n_head
        self.d_k = hidden // n_head
        self.qkv = nn.Linear(hidden, hidden * 3)  # 一次性算 Q K V
        self.out = nn.Linear(hidden, hidden)

    def forward(self, x, mask=None):
        B, T, H = x.shape
        # [B, T, 3H] -> 3 个 [B, n_head, T, d_k]
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)

        # scaled dot-product
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)

        out = attn @ v  # [B, n_head, T, d_k]
        out = out.transpose(1, 2).contiguous().view(B, T, H)
        return self.out(out)


class EncoderLayer(nn.Module):
    def __init__(self, hidden, n_head, ff):
        super().__init__()
        self.attn = MultiHeadAttention(hidden, n_head)
        self.ln1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, ff),
            nn.GELU(),
            nn.Linear(ff, hidden),
        )
        self.ln2 = nn.LayerNorm(hidden)

    def forward(self, x, mask=None):
        x = self.ln1(x + self.attn(x, mask))  # 残差 + LN
        x = self.ln2(x + self.ffn(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, hidden=768, n_layer=12, n_head=12, ff=3072):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(hidden, n_head, ff) for _ in range(n_layer)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


if __name__ == "__main__":
    model = TransformerEncoder(hidden=512, n_layer=6, n_head=8, ff=1024)
    x = torch.randn(2, 16, 512)  # [B, T, H]
    print(model(x).shape)  # [2, 16, 512]
