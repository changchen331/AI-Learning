"""
GPT 语言模型定义（Transformer Decoder-only 架构）

教学重点：
  1. 多头自注意力（Multi-Head Self-Attention）：Q/K/V 投影 + 缩放点积
  2. 因果掩码（Causal Mask）：预训练时每个位置只能看到自己及左侧 token
  3. 前馈网络（FFN）：两层线性 + GELU，参数量约占模型总量的 2/3
  4. 位置编码（Learned Positional Embedding）：可学习 vs 固定 sin/cos
  5. 语言模型头（LM Head）：最后一层映射到 vocab_size，权重与 embedding 共享

默认配置（Mini GPT，~25M 参数，适合 8G 显存）：
  vocab_size=21128, seq_len=256, d_model=384, n_heads=6, n_layers=6, d_ff=1536
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


class CausalSelfAttention(nn.Module):
    """
    因果多头自注意力

    "因果"的含义：位置 i 的输出只依赖位置 0..i 的输入，
    通过在注意力分数上加下三角掩码实现，保证自回归生成时不作弊。
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # 三个投影合并成一个大矩阵，效率更高
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, seq_len, d_model

        # 一次前向得到 Q/K/V，再 split
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=-1)  # 各 (B, T, C)

        # 拆分多头：(B, T, C) → (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # 缩放点积注意力：score = QK^T / sqrt(d_head)
        scale = math.sqrt(self.d_head)
        attn = (q @ k.transpose(-2, -1)) / scale  # (B, n_heads, T, T)

        # 因果掩码：上三角（不含对角线）置为 -inf，softmax 后趋近于 0
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(causal_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # 加权求和，合并多头
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out))


class FeedForward(nn.Module):
    """位置独立的前馈网络：Linear → GELU → Linear"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """单个 Transformer Decoder 块：Pre-LN 结构（LN 在残差支路之前）"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 残差连接：x + sublayer(LN(x))
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """
    Decoder-only GPT 语言模型

    参数规模（默认配置）：
      Token Embedding:  21128 × 384 = 8.1M
      Position Embed:     256 × 384 = 0.1M
      6 × TransformerBlock:
        Attention QKV+Out: 4 × 384² ≈ 0.6M/层 × 6 = 3.5M
        FFN (384→1536→384): 2 × 384×1536 ≈ 1.2M/层 × 6 = 7.1M
        LayerNorm × 2:      negligible
      LM Head: 共享 Token Embedding 权重，不额外计参数
    总计：~25M 参数
    """

    def __init__(
            self,
            vocab_size: int = 21128,
            seq_len: int = 256,
            d_model: int = 384,
            n_heads: int = 6,
            n_layers: int = 6,
            d_ff: int = 1536,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)
        # LM Head：将隐状态映射到词表概率，权重与 token_emb 共享（weight tying）
        # 好处：减少参数量，且经验上提升生成质量
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        # 参数初始化：小标准差有助于训练初期稳定
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (B, T)  long tensor，T <= seq_len
        返回 logits: (B, T, vocab_size)
        """
        B, T = input_ids.shape
        assert T <= self.seq_len, f"输入长度 {T} 超过最大序列长度 {self.seq_len}"

        positions = torch.arange(T, device=input_ids.device)  # (T,)
        x = self.emb_dropout(self.token_emb(input_ids) + self.pos_emb(positions))

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        return self.lm_head(x)  # (B, T, vocab_size)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(vocab_size: int = 21128, seq_len: int = 256) -> MiniGPT:
    """构建默认的 Mini GPT 模型"""
    model = MiniGPT(
        vocab_size=vocab_size,
        seq_len=seq_len,
        d_model=384,
        n_heads=6,
        n_layers=6,
        d_ff=1536,
        dropout=0.1,
    )
    return model


if __name__ == "__main__":
    model = build_model()
    n_params = model.count_parameters()
    print(f"模型参数量：{n_params / 1e6:.1f}M")

    # 测试前向传播
    dummy = torch.randint(0, 21128, (2, 256))
    logits = model(dummy)
    print(f"输入形状：{dummy.shape} → 输出 logits：{logits.shape}")
