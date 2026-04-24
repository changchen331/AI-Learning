# coding:utf8

import numpy as np
import torch
import torch.nn as nn

"""
手动实现RNN前向计算过程，并与PyTorch结果对比
同时实现双向RNN的手动计算

RNN 公式:
  h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)   隐藏状态
"""


# ─────────────────────────────────────────────────────────────────────────────
# 1. PyTorch 单向 RNN
# ─────────────────────────────────────────────────────────────────────────────
class TorchRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layer = nn.RNN(input_size, hidden_size, bias=True, batch_first=True)

    def forward(self, x):
        return self.layer(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2. 手动实现单向 RNN
# ─────────────────────────────────────────────────────────────────────────────
class DiyRNN:
    """
    weight_ih: [hidden, input]   输入到隐藏层的权重
    weight_hh: [hidden, hidden]  隐藏层到隐藏层的权重
    bias_ih / bias_hh: [hidden]  偏置（PyTorch 拆成 ih 和 hh 两组）
    """

    def __init__(self, weight_ih, weight_hh, bias_ih, bias_hh, hidden_size):
        self.weight_ih = weight_ih
        self.weight_hh = weight_hh
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh
        self.hidden_size = hidden_size

    def forward(self, x):
        h = np.zeros(self.hidden_size)
        output = []

        for xt in x:
            h = np.tanh(self.weight_ih @ xt + self.bias_ih
                        + self.weight_hh @ h + self.bias_hh)
            output.append(h.copy())

        return np.array(output), h


# ─────────────────────────────────────────────────────────────────────────────
# 3. PyTorch 双向 RNN
# ─────────────────────────────────────────────────────────────────────────────
class TorchBiRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layer = nn.RNN(input_size, hidden_size,
                            bias=True, batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.layer(x)


# ─────────────────────────────────────────────────────────────────────────────
# 4. 手动实现双向 RNN
#    正向：从 t=0 → t=T-1
#    反向：从 t=T-1 → t=0
#    每个时间步的输出 = [正向 h_t ; 反向 h_t]（拼接）
# ─────────────────────────────────────────────────────────────────────────────
class DiyBiRNN:
    def __init__(self, weight_ih_fwd, weight_hh_fwd, bias_ih_fwd, bias_hh_fwd,
                 weight_ih_bwd, weight_hh_bwd, bias_ih_bwd, bias_hh_bwd,
                 hidden_size):
        self.fwd = DiyRNN(weight_ih_fwd, weight_hh_fwd,
                          bias_ih_fwd, bias_hh_fwd, hidden_size)
        self.bwd = DiyRNN(weight_ih_bwd, weight_hh_bwd,
                          bias_ih_bwd, bias_hh_bwd, hidden_size)

    def forward(self, x):
        fwd_out, fwd_h = self.fwd.forward(x)  # 正向
        bwd_out, bwd_h = self.bwd.forward(x[::-1])  # 反向（输入逆序）
        bwd_out = bwd_out[::-1]  # 还原成正序对齐

        output = np.concatenate([fwd_out, bwd_out], axis=-1)  # 拼接两个方向
        return output, fwd_h, bwd_h


# ─────────────────────────────────────────────────────────────────────────────
# 实验入口
# ─────────────────────────────────────────────────────────────────────────────
x_np = np.array([[1, 2, 3],
                 [3, 4, 5],
                 [5, 6, 7]], dtype=np.float32)  # shape: (seq_len=3, input_size=3)

input_size = 3
hidden_size = 4

# ── 单向 RNN ───────────────────────────────────────────────────────────────
print("=" * 60)
print("【单向 RNN】")
print("=" * 60)

torch_rnn = TorchRNN(input_size, hidden_size)
sd = torch_rnn.state_dict()

w_ih = sd["layer.weight_ih_l0"].detach().numpy()  # [4, 3]
w_hh = sd["layer.weight_hh_l0"].detach().numpy()  # [4, 4]
b_ih = sd["layer.bias_ih_l0"].detach().numpy()  # [4]
b_hh = sd["layer.bias_hh_l0"].detach().numpy()  # [4]

torch_x = torch.FloatTensor([x_np])  # [batch=1, seq=3, input=3]
torch_out, torch_hn = torch_rnn(torch_x)

print("PyTorch output:\n", torch_out.detach().numpy()[0])
print("PyTorch h_n:\n", torch_hn.detach().numpy()[0])

diy_rnn = DiyRNN(w_ih, w_hh, b_ih, b_hh, hidden_size)
diy_out, diy_h = diy_rnn.forward(x_np)

print("\nDIY    output:\n", diy_out)
print("DIY    h_n:\n", diy_h)

print("\n输出最大误差:", np.abs(torch_out.detach().numpy()[0] - diy_out).max())
print("h_n 最大误差:", np.abs(torch_hn.detach().numpy()[0] - diy_h).max())

# ── 双向 RNN ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("【双向 RNN】")
print("=" * 60)

torch_birnn = TorchBiRNN(input_size, hidden_size)
sd2 = torch_birnn.state_dict()

# 正向权重
w_ih_fwd = sd2["layer.weight_ih_l0"].detach().numpy()
w_hh_fwd = sd2["layer.weight_hh_l0"].detach().numpy()
b_ih_fwd = sd2["layer.bias_ih_l0"].detach().numpy()
b_hh_fwd = sd2["layer.bias_hh_l0"].detach().numpy()

# 反向权重（PyTorch 命名后缀 _reverse）
w_ih_bwd = sd2["layer.weight_ih_l0_reverse"].detach().numpy()
w_hh_bwd = sd2["layer.weight_hh_l0_reverse"].detach().numpy()
b_ih_bwd = sd2["layer.bias_ih_l0_reverse"].detach().numpy()
b_hh_bwd = sd2["layer.bias_hh_l0_reverse"].detach().numpy()

torch_out2, torch_hn2 = torch_birnn(torch_x)

print("PyTorch output (每步拼接正+反向):\n", torch_out2.detach().numpy()[0])
# h_n shape: [2, batch, hidden]  第0层正向, 第1层反向
print("PyTorch h_n fwd:\n", torch_hn2[0].detach().numpy())
print("PyTorch h_n bwd:\n", torch_hn2[1].detach().numpy())

diy_birnn = DiyBiRNN(w_ih_fwd, w_hh_fwd, b_ih_fwd, b_hh_fwd,
                     w_ih_bwd, w_hh_bwd, b_ih_bwd, b_hh_bwd,
                     hidden_size)
diy_out2, diy_hf, diy_hb = diy_birnn.forward(x_np)

print("\nDIY    output:\n", diy_out2)
print("DIY    h_n fwd:\n", diy_hf)
print("DIY    h_n bwd:\n", diy_hb)

print("\n输出最大误差:", np.abs(torch_out2.detach().numpy()[0] - diy_out2).max())
