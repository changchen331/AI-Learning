# coding:utf8

import numpy as np
import torch
import torch.nn as nn

"""
手动实现LSTM前向计算过程，并与PyTorch结果对比
同时实现双向LSTM的手动计算

LSTM门控公式（PyTorch约定顺序：i, f, g, o）:
  i_t = sigmoid(W_ii * x_t + b_ii + W_hi * h_{t-1} + b_hi)   输入门
  f_t = sigmoid(W_if * x_t + b_if + W_hf * h_{t-1} + b_hf)   遗忘门
  g_t =    tanh(W_ig * x_t + b_ig + W_hg * h_{t-1} + b_hg)   候选记忆
  o_t = sigmoid(W_io * x_t + b_io + W_ho * h_{t-1} + b_ho)   输出门
  c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t                            细胞状态
  h_t = o_t ⊙ tanh(c_t)                                       隐藏状态
"""


# ─────────────────────────────────────────────────────────────────────────────
# 1. PyTorch 单向 LSTM
# ─────────────────────────────────────────────────────────────────────────────
class TorchLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layer = nn.LSTM(input_size, hidden_size, bias=True, batch_first=True)

    def forward(self, x):
        return self.layer(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2. 手动实现单向 LSTM
# ─────────────────────────────────────────────────────────────────────────────
class DiyLSTM:
    """
    weight_ih: [4*hidden, input]   按行分为 W_i, W_f, W_g, W_o 四块
    weight_hh: [4*hidden, hidden]  同上
    bias_ih / bias_hh: [4*hidden]  各门的偏置（PyTorch 拆成 ih 和 hh 两组）
    """

    def __init__(self, weight_ih, weight_hh, bias_ih, bias_hh, hidden_size):
        self.weight_ih = weight_ih
        self.weight_hh = weight_hh
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh
        self.hidden_size = hidden_size

    def forward(self, x):
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        output = []

        for xt in x:
            # 将四个门的线性变换一次算完，再按 hidden_size 切分
            gates = (self.weight_ih @ xt + self.bias_ih
                     + self.weight_hh @ h + self.bias_hh)

            i = sigmoid(gates[0: self.hidden_size])  # 输入门
            f = sigmoid(gates[self.hidden_size: 2 * self.hidden_size])  # 遗忘门
            g = tanh(gates[2 * self.hidden_size: 3 * self.hidden_size])  # 候选记忆
            o = sigmoid(gates[3 * self.hidden_size:])  # 输出门

            c = f * c + i * g  # 更新细胞状态
            h = o * tanh(c)  # 更新隐藏状态
            output.append(h.copy())

        return np.array(output), h, c


# ─────────────────────────────────────────────────────────────────────────────
# 3. PyTorch 双向 LSTM
# ─────────────────────────────────────────────────────────────────────────────
class TorchBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layer = nn.LSTM(input_size, hidden_size,
                             bias=True, batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.layer(x)


# ─────────────────────────────────────────────────────────────────────────────
# 4. 手动实现双向 LSTM
#    正向：从 t=0 → t=T-1
#    反向：从 t=T-1 → t=0
#    每个时间步的输出 = [正向 h_t ; 反向 h_t]（拼接）
# ─────────────────────────────────────────────────────────────────────────────
class DiyBiLSTM:
    def __init__(self, weight_ih_fwd, weight_hh_fwd, bias_ih_fwd, bias_hh_fwd,
                 weight_ih_bwd, weight_hh_bwd, bias_ih_bwd, bias_hh_bwd,
                 hidden_size):
        self.fwd = DiyLSTM(weight_ih_fwd, weight_hh_fwd,
                           bias_ih_fwd, bias_hh_fwd, hidden_size)
        self.bwd = DiyLSTM(weight_ih_bwd, weight_hh_bwd,
                           bias_ih_bwd, bias_hh_bwd, hidden_size)

    def forward(self, x):
        fwd_out, fwd_h, fwd_c = self.fwd.forward(x)  # 正向
        bwd_out, bwd_h, bwd_c = self.bwd.forward(x[::-1])  # 反向（输入逆序）
        bwd_out = bwd_out[::-1]  # 还原成正序对齐

        output = np.concatenate([fwd_out, bwd_out], axis=-1)  # 拼接两个方向
        return output, fwd_h, bwd_h, fwd_c, bwd_c


# ─────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────────────────
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


# ─────────────────────────────────────────────────────────────────────────────
# 实验入口
# ─────────────────────────────────────────────────────────────────────────────
x_np = np.array([[1, 2, 3],
                 [3, 4, 5],
                 [5, 6, 7]], dtype=np.float32)  # shape: (seq_len=3, input_size=3)

input_size = 3
hidden_size = 4

# ── 单向 LSTM ──────────────────────────────────────────────────────────────
print("=" * 60)
print("【单向 LSTM】")
print("=" * 60)

torch_lstm = TorchLSTM(input_size, hidden_size)
sd = torch_lstm.state_dict()

w_ih = sd["layer.weight_ih_l0"].detach().numpy()  # [16, 3]
w_hh = sd["layer.weight_hh_l0"].detach().numpy()  # [16, 4]
b_ih = sd["layer.bias_ih_l0"].detach().numpy()  # [16]
b_hh = sd["layer.bias_hh_l0"].detach().numpy()  # [16]

torch_x = torch.FloatTensor([x_np])  # [batch=1, seq=3, input=3]
torch_out, (torch_hn, torch_cn) = torch_lstm(torch_x)

print("PyTorch output:\n", torch_out.detach().numpy()[0])
print("PyTorch h_n:\n", torch_hn.detach().numpy()[0])
print("PyTorch c_n:\n", torch_cn.detach().numpy()[0])

diy_lstm = DiyLSTM(w_ih, w_hh, b_ih, b_hh, hidden_size)
diy_out, diy_h, diy_c = diy_lstm.forward(x_np)

print("\nDIY    output:\n", diy_out)
print("DIY    h_n:\n", diy_h)
print("DIY    c_n:\n", diy_c)

print("\n输出最大误差:", np.abs(torch_out.detach().numpy()[0] - diy_out).max())
print("h_n 最大误差:", np.abs(torch_hn.detach().numpy()[0] - diy_h).max())
print("c_n 最大误差:", np.abs(torch_cn.detach().numpy()[0] - diy_c).max())

# ── 双向 LSTM ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("【双向 LSTM】")
print("=" * 60)

torch_bilstm = TorchBiLSTM(input_size, hidden_size)
sd2 = torch_bilstm.state_dict()

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

torch_out2, (torch_hn2, torch_cn2) = torch_bilstm(torch_x)

print("PyTorch output (每步拼接正+反向):\n", torch_out2.detach().numpy()[0])
# h_n shape: [2, batch, hidden]  第0层正向, 第1层反向
print("PyTorch h_n fwd:\n", torch_hn2[0].detach().numpy())
print("PyTorch h_n bwd:\n", torch_hn2[1].detach().numpy())

diy_bilstm = DiyBiLSTM(w_ih_fwd, w_hh_fwd, b_ih_fwd, b_hh_fwd,
                       w_ih_bwd, w_hh_bwd, b_ih_bwd, b_hh_bwd,
                       hidden_size)
diy_out2, diy_hf, diy_hb, diy_cf, diy_cb = diy_bilstm.forward(x_np)

print("\nDIY    output:\n", diy_out2)
print("DIY    h_n fwd:\n", diy_hf)
print("DIY    h_n bwd:\n", diy_hb)

print("\n输出最大误差:", np.abs(torch_out2.detach().numpy()[0] - diy_out2).max())
