import numpy as np
import torch
from torch import nn

"""
手动实现LSTM前向计算过程，并与PyTorch结果对比
同时实现双向LSTM的手动计算

LSTM门控公式（PyTorch约定顺序：i, f, g, o）:
  i_t = sigmoid(W_ii * x_t + b_ii + W_hi * h_{t-1} + b_hi)   输入门
  f_t = sigmoid(W_if * x_t + b_if + W_hf * h_{t-1} + b_hf)   遗忘门
  o_t = sigmoid(W_io * x_t + b_io + W_ho * h_{t-1} + b_ho)   输出门
  g_t =    tanh(W_ig * x_t + b_ig + W_hg * h_{t-1} + b_hg)   候选记忆
  c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t                            细胞状态
  h_t = o_t ⊙ tanh(c_t)                                      隐藏状态
"""

LENGTH = 40


# Torch LSTM
class TorchLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TorchLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, bias=True, batch_first=True)

    def forward(self, x):
        return self.lstm(x)


# Mine LSTM
class MineLSTM:
    def __init__(self, weight_hh, weight_ih, bias_hh, bias_ih, hidden_dim):
        self.weight_hh = weight_hh
        self.weight_ih = weight_ih
        self.bias_hh = bias_hh
        self.bias_ih = bias_ih
        self.hidden_dim = hidden_dim

    def forward(self, x):
        ct = np.zeros(self.hidden_dim)
        ht = np.zeros(self.hidden_dim)
        output = []

        for xt in x:
            gates = (self.weight_hh @ ht + self.bias_hh +
                     self.weight_ih @ xt + self.bias_ih)

            input_gate = sigmoid(gates[0: self.hidden_dim])  # 输入门
            forget_gate = sigmoid(gates[self.hidden_dim: 2 * self.hidden_dim])  # 遗忘门
            gt = tanh(gates[2 * self.hidden_dim: 3 * self.hidden_dim])  # 候选记忆
            output_gate = sigmoid(gates[3 * self.hidden_dim:4 * self.hidden_dim])  # 输出门

            ct = forget_gate * ct + input_gate * gt  # 更新细胞状态
            ht = output_gate * tanh(ct)  # 更新隐藏状态
            output.append(ht.copy())

        return np.array(output), ht, ct


# Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Tanh
def tanh(x):
    return np.tanh(x)


# Torch BiLSTM
class TorchBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TorchBiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, bias=True, batch_first=True,
                            bidirectional=True)

    def forward(self, x):
        return self.lstm(x)


# Mine BiLSTM
class MineBiLSTM:
    def __init__(self, weight_hh_f, weight_ih_f, bias_hh_f, bias_ih_f,
                 weight_hh_b, weight_ih_b, bias_hh_b, bias_ih_b, hidden_dim):
        self.fwd = MineLSTM(weight_hh_f, weight_ih_f, bias_hh_f, bias_ih_f, hidden_dim)
        self.bwd = MineLSTM(weight_hh_b, weight_ih_b, bias_hh_b, bias_ih_b, hidden_dim)

    def forward(self, x):
        fwd, fh, fc = self.fwd.forward(x)  # 正向
        bwd, bh, bc = self.bwd.forward(x[::-1])  # 反向（逆序）
        bwd = bwd[::-1]  # 反向（正序）

        output = np.concatenate([fwd, bwd], axis=-1)  # 根据最后一个维度拼接
        return output, fh, bh, fc, bc


if __name__ == '__main__':
    size = np.random.randint(5, 10)
    input_size = np.random.randint(1, 5)
    hidden_size = np.random.randint(5, 10)

    print("=" * LENGTH, " 输入 ", "=" * LENGTH)
    X = np.random.randn(1, size, input_size)
    print("X:\n", X)

    # 单向 LSTM
    print("=" * LENGTH, " 单向 LSTM", "=" * LENGTH)
    torch_lstm = TorchLSTM(input_size, hidden_size)
    state_dict = torch_lstm.state_dict()

    w_hh = state_dict["lstm.weight_hh_l0"].detach().numpy()
    w_ih = state_dict["lstm.weight_ih_l0"].detach().numpy()
    b_hh = state_dict["lstm.bias_hh_l0"].detach().numpy()
    b_ih = state_dict["lstm.bias_ih_l0"].detach().numpy()

    torch_out, (torch_ht, torch_ct) = torch_lstm.forward(torch.FloatTensor(X))
    torch_out = torch_out.detach().numpy()[0]
    torch_ht = torch_ht.detach().numpy()[0]
    torch_ct = torch_ct.detach().numpy()[0]
    print("torch_out:\n", torch_out)
    print("torch_ht:\n", torch_ht)
    print("torch_ct:\n", torch_ct)
    print('-' * LENGTH * 2)

    mine_lstm = MineLSTM(w_hh, w_ih, b_hh, b_ih, hidden_size)
    mine_out, mine_ht, mine_ct = mine_lstm.forward(X[0])
    print("mine_out:\n", mine_out)
    print("mine_ht:\n", mine_ht)
    print("mine_ct:\n", mine_ct)
    print("-" * LENGTH * 2)

    print("Output 最大误差: ", np.abs(torch_out - mine_out).max())
    print("Ht 最大误差: ", np.abs(torch_ht - mine_ht).max())
    print("Ct 最大误差: ", np.abs(torch_ct - mine_ct).max())

    # 双向 LSTM
    print("=" * LENGTH, " 双向 LSTM", "=" * LENGTH)
    torch_bi_lstm = TorchBiLSTM(input_size, hidden_size)
    state_dict = torch_bi_lstm.state_dict()
    print(state_dict)

    # 正向权重
    w_hh_f = state_dict["lstm.weight_hh_l0"].detach().numpy()
    w_ih_f = state_dict["lstm.weight_ih_l0"].detach().numpy()
    b_hh_f = state_dict["lstm.bias_hh_l0"].detach().numpy()
    b_ih_f = state_dict["lstm.bias_ih_l0"].detach().numpy()

    # 反向权重
    w_hh_b = state_dict["lstm.weight_hh_l0_reverse"].detach().numpy()
    w_ih_b = state_dict["lstm.weight_ih_l0_reverse"].detach().numpy()
    b_hh_b = state_dict["lstm.bias_hh_l0_reverse"].detach().numpy()
    b_ih_b = state_dict["lstm.bias_ih_l0_reverse"].detach().numpy()

    torch_out, (torch_ht, torch_ct) = torch_bi_lstm.forward(torch.FloatTensor(X))
    torch_out = torch_out.detach().numpy()[0]
    torch_ht = torch_ht.detach().numpy()
    torch_fht = torch_ht[0]
    torch_bht = torch_ht[1]
    torch_ct = torch_ct.detach().numpy()
    torch_fct = torch_ct[0]
    torch_bct = torch_ct[1]
    print("torch_out:\n", torch_out)
    print("torch_fht:\n", torch_fht)
    print("torch_bht:\n", torch_bht)
    print("torch_fct:\n", torch_fct)
    print("torch_bct:\n", torch_bct)
    print("-" * LENGTH * 2)

    mine_bi_lstm = MineBiLSTM(w_hh_f, w_ih_f, b_hh_f, b_ih_f,
                              w_hh_b, w_ih_b, b_hh_b, b_ih_b,
                              hidden_size)
    mine_out, mine_fht, mine_bht, mine_fct, mine_bct = mine_bi_lstm.forward(X[0])
    print("mine_out:\n", mine_out)
    print("mine_fht:\n", mine_fht)
    print("mine_bht:\n", mine_bht)
    print("mine_fct:\n", mine_fct)
    print("mine_bct:\n", mine_bct)
    print('-' * LENGTH * 2)

    print("Output 最大误差: ", np.abs(torch_out - mine_out).max())
    print("Forward Hidden State 最大误差:", np.abs(torch_fht - mine_fht).max())
    print("Backward Hidde State 最大误差:", np.abs(torch_bht - mine_bht).max())
    print("Forward Cell State 最大误差:", np.abs(torch_fct - mine_fct).max())
    print("Backward Cell State 最大误差:", np.abs(torch_bct - mine_bct).max())
