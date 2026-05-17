import numpy as np
import torch
from torch import nn

LENGTH = 40

"""
手动实现 RNN 前向计算过程，并与 PyTorch 结果对比
同时实现双向 RNN 的手动计算

RNN 公式:
  h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh) 隐藏状态
"""


# Torch RNN
class TorchRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TorchRNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, bias=True, batch_first=True)

    def forward(self, x):
        return self.rnn(x)


# Mine RNN
class MineRNN:
    def __init__(self, weight_hh, weight_ih, bias_hh, bias_ih, hidden_dim):
        self.weight_hh = weight_hh
        self.weight_ih = weight_ih
        self.bias_hh = bias_hh
        self.bias_ih = bias_ih
        self.hidden_dim = hidden_dim

    def forward(self, x):
        h = np.zeros(self.hidden_dim)
        y = []

        for xt in x:
            h = np.tanh(self.weight_hh @ h + self.bias_hh +
                        self.weight_ih @ xt + self.bias_ih)
            y.append(h.copy())

        return np.array(y), h


# Torch BiRNN
class TorchBiRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TorchBiRNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, bias=True,
                          batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.rnn(x)


# Mine BiRNN
class MineBiRNN:
    def __init__(self, weight_hh_f, weight_ih_f, bias_hh_f, bias_ih_f,
                 weight_hh_b, weight_ih_b, bias_hh_b, bias_ih_b, hidden_dim):
        self.fwd = MineRNN(weight_hh_f, weight_ih_f, bias_hh_f, bias_ih_f, hidden_dim)
        self.bwd = MineRNN(weight_hh_b, weight_ih_b, bias_hh_b, bias_ih_b, hidden_dim)

    def forward(self, x):
        fwd, fh = self.fwd.forward(x)  # 正向
        bwd, bh = self.bwd.forward(x[::-1])  # 反向（逆序）
        bwd = bwd[::-1]  # 反向（正序）

        y = np.concatenate([fwd, bwd], axis=-1)
        return y, fh, bh


if __name__ == '__main__':
    size = np.random.randint(5, 10)
    input_size = np.random.randint(1, 5)
    hidden_size = np.random.randint(5, 10)

    print("=" * LENGTH, " 输入 ", "=" * LENGTH)
    X = np.random.randn(1, size, input_size)
    print("X:\n", X)

    # 单向 RNN
    print("=" * LENGTH, " 单向 RNN ", "=" * LENGTH)
    torch_rnn = TorchRNN(input_size, hidden_size)
    state_dict = torch_rnn.state_dict()
    w_hh = state_dict["rnn.weight_hh_l0"].detach().numpy()
    w_ih = state_dict["rnn.weight_ih_l0"].detach().numpy()
    b_hh = state_dict["rnn.bias_hh_l0"].detach().numpy()
    b_ih = state_dict["rnn.bias_ih_l0"].detach().numpy()

    torch_out, torch_h = torch_rnn.forward(torch.FloatTensor(X))
    torch_out = torch_out.detach().numpy()[0]
    torch_h = torch_h.detach().numpy()[0]
    print("Torch RNN output:\n", torch_out)
    print("Torch RNN hidden state:\n", torch_h)
    print("-" * LENGTH * 2)

    mine_rnn = MineRNN(w_hh, w_ih, b_hh, b_ih, hidden_size)
    mine_out, mine_h = mine_rnn.forward(X[0])
    print("Mine RNN output:\n", mine_out)
    print("Mine RNN hidden state:\n", mine_h)
    print("-" * LENGTH * 2)

    print("Output 最大误差: ", np.abs(torch_out - mine_out).max())
    print("Hidden State 最大误差: ", np.abs(torch_h - mine_h).max())
    print()

    # 双向 RNN
    print("=" * LENGTH, " 双向 RNN ", "=" * LENGTH)
    torch_bi_rnn = TorchBiRNN(input_size, hidden_size)
    state_dict = torch_bi_rnn.state_dict()
    # 正向权重
    w_hh_f = state_dict["rnn.weight_hh_l0"].detach().numpy()
    w_ih_f = state_dict["rnn.weight_ih_l0"].detach().numpy()
    b_hh_f = state_dict["rnn.bias_hh_l0"].detach().numpy()
    b_ih_f = state_dict["rnn.bias_ih_l0"].detach().numpy()
    # 反向权重
    w_hh_b = state_dict["rnn.weight_hh_l0_reverse"].detach().numpy()
    w_ih_b = state_dict["rnn.weight_ih_l0_reverse"].detach().numpy()
    b_hh_b = state_dict["rnn.bias_hh_l0_reverse"].detach().numpy()
    b_ih_b = state_dict["rnn.bias_ih_l0_reverse"].detach().numpy()

    torch_out, torch_h = torch_bi_rnn.forward(torch.FloatTensor(X))
    torch_out = torch_out.detach().numpy()[0]
    torch_h = torch_h.detach().numpy()
    torch_fh = torch_h[0]
    torch_bh = torch_h[1]
    print("Torch BiRNN output:\n", torch_out)
    print("Torch BiRNN forward hidden state:\n", torch_fh)
    print("Torch BiRNN backward hidden state:\n", torch_bh)
    print("-" * LENGTH * 2)

    mine_bi_rnn = MineBiRNN(w_hh_f, w_ih_f, b_hh_f, b_ih_f,
                            w_hh_b, w_ih_b, b_hh_b, b_ih_b,
                            hidden_size)
    mine_out, mine_fh, mine_bh = mine_bi_rnn.forward(X[0])
    print("Mine BiRNN output:\n", mine_out)
    print("Mine BiRNN forward hidden state:\n", mine_fh)
    print("Mine BiRNN backward hidden state:\n", mine_bh)
    print("-" * LENGTH * 2)

    print("Output 最大误差: ", np.abs(torch_out - mine_out).max())
    print("Forward Hidden State 最大误差:", np.abs(torch_fh - mine_fh).max())
    print("Backward Hidden State 最大误差:", np.abs(torch_bh - mine_bh).max())
