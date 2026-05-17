import numpy as np
import torch
import torch.nn as nn

# ── 可重复性 ──────────────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── 超参数 ────────────────────────────────────────────────────────────────────
LR = 1e-3
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8


# ── 简单网络（单线性层，便于手动核验） ─────────────────────────────────────────
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2, bias=False)

    def forward(self, x):
        return self.fc(x)


# ── 构造一个 batch ────────────────────────────────────────────────────────────
x = torch.randn(8, 4)  # 8 个样本，4 维输入
target = torch.randn(8, 2)  # 回归目标

# ══════════════════════════════════════════════════════════════════════════════
# Part 1 — PyTorch Adam 计算一步
# ══════════════════════════════════════════════════════════════════════════════
net_torch = SimpleNet()
W_init = net_torch.fc.weight.data.clone()  # 保存初始权重，后面手动验证用

optimizer = torch.optim.Adam(net_torch.parameters(), lr=LR,
                             betas=(BETA1, BETA2), eps=EPS)

loss_fn = nn.MSELoss()
pred = net_torch(x)
loss = loss_fn(pred, target)

optimizer.zero_grad()
loss.backward()

# 拿到梯度（反向传播之后、step 之前）
grad_torch = net_torch.fc.weight.grad.clone()

optimizer.step()
W_after_torch = net_torch.fc.weight.data.clone()

print("=" * 60)
print("【PyTorch Adam】")
print(f"  loss          : {loss.item():.6f}")
print(f"  weight (init) :\n{W_init.numpy()}")
print(f"  grad          :\n{grad_torch.numpy()}")
print(f"  weight (after):\n{W_after_torch.numpy()}")

# ══════════════════════════════════════════════════════════════════════════════
# Part 2 — 手动 Adam，逐步还原公式
# ══════════════════════════════════════════════════════════════════════════════
#
#   超参数
#   alpha = 学习率 (lr)
#   beta1 = 一阶矩衰减系数（默认 0.9）
#   beta2 = 二阶矩衰减系数（默认 0.999）
#   eps   = 防止除零的小常数
#
#   状态变量（每个参数独立维护）
#   t  = 时间步，每次 step +1
#   mt = 一阶矩（梯度的指数移动均值，估计梯度期望）
#   vt = 二阶矩（梯度平方的指数移动均值，估计梯度方差）
#
#   每步计算
#   gt   = 当前梯度
#   mt   = beta1 * mt  + (1 - beta1) * gt          ← 更新一阶矩
#   vt   = beta2 * vt  + (1 - beta2) * gt**2       ← 更新二阶矩
#   m̂t   = mt / (1 - beta1**t)                     ← 偏差修正（早期 t 小，mt 被压低）
#   v̂t   = vt / (1 - beta2**t)                     ← 偏差修正
#   w    = w - alpha * m̂t / (sqrt(v̂t) + eps)      ← 参数更新
#

alpha = LR
beta1 = BETA1
beta2 = BETA2
eps = EPS

# 从相同初始权重出发，使用相同梯度
W_manual = W_init.numpy().copy()
grad = grad_torch.numpy().copy()

# 初始化状态（t=0 时刻，矩为 0）
t = 0
mt = np.zeros_like(W_manual)
vt = np.zeros_like(W_manual)

# ── 执行一步 ──────────────────────────────────────────────────────────────────
t = t + 1  # step 1
gt = grad

mt = beta1 * mt + (1 - beta1) * gt  # 一阶矩更新
vt = beta2 * vt + (1 - beta2) * gt ** 2  # 二阶矩更新

mth = mt / (1 - beta1 ** t)  # 偏差修正
vth = vt / (1 - beta2 ** t)  # 偏差修正

W_manual = W_manual - alpha * mth / (np.sqrt(vth) + eps)  # 权重更新

print()
print("=" * 60)
print("【手动 Adam】")
print(f"  t             : {t}")
print(f"  mt (一阶矩)   :\n{mt}")
print(f"  vt (二阶矩)   :\n{vt}")
print(f"  m̂t (修正后)   :\n{mth}")
print(f"  v̂t (修正后)   :\n{vth}")
print(f"  weight (after):\n{W_manual}")

# ══════════════════════════════════════════════════════════════════════════════
# Part 3 — 对比
# ══════════════════════════════════════════════════════════════════════════════
diff = np.abs(W_after_torch.numpy() - W_manual)
print()
print("=" * 60)
print("【对比】权重差异（手动 vs PyTorch）")
print(f"  最大误差: {diff.max():.2e}")
print(f"  平均误差: {diff.mean():.2e}")
print("  结论:", "✓ 完全一致（误差在浮点精度范围内）"
if diff.max() < 1e-6 else "✗ 存在较大差异，请检查")
