import numpy as np
import torch

x = np.random.random((4, 5))

bn = torch.nn.BatchNorm1d(5)

# y = bn(torch.from_numpy(x).float())
y = bn(torch.FloatTensor(x))

print(x)
print("torch y:", y)
print(bn.state_dict())

# numpy实现

gamma = bn.state_dict()["weight"].numpy()
beta = bn.state_dict()["bias"].numpy()

num_features = 5
eps = 1e-05
momentum = 0.1

# initialize the running mean and variance to zero
running_mean = np.zeros(num_features)
running_var = np.zeros(num_features)

mean = np.mean(x, axis=0)
var = np.var(x, axis=0)

# normalize the input with the mean and variance
x_norm = (x - mean) / np.sqrt(var + eps)

# scale and shift the normalized input with gamma and beta
y = gamma * x_norm + beta
print("ours y:", y)
