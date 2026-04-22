import random

import numpy as np
import torch

'''
手动实现 softmax
'''


def softmax(nums):
    response = []
    for num in nums:
        response.append(np.exp(num))
    summation = sum(response)
    response = [(resp / summation) for resp in response]
    return response


if __name__ == '__main__':
    xs = []
    for i in range(10):
        xs.append(random.uniform(-3, 3))

    print(f"xs: {xs}")

    torch_softmax = torch.softmax(torch.tensor(xs).float(), 0)
    print(f"Torch: {torch_softmax}")

    mine_softmax = softmax(xs)
    mine_softmax_print = [f"{y:.4f}" for y in mine_softmax]
    print(f"Mine_1: {mine_softmax_print}")
    print(f"Mine_2: {mine_softmax}")
    print(f"Sum = {sum(mine_softmax)}")
