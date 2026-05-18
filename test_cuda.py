import torch

# 查看torch版本.
print(torch.__version__)

# 检查使用设备.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# 查看cuda是否可用.
print(torch.cuda.is_available())

# 查看cuda版本.
print(torch.version.cuda)

# 查看GPU数量
print(torch.cuda.device_count())

# 查看设备名称.
print(torch.cuda.get_device_name(0))