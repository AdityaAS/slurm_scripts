import torch
import torchvision
import socket

print(f"Hostname: {socket.gethostname()}")
print(f"Torch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"Torch file: {torch.__file__}")

print(f"Torch device count: {torch.cuda.device_count()}")
print(f"Torch current device: {torch.cuda.current_device()}")
print(f"Torch current device name: {torch.cuda.get_device_name(0)}")
print(f"Torch current device name: {torch.cuda.get_device_name(1)}")

print(f"CUDA status: {torch.cuda.is_available()}")

result = 2 ** 2
print("Result of 2 ^ 2: {}".format(result))
