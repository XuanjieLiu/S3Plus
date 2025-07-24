import torch

HAS_CUDA = torch.cuda.is_available()
CUDA = torch.device("cuda:0")
CPU  = torch.device("cpu")
if HAS_CUDA:
    DEVICE = CUDA
    # print(f'We have CUDA {torch.version.cuda}.')
else:
    DEVICE = CPU
    # print("We DON'T have CUDA.")
