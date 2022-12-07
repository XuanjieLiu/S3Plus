import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt

# def get_sample_prob(step):
#     alpha = 2200
#     beta = 8000
#     return alpha / (alpha + np.exp((step + beta) / alpha))
#
#
# trend = [get_sample_prob(a) for a in range(0, 50000)]
# plt.plot(trend)
# plt.show()
# data_root = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/')
# print(data_root)

a = torch.tensor(0.2)
b = torch.tensor(0.5)
c = torch.stack((a, b), dim=0)
d = torch.std(c)
m = torch.mean(c)
e = nn.MSELoss()(a, b)
print(e/torch.pow(d, 2))