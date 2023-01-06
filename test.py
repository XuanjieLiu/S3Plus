import torch
import torch.nn as nn

import random
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt

# def get_sample_prob(step):
#     alpha = 2200
#     beta = 6000
#     return alpha / (alpha + np.exp((step + beta) / alpha))
#
#
# trend = [get_sample_prob(a) for a in range(0, 50000)]
# plt.plot(trend)
# plt.show()
# data_root = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/')
# print(data_root)

# a = torch.tensor(0.2)
# b = torch.tensor(0.5)
# c = torch.stack((a, b), dim=0)
# d = torch.std(c)
# m = torch.mean(c)
# e = nn.MSELoss()(a, b)
# print(e/torch.pow(d, 2))

# embedding = nn.Embedding(10, 3)
# input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
#
#
# def get_code_indices(flat_x):
#     embeddings = nn.Embedding(3, 5)
#     print("embedding: ")
#     print(embeddings)
#     # compute L2 distance
#     distances = (
#             torch.sum(flat_x ** 2, dim=1, keepdim=True) +
#             torch.sum(embeddings.weight ** 2, dim=1) -
#             2. * torch.matmul(flat_x, embeddings.weight.t())
#     )  # [N, M]
#     encoding_indices = torch.argmin(distances, dim=1)  # [N,]
#     print(encoding_indices)
#     return encoding_indices
#
# x = torch.ones(2, 5)
# get_code_indices(x)

print(np.ones(3))
