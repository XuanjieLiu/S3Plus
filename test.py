import torch
import torch.nn as nn

import random
import torch.nn.functional as F
import os
import numpy as np
import random
from scipy import stats
# import matplotlib.pyplot as plt

"""Embedding test"""
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


# """Ling ren Knight test"""
# def dps(bounce_num, decay):
#     init_dmg = 1
#     total_dmg = 1
#     for i in range(bounce_num):
#         init_dmg *= decay
#         total_dmg += init_dmg
#     print(total_dmg)
#
#
#
#
# print(4.28/3.05)
# print(4.97/3.24)
def calc_accu(s1, s2):
    mode_enc = stats.mode(s1, keepdims=False)[0]
    total = len(s2)
    correct = len(list(filter(lambda x: x == mode_enc, s2)))
    return correct / total

sample1 = [3,4.0,4,4]
sample2 = [3, 3, 4, 4]
# ks_statistic, p_value = stats.ks_2samp(sample1, sample2)
# print(ks_statistic)
# print(p_value)
#
# a = [0.1,0.2,0.1]
# a = [i + np.mean(a) for i in a]
# b = [0.01,0.02,0.01] + [1 for j in range(3)]
# b = [i + np.mean(b) for i in b]
# print(np.std(a))
# print(np.std(b))
print(stats.mode(sample1, keepdims=False)[0])
print(calc_accu(sample1, sample2))

EXP_NUM_LIST = [str(i) for i in range(1, 21)]
print(EXP_NUM_LIST)
