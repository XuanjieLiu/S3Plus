import os
import random
import sys
import math
from scipy.stats import mode
import sys
import numpy as np

# if len(sys.argv) < 2:
#     print("Usage: python myscript.py arg1 arg2 ...")
#     sys.exit()
#
# arg1 = sys.argv[1]
# arg2 = sys.argv[2]
# # 依次类推，可以根据需要读取更多的命令行参数
#
# print("arg List: ", sys.argv)
# print("arg1 =", arg1)
# print("arg2 =", arg2)
# # 输出读取到的命令行参数


# class TestA:
#     def __init__(self, a):
#         self.a = a
#
# x = TestA(3)
# y = [x for n in range(2)]
# for item in y:
#     print(item.a)
# x.a = 8
# for item in y:
#     print(item.a)

# d = {
#     '1': 1,
#     '2': 2,
#     '1.4': 1.5,
# }
#
# print(d[str(1.4)])

# a = pow(math.pi, 2)
# b = pow(math.pi, 2.5)
# c = pow(math.e, 3)
# print(pow(a, 2) + pow(b, 2), pow(c, 2))
# print(pow(math.pi, 3))


# import numpy as np
# # L2 norm
# def l2_norm(a):
#     return np.linalg.norm(a)
#
# def relative_change(a, b):
#     return l2_norm(a-b) / l2_norm(b)
#
# # random a vector
# a = np.random.rand(10)
# b = np.random.rand(10)
# c = 3*b
#
# print(relative_change(c, b))
# #print(relative_change(a*2, b*2))


# test mode
labels = [2,2,3,3,2,4,5,3,3,2]
mode_label = mode(labels, keepdims=False)[0]
print (f"Mode label: {mode_label}")

