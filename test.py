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


def mode_of_rows(matrix):
    # 计算每一行的众数
    return mode(matrix, axis=1, keepdims=True)  # mode函数返回众数和它出现的频次，这里只取众数

def mode_of_columns(matrix):
    # 计算每一列的众数
    return mode(matrix, axis=0, keepdims=True)  # 同上，只取众数

# 示例使用
matrix = np.array([[1, 2, 3, 3], [4, 4, 5, 6], [7, 8, 9, 9]])

# 计算行众数和列众数
row_modes = mode_of_rows(matrix)
column_modes = mode_of_columns(matrix)

print("Row modes:", row_modes)
print("Column modes:", column_modes)


