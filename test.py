import os
import random
import sys

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

a = [1,2,3,4,5]
# print(a[-3])
# print(a[2])
for i in range(10):
    print(random.randint(0,3))