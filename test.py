import os
import sys

import sys

if len(sys.argv) < 2:
    print("Usage: python myscript.py arg1 arg2 ...")
    sys.exit()

arg1 = sys.argv[1]
arg2 = sys.argv[2]
# 依次类推，可以根据需要读取更多的命令行参数

print("arg List: ", sys.argv)
print("arg1 =", arg1)
print("arg2 =", arg2)
# 输出读取到的命令行参数