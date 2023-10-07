import sys
import os
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))


if len(sys.argv) < 2:
    print("Usage: python myscript.py arg1 arg2 ...")
    sys.exit()

EXP_ROOT_PATH = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/exp')
EXP_NAME_LIST = sys.argv[1:]
EXP_NUM_LIST = [str(i) for i in range(1, 21)]
OTHER_TASK_EXP_NUM_LIST = [str(i) for i in range(1, 21)]
