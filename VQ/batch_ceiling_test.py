import sys
import os
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from importlib import reload
from ceilingTest import CeilTester, is_need_train

if len(sys.argv) < 2:
    print("Usage: python myscript.py arg1 arg2 ...")
    sys.exit()

EXP_ROOT_PATH = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/exp')
sys.path.append(EXP_ROOT_PATH)
EXP_NAME_LIST = sys.argv[1:]
EXP_NUM_LIST = [str(i) for i in range(1, 21)]

for exp_num in EXP_NUM_LIST:
    for exp_name in EXP_NAME_LIST:
        exp_path = os.path.join(EXP_ROOT_PATH, exp_name)
        os.chdir(exp_path)
        sys.path.append(exp_path)
        print(f'Exp path: {exp_path}')
        ceiling_test_config = __import__('ceiling_test_config')
        reload(ceiling_test_config)
        sys.path.pop()
        print(ceiling_test_config.CONFIG)
        sub_exp_path = os.path.join(exp_path, exp_num)
        os.makedirs(exp_num, exist_ok=True)
        print(f'Sub-Exp path: {sub_exp_path}')
        os.chdir(sub_exp_path)
        for i in range(0, len(ceiling_test_config.CONFIG)):
            if is_need_train(ceiling_test_config.CONFIG[i]):
                ceiling_test = CeilTester(ceiling_test_config.CONFIG[i])
                ceiling_test.train()

