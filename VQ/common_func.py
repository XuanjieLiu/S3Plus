import sys
import os
import numpy as np

sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from importlib import reload

DATASET_ROOT = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../dataset/')
EXP_ROOT = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/exp')


def load_config_from_exp_name(exp_name, exp_root=EXP_ROOT):
    exp_path = os.path.join(exp_root, exp_name)
    sys.path.append(exp_path)
    os.chdir(exp_path)
    print(f'Exp path: {exp_path}')
    t_config = __import__('train_config')
    reload(t_config)
    sys.path.pop()
    return t_config.CONFIG

def record_num_list(record_path, accu_list, exp_num_list=None):
    if exp_num_list is None:
        exp_num_list = [str(i) for i in range(1, 21)]
    mean_accu = sum(accu_list) / len(accu_list)
    std_accu = np.std(accu_list)
    with open(record_path, 'w') as f:
        f.write(f'Mean accu: {mean_accu}\n')
        f.write(f'Std accu: {std_accu}\n')
        for i in range(len(accu_list)):
            f.write(f'Exp {exp_num_list[i]}: {accu_list[i]}\n')
