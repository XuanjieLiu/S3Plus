import sys
import os
import numpy as np
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from importlib import reload
from interpolate_plus_accu_eval import InterpolatePlusAccuEval

# EXP_NUM_LIST = ['16']
# EXP_NAME_LIST = [
#     '2023.09.25_100vq_Zc[1]_Zs[0]_edim2_[0-20]_plus1024_1'
# ]


EXP_ROOT_PATH = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/exp')
CHECK_POINT = 'checkpoint_50000.pt'
EXP_NUM_LIST = [str(i) for i in range(1, 21)]
EXP_NAME_LIST = [
    '2023.11.23_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet',
    '2023.11.23_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_noAssoc',
    '2023.11.23_100vq_Zc[1]_Zs[0]_edim2_[0-20]_plus1024_1_tripleSet',
    '2023.11.23_100vq_Zc[1]_Zs[0]_edim2_[0-20]_plus1024_1_tripleSet_noAssoc',
]
ITP_PLUS_RECORD_PATH_ALL = 'interpolate_plus_accu_all.txt'
ITP_PLUS_RECORD_PATH_TRAIN = 'interpolate_plus_accu_train.txt'

def init_an_evaler(exp_name):
    exp_path = os.path.join(EXP_ROOT_PATH, exp_name)
    sys.path.append(exp_path)
    os.chdir(exp_path)
    print(f'Exp path: {exp_path}')
    t_config = __import__('train_config')
    reload(t_config)
    sys.path.pop()
    evaler = InterpolatePlusAccuEval(t_config.CONFIG)
    return evaler


def record(record_path, accu_list):
    mean_accu = sum(accu_list) / len(accu_list)
    std_accu = np.std(accu_list)
    with open(record_path, 'w') as f:
        f.write(f'Mean accu: {mean_accu}\n')
        f.write(f'Std accu: {std_accu}\n')
        for i in range(len(accu_list)):
            f.write(f'Exp {EXP_NUM_LIST[i]}: {accu_list[i]}\n')


if __name__ == "__main__":
    for exp_name in EXP_NAME_LIST:
        exp_path = os.path.join(EXP_ROOT_PATH, exp_name)
        evaler = init_an_evaler(exp_name)
        accu_all_list = []
        accu_train_list = []
        for sub_exp in EXP_NUM_LIST:
            checkpoint_path = os.path.join(exp_path, sub_exp, CHECK_POINT)
            accu_all, accu_train = evaler.eval(checkpoint_path)
            accu_all_list.append(accu_all)
            accu_train_list.append(accu_train)
        record(os.path.join(exp_path, ITP_PLUS_RECORD_PATH_ALL), accu_all_list)
        record(os.path.join(exp_path, ITP_PLUS_RECORD_PATH_TRAIN), accu_train_list)
