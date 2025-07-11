import sys
import os
import numpy as np
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from importlib import reload
from interpolate_plus_accu_eval import InterpolatePlusAccuEval
from common_func import load_config_from_exp_name, record_num_list, EXP_ROOT, find_optimal_checkpoint_num_by_train_config


EXP_NUM_LIST = [str(i) for i in range(1, 21)]
EXP_NAME_LIST = [
    "2025.06.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyleMahjong_PureVQ",
    "2025.06.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyleMahjong_symm",
    "2025.06.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyleMahjong_trainAll",
    "2025.06.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyleMahjong_nothing",
]
ITP_PLUS_RECORD_PATH_ALL = 'interpolate_plus_accu_all.txt'
ITP_PLUS_RECORD_PATH_TRAIN = 'interpolate_plus_accu_train.txt'


if __name__ == "__main__":
    for exp_name in EXP_NAME_LIST:
        exp_path = os.path.join(EXP_ROOT, exp_name)
        config = load_config_from_exp_name(exp_name)
        evaler = InterpolatePlusAccuEval(config)
        accu_all_list = []
        accu_train_list = []
        for sub_exp in EXP_NUM_LIST:
            sub_exp_path = os.path.join(exp_path, sub_exp)
            optimal_checkpoint_num = find_optimal_checkpoint_num_by_train_config(sub_exp_path, config)
            checkpoint_path = os.path.join(exp_path, sub_exp, f'checkpoint_{optimal_checkpoint_num}.pt')
            accu_all, accu_train = evaler.eval(checkpoint_path)
            accu_all_list.append(accu_all)
            accu_train_list.append(accu_train)
        record_num_list(os.path.join(exp_path, ITP_PLUS_RECORD_PATH_ALL), accu_all_list, EXP_NUM_LIST)
        record_num_list(os.path.join(exp_path, ITP_PLUS_RECORD_PATH_TRAIN), accu_train_list, EXP_NUM_LIST)
