import sys
import os
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from importlib import reload
from common_func import load_config_from_exp_name, EXP_ROOT, record_num_list, find_optimal_checkpoint_num_by_train_config
from loss_counter import read_record


EXP_NUM_LIST = [str(i) for i in range(1, 21)]
EXP_NAME_LIST = [
    "2024.03.17_10vq_Zc[2]_Zs[0]_edim4_[0-20]_plus1024_1_tripleSet_AssocCommuAll",
    "2024.03.17_10vq_Zc[2]_Zs[0]_edim4_[0-20]_plus1024_1_tripleSet_AssocSymmCommuAll",
    "2024.03.17_10vq_Zc[2]_Zs[0]_edim4_[0-20]_plus1024_1_tripleSet_CommuAll",
    "2024.03.17_10vq_Zc[2]_Zs[0]_edim4_[0-20]_plus1024_1_tripleSet_SymmCommuAll",
    "2024.03.17_10vq_Zc[2]_Zs[0]_edim4_[0-20]_plus1024_1_tripleSet_Assoc",
    "2024.03.17_10vq_Zc[2]_Zs[0]_edim4_[0-20]_plus1024_1_tripleSet_AssocSymm",
    "2024.03.17_10vq_Zc[2]_Zs[0]_edim4_[0-20]_plus1024_1_tripleSet_Symm",
    "2024.03.17_10vq_Zc[2]_Zs[0]_edim4_[0-20]_plus1024_1_tripleSet_Nothing",
]
EVAL_RECORD_NAME = 'plus_eval.txt'
EVAL_KEYS = ['train_accu', 'eval_accu', 'eval_accu_2']


def batch_statistic():
    for exp_name in EXP_NAME_LIST:
        record_lists = [[] for _ in range(len(EVAL_KEYS))]
        exp_path = os.path.join(EXP_ROOT, exp_name)
        config = load_config_from_exp_name(exp_name)
        for sub_exp in EXP_NUM_LIST:
            sub_exp_path = os.path.join(exp_path, sub_exp)
            optimal_checkpoint_num = find_optimal_checkpoint_num_by_train_config(sub_exp_path, config)
            eval_txt_path = os.path.join(sub_exp_path, EVAL_RECORD_NAME)
            eval_record = read_record(eval_txt_path)
            for i in range(len(EVAL_KEYS)):
                key = EVAL_KEYS[i]
                if key in eval_record.keys():
                    value = eval_record[key].filter_Y_by_X_nums([optimal_checkpoint_num])[1][0]
                    record_lists[i].append(value)
        for i in range(len(EVAL_KEYS)):
            key = EVAL_KEYS[i]
            results_path = os.path.join(exp_path, f'statistic_{key}.txt')
            record_num_list(results_path, record_lists[i], EXP_NUM_LIST)


if __name__ == "__main__":
    batch_statistic()

