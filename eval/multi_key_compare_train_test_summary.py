import sys
import os
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from typing import List
from load_batch_record import ExpGroup
from multi_key_compare import MultiKeyCompareGroup, plot_graph
import numpy as np


exp_groups = [
    ExpGroup(
        exp_name="2023.11.12_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_doubleSet",
        exp_alias='w/ assoc',
        sub_exp=[i for i in range(1, 21)],
        record_name="plus_eval.txt",
    ),
    ExpGroup(
        exp_name="2024.01.20_2023.11.12_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_plusOneSet_assocCommu",
        exp_alias='w/ assoc & commu',
        sub_exp=[i for i in range(1, 21)],
        record_name="plus_eval.txt",
    ),
    ExpGroup(
        exp_name="2024.01.17_2023.11.12_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_plusOneSet_symm",
        exp_alias='w/ symm',
        sub_exp=[i for i in range(1, 21)],
        record_name="plus_eval.txt",
    ),
    ExpGroup(
        exp_name="2024.01.20_2023.11.12_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_plusOneSet_symmCommu",
        exp_alias='w/ symm & commu',
        sub_exp=[i for i in range(1, 21)],
        record_name="plus_eval.txt",
    ),
    ExpGroup(
        exp_name="2024.01.18_2023.11.12_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_plusOneSet_symmAssoc",
        exp_alias='w/ symm & assoc',
        sub_exp=[i for i in range(1, 21)],
        record_name="plus_eval.txt",
    )
]



YTICKS = [i * 0.1 for i in range(0, 11)]
COMPARE_KEYS = ['train_accu', 'eval_accu']
COMPARE_KEYS_NAME = ['TrainSet', 'TestSet']
BEST_MEAN = 5
IS_MAX_BETTER = True
OUTPUT_PATH = "train_test_summary/"
EXTREME_NUM = 5
ITER_AFTER = 20000
Y_NAME = "Plus Accuracy (max=1.0) ↑"
RESULT_NAME = f"{'.'.join(COMPARE_KEYS)}_{exp_groups[0].exp_name}_{len(exp_groups)}.png"


def exp_group2compare_group(exp_group: ExpGroup):
    y_list = []
    for record in exp_group.sub_record:
        y = []
        for key in COMPARE_KEYS:
            record_data = record[key]
            mean = float(np.mean(record_data.find_topMin_after_iter(EXTREME_NUM, ITER_AFTER, IS_MAX_BETTER)))
            y.append(mean)
        y_list.append(y)
    compare_group = MultiKeyCompareGroup(
        title=exp_group.exp_alias,
        y_name=Y_NAME,
        keys=COMPARE_KEYS_NAME,
        values=y_list
    )
    return compare_group


def gen_compare_groups(exp_groups: List[ExpGroup]):
    compare_groups = []
    for eg in exp_groups:
        compare_groups.append(exp_group2compare_group(eg))
    return compare_groups


if __name__ == '__main__':
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    cg_list = gen_compare_groups(exp_groups)
    save_path = os.path.join(OUTPUT_PATH, RESULT_NAME)
    plot_graph(cg_list, save_path, yticks=YTICKS)
