import sys
import os
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from typing import List
from load_batch_record import ExpGroup
from multi_key_compare import MultiKeyCompareGroup, plot_graph
import numpy as np


exp_groups = [
    ExpGroup(
        exp_name="2024.03.23_ceilingTest_book2_fc1024_16_dim8",
        exp_alias='FC: 1024×1, Linear Repr',
        sub_exp=[i for i in range(1, 21)],
        record_name="book2_minus_1024_1_linear_eval_record.txt",
    ),
    ExpGroup(
        exp_name="2024.03.23_ceilingTest_book2_fc1024_16_dim8",
        exp_alias='FC: 16×1, Linear Repr',
        sub_exp=[i for i in range(1, 21)],
        record_name="book2_minus_1024_1_linear_eval_record.txt",
    ),
    ExpGroup(
        exp_name="2024.03.23_ceilingTest_book2_fc1024_16_dim8",
        exp_alias='FC: 16×1, Random Repr',
        sub_exp=[i for i in range(1, 21)],
        record_name="book2_minus_16_1_random_eval_record.txt",
    )
]



COMPARE_KEYS = ['accu', 'loss_recon']
COMPARE_KEYS_NAME = ['Accuracy', 'Repr_pred_loss']
IS_MAX_BETTER = [True, False]
OUTPUT_PATH = "train_test_summary/"
EXTREME_NUM = 1
ITER_AFTER = 2000
Y_NAME = "Plus Accuracy (max=1.0) ↑"


def exp_group2compare_group(exp_group: ExpGroup):
    y_list = []
    for record in exp_group.sub_record:
        y = []
        i = 0
        for key in COMPARE_KEYS:
            record_data = record[key]
            mean = float(np.mean(record_data.find_topMin_after_iter(EXTREME_NUM, ITER_AFTER, IS_MAX_BETTER[i])))
            y.append(mean)
            i += 1
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
    for cg in cg_list:
        print(f'Exp: {cg.title}')
        for i in range(0, len(cg.keys)):
            values = [item[i] for item in cg.values]
            mean = float(np.mean(values))
            std = float(np.std(values))
            print(f'{cg.keys[i]}: {round(mean, ndigits=2)} ± {round(std, ndigits=2)}')
        print('\n')

