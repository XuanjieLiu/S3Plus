import sys
import os
from load_batch_record import ExpGroup
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from multi_key_compare import MultiKeyCompareGroup
import numpy as np
from typing import List
from loss_counter import LossCounter, RECORD_PATH_DEFAULT


eg1 = ExpGroup(
    exp_name="2023.09.25_100vq_Zc[1]_Zs[0]_edim2_[0-20]_plus1024_1_noAssoc",
    exp_alias='fc 16',
    sub_exp=[i for i in range(1, 21)],
    record_name="minus_16_1_eval_record.txt",
    is_load_record=False
)

EXP_ROOT_PATH = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../VQ/exp')
EXP_NUM_LIST = [str(i) for i in range(1, 21)]
OTHER_TASK_EXP_NUM_LIST = [str(i) for i in range(1, 21)]


COMPARE_KEYS = ['accu', 'loss_recon']
COMPARE_KEYS_NAME = ['Accuracy', 'Repr_pred_loss']
IS_MAX_BETTER = [True, False]
OUTPUT_PATH = "train_test_summary/"
EXTREME_NUM = 10
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

def group_to_subgroups(eg: ExpGroup):
    sub_eg_list = []
    for other_task in EXP_NUM_LIST:
        sub_eg = ExpGroup(
            exp_name=os.path.join(eg.exp_name, other_task),
            exp_alias=eg.exp_alias,
            sub_exp=OTHER_TASK_EXP_NUM_LIST,
            record_name=eg.record_name,
        )
        sub_eg_list.append(sub_eg)
    return sub_eg_list

def summary_a_group(eg: ExpGroup):
    sub_eg_list = group_to_subgroups(eg)
    cg_list = gen_compare_groups(sub_eg_list)
    for i in range(0, len(cg_list)):
        summary_path = os.path.join(EXP_ROOT_PATH, sub_eg_list[i].exp_name, eg.record_name)
        loss_counter = LossCounter(COMPARE_KEYS, summary_path)
        mean_values = []
        for j in range(0, len(COMPARE_KEYS)):
            mean_value = np.mean([value[j] for value in cg_list[i].values])
            mean_values.append(mean_value)
        loss_counter.add_values(mean_values)
        loss_counter.record_and_clear(RECORD_PATH_DEFAULT, 10000)
    print('aaa')



if __name__ == '__main__':
    summary_a_group(eg1)