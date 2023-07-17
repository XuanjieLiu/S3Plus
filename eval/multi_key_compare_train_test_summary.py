import sys
import os
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from typing import List
from load_batch_record import ExpGroup
from multi_key_compare import MultiKeyCompareGroup, plot_graph
import numpy as np

# eg1 = ExpGroup(
#     exp_name="2023.03.19_10vq_Zc[2]_Zs[0]_edim1_singleS",
#     exp_alias='w/ associative loss',
#     sub_exp=[i for i in range(1, 6)],
#     record_name="plus_eval.txt",
# )
#
# eg2 = ExpGroup(
#     exp_name="2023.03.19_10vq_Zc[2]_Zs[0]_edim1_singleS_noAssoc",
#     exp_alias='w/o associative loss',
#     sub_exp=[i for i in range(1, 6)],
#     record_name="plus_eval.txt",
# )

# eg1 = ExpGroup(
#     exp_name="2023.07.09_[10,10]vq_Zc[2]_Zs[0]_edim8_switch_[1-20]",
#     exp_alias='w/ associative loss',
#     sub_exp=[i for i in range(1, 21)],
#     record_name="plus_eval.txt",
# )
#
# eg2 = ExpGroup(
#     exp_name="2023.07.09_[10,10]vq_Zc[2]_Zs[0]_edim8_switch_[1-20]_noAssoc",
#     exp_alias='w/o associative loss',
#     sub_exp=[i for i in range(1, 21)],
#     record_name="plus_eval.txt",
# )
# exp_groups = [eg1, eg2]

eg1 = ExpGroup(
    exp_name="2023.07.09_10vq_Zc[2]_Zs[0]_edim1_[0-20]",
    exp_alias='10*10, 1dim',
    sub_exp=[i for i in range(1, 21)],
    record_name="plus_eval.txt",
)

eg2 = ExpGroup(
    exp_name="2023.07.09_10vq_Zc[2]_Zs[0]_edim2_[0-20]",
    exp_alias='10*10, 2dim',
    sub_exp=[i for i in range(1, 21)],
    record_name="plus_eval.txt",
)

eg3 = ExpGroup(
    exp_name="2023.07.09_10vq_Zc[2]_Zs[0]_edim8_[0-20]",
    exp_alias='10*10, 8dim',
    sub_exp=[i for i in range(1, 21)],
    record_name="plus_eval.txt",
)

eg4 = ExpGroup(
    exp_name="2023.07.09_10vq_Zc[2]_Zs[0]_edim32_[0-20]",
    exp_alias='10*10, 32dim',
    sub_exp=[i for i in range(1, 21)],
    record_name="plus_eval.txt",
)
eg5 = ExpGroup(
    exp_name="2023.07.09_10vq_Zc[2]_Zs[0]_edim128_[0-20]",
    exp_alias='10*10, 128dim',
    sub_exp=[i for i in range(1, 21)],
    record_name="plus_eval.txt",
)
exp_groups = [eg1, eg2, eg3, eg4, eg5]


YTICKS = [i * 0.1 for i in range(0, 11)]
COMPARE_KEYS = ['train_accu', 'eval_accu']
COMPARE_KEYS_NAME = ['TrainSet', 'TestSet']
BEST_MEAN = 5
IS_MAX_BETTER = True
OUTPUT_PATH = "train_test_summary/"
EXTREME_NUM = 5
ITER_AFTER = 10000
Y_NAME = "Plus Accuracy (max=1.0) ↑"
RESULT_NAME = f"{'.'.join(COMPARE_KEYS)}_{eg1.exp_name}.png"


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
