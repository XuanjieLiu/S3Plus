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
#     exp_name="2023.08.14_10vq_Zc[2]_Zs[0]_edim12_[0-20]_plus512_2",
#     # exp_alias='w/ associative loss',
#     exp_alias='12dim 512_2',
#     sub_exp=[i for i in range(1, 21)],
#     record_name="plus_eval.txt",
# )
#
# eg2 = ExpGroup(
#     exp_name="2023.08.14_10vq_Zc[2]_Zs[0]_edim12_[0-20]_plus1024_2",
#     exp_alias='12dim 1024_2',
#     sub_exp=[i for i in range(1, 21)],
#     record_name="plus_eval.txt",
# )
# exp_groups = [eg1, eg2]

eg1 = ExpGroup(
    exp_name="2023.08.15_2vq_Zc[6]_Zs[0]_edim4_[0-20]_plus1024_2",
    exp_alias='2^6 (dim=24, comb=64)',
    sub_exp=[i for i in range(1, 21)],
    record_name="plus_eval.txt",
)

eg2 = ExpGroup(
    exp_name="2023.08.15_2vq_Zc[7]_Zs[0]_edim3_[0-20]_plus1024_2",
    exp_alias='2^7 (dim=21, comb=128)',
    sub_exp=[i for i in range(1, 21)],
    record_name="plus_eval.txt",
)

eg3 = ExpGroup(
    exp_name="2023.08.15_3vq_Zc[4]_Zs[0]_edim6_[0-20]_plus1024_2",
    exp_alias='3^4 (dim=24, comb=81)',
    sub_exp=[i for i in range(1, 21)],
    record_name="plus_eval.txt",
)

eg4 = ExpGroup(
    exp_name="2023.08.15_5vq_Zc[3]_Zs[0]_edim8_[0-20]_plus1024_2",
    exp_alias='5^3 (dim=24, comb=125)',
    sub_exp=[i for i in range(1, 21)],
    record_name="plus_eval.txt",
)

eg5 = ExpGroup(
    exp_name="2023.08.14_10vq_Zc[2]_Zs[0]_edim12_[0-20]_plus1024_2",
    exp_alias='10^2 (dim=24, comb=100)',
    sub_exp=[i for i in range(1, 21)],
    record_name="plus_eval.txt",
)

eg6 = ExpGroup(
    exp_name="2023.08.15_100vq_Zc[1]_Zs[0]_edim24_[0-20]_plus1024_2",
    exp_alias='100^1 (dim=24, comb=100)',
    sub_exp=[i for i in range(1, 21)],
    record_name="plus_eval.txt",
)
exp_groups = [eg1, eg2, eg3, eg4, eg5, eg6]


YTICKS = [i * 0.1 for i in range(0, 11)]
COMPARE_KEYS = ['train_accu', 'eval_accu']
COMPARE_KEYS_NAME = ['TrainSet', 'TestSet']
BEST_MEAN = 5
IS_MAX_BETTER = True
OUTPUT_PATH = "train_test_summary/"
EXTREME_NUM = 5
ITER_AFTER = 10000
Y_NAME = "Plus Accuracy (max=1.0) ↑"
RESULT_NAME = f"{'.'.join(COMPARE_KEYS)}_{eg1.exp_name}_{len(exp_groups)}.png"


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
