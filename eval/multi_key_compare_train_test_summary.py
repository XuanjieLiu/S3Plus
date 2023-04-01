import sys
import os
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from typing import List
from load_batch_record import ExpGroup
from multi_key_compare import MultiKeyCompareGroup, plot_graph
import numpy as np

eg1 = ExpGroup(
    exp_name="2023.03.25_10vq_Zc[2]_Zs[0]_edim1_singleS",
    exp_alias='w/ associative loss',
    sub_exp=[i for i in range(1, 21)],
    record_name="plus_eval.txt",
)

eg2 = ExpGroup(
    exp_name="2023.03.25_10vq_Zc[2]_Zs[0]_edim1_singleS_noAssoc",
    exp_alias='w/o associative loss',
    sub_exp=[i for i in range(1, 21)],
    record_name="plus_eval.txt",
)

# egd1 = ExpGroup(
#     exp_name="2023.03.19_10vq_Zc[2]_Zs[0]_edim1_singleS",
#     exp_alias='emb_dim = 1',
#     sub_exp=[1, 2, 3, 4, 5],
#     record_name="plus_eval.txt",
# )
#
# egd8 = ExpGroup(
#     exp_name="2023.03.19_10vq_Zc[2]_Zs[0]_edim8_singleS",
#     exp_alias='emb_dim = 8',
#     sub_exp=[1, 2, 3, 4, 5],
#     record_name="plus_eval.txt",
# )
#
# egd64 = ExpGroup(
#     exp_name="2023.03.19_10vq_Zc[2]_Zs[0]_edim64_singleS",
#     exp_alias='emb_dim = 64',
#     sub_exp=[1, 2, 3, 4, 5],
#     record_name="plus_eval.txt",
# )

exp_groups = [eg1, eg2]
# exp_groups_2 = [egd1, egd8, egd64]

COMPARE_KEYS = ['train_accu', 'eval_accu']
COMPARE_KEYS_NAME = ['TrainSet', 'TestSet']
BEST_MEAN = 5
IS_MAX_BETTER = True
OUTPUT_PATH = "train_test_summary/"
EXTREME_NUM = 5
ITER_AFTER = 8000
Y_NAME = "Plus Accuracy (max=1.0)"
RESULT_NAME = "dim_1_3.25_compare.png"


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
    plot_graph(cg_list, save_path)
