import sys
import os

sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from typing import List
from load_batch_record import ExpGroup
from multi_key_compare import MultiKeyCompareGroup, plot_graph
import numpy as np

# eg1 = ExpGroup(
#     exp_name="2023.11.23_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet",
#     exp_alias='w/ assoc, 10 vq, 2 emb, 1dim',
#     sub_exp=[i for i in range(1, 21)],
#     record_name="plus_eval.txt",
# )
#
# eg2 = ExpGroup(
#     exp_name="2023.11.23_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_noAssoc",
#     exp_alias='w/o assoc, 10 vq, 2 emb, 1dim',
#     sub_exp=[i for i in range(1, 21)],
#     record_name="plus_eval.txt",
# )
#
# eg3 = ExpGroup(
#     exp_name="2023.11.23_100vq_Zc[1]_Zs[0]_edim2_[0-20]_plus1024_1_tripleSet",
#     exp_alias='w/ assoc, 100 vq, 1 emb, 2dim',
#     sub_exp=[i for i in range(1, 21)],
#     record_name="plus_eval.txt",
# )
#
# eg4 = ExpGroup(
#     exp_name="2023.11.23_100vq_Zc[1]_Zs[0]_edim2_[0-20]_plus1024_1_tripleSet_noAssoc",
#     exp_alias='w/o assoc, 100 vq, 1 emb, 2dim',
#     sub_exp=[i for i in range(1, 21)],
#     record_name="plus_eval.txt",
# )
#
#
# exp_groups = [eg1, eg2, eg3, eg4]

exp_groups = [
    ExpGroup(
        exp_name="2024.01.18_2023.11.23_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_assocCommu",
        exp_alias='w/o assoc & commu',
        sub_exp=[i for i in range(1, 21)],
        record_name="plus_eval.txt",
    ),
    ExpGroup(
        exp_name="2024.01.18_2023.11.23_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_symmAssoc",
        exp_alias='w/o symm & assoc',
        sub_exp=[i for i in range(1, 21)],
        record_name="plus_eval.txt",
    ),
    ExpGroup(
        exp_name="2024.01.17_2023.11.23_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_symmCommu",
        exp_alias='w/o symm & commu',
        sub_exp=[i for i in range(1, 21)],
        record_name="plus_eval.txt",
    ),
    ExpGroup(
        exp_name="2023.11.23_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet",
        exp_alias='w/ assoc',
        sub_exp=[i for i in range(1, 21)],
        record_name="plus_eval.txt",
    ),
    ExpGroup(
        exp_name="2024.01.17_2023.11.23_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_symm",
        exp_alias='w/ symm',
        sub_exp=[i for i in range(1, 21)],
        record_name="plus_eval.txt",
    ),
    ExpGroup(
        exp_name="2024.01.17_2023.11.23_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_commu",
        exp_alias='w/ commu',
        sub_exp=[i for i in range(1, 21)],
        record_name="plus_eval.txt",
    ),
    ExpGroup(
        exp_name="2023.11.23_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_noAssoc",
        exp_alias='w/ nothing',
        sub_exp=[i for i in range(1, 21)],
        record_name="plus_eval.txt",
    ),
]



YTICKS = [i * 0.1 for i in range(0, 11)]
BEST_MEAN = 5
IS_MAX_BETTER = True
OUTPUT_PATH = "train_test_summary/"
EXTREME_NUM = 5
ITER_AFTER = 20000
Y_NAME = "Plus Accuracy (max=1.0) ↑"
COMPARE_KEYS_1 = ['train_accu', 'eval_accu_2']
COMPARE_KEYS_NAME_1 = ['TrainSet', 'Comm.TestSet']
RESULT_NAME_1 = f"triple_train2comm_{'.'.join(COMPARE_KEYS_1)}_{eg1.exp_name}_{len(exp_groups)}.png"

COMPARE_KEYS_2 = ['eval_accu_2', 'eval_accu']
COMPARE_KEYS_NAME_2 = ['Comm.TestSet', 'Assoc.TestSet']
RESULT_NAME_2 = f"triple_comm2assoc_{'.'.join(COMPARE_KEYS_2)}_{eg1.exp_name}_{len(exp_groups)}.png"


def exp_group2compare_group(exp_group: ExpGroup, compare_keys, compare_keys_name):
    y_list = []
    for record in exp_group.sub_record:
        y = []
        for key in compare_keys:
            record_data = record[key]
            mean = float(np.mean(record_data.find_topMin_after_iter(EXTREME_NUM, ITER_AFTER, IS_MAX_BETTER)))
            y.append(mean)
        y_list.append(y)
    compare_group = MultiKeyCompareGroup(
        title=exp_group.exp_alias,
        y_name=Y_NAME,
        keys=compare_keys_name,
        values=y_list
    )
    return compare_group


def gen_compare_groups(exp_groups: List[ExpGroup], compare_keys, compare_keys_name):
    compare_groups = []
    for eg in exp_groups:
        compare_groups.append(exp_group2compare_group(eg, compare_keys, compare_keys_name))
    return compare_groups


def compare(compare_keys, compare_keys_name, result_name):
    cg_list = gen_compare_groups(exp_groups, compare_keys, compare_keys_name)
    save_path = os.path.join(OUTPUT_PATH, result_name)
    plot_graph(cg_list, save_path, YTICKS)


if __name__ == '__main__':
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    compare(COMPARE_KEYS_1, COMPARE_KEYS_NAME_1, RESULT_NAME_1)
    compare(COMPARE_KEYS_2, COMPARE_KEYS_NAME_2, RESULT_NAME_2)
