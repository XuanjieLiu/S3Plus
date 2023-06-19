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
#     exp_name="2023.06.18_10vq_Zc[2]_Zs[0]_edim8_plus0.02_switch1dig_assocIn_[0-20]",
#     exp_alias='w/ associative loss',
#     sub_exp=[i for i in range(1, 21)],
#     record_name="plus_eval.txt",
# )
#
# eg2 = ExpGroup(
#     exp_name="2023.06.18_10vq_Zc[2]_Zs[0]_edim8_plus0.02_switch1dig_assocIn_[0-20]_noAssoc",
#     exp_alias='w/o associative loss',
#     sub_exp=[i for i in range(1, 21)],
#     record_name="plus_eval.txt",
# )



eg1 = ExpGroup(
    exp_name="2023.06.18_10vq_Zc[2]_Zs[0]_edim8_plus0.02_switchOff_assocIn_[0-20]",
    exp_alias='w/ assoc, [0, 20], no-switch',
    sub_exp=[i for i in range(1, 21)],
    record_name="Eval_record.txt",
)

eg2 = ExpGroup(
    exp_name="2023.06.18_10vq_Zc[2]_Zs[0]_edim8_plus0.02_switch1dig_assocIn_[0-20]",
    exp_alias='w/ assoc, [0, 20], digit-switch',
    sub_exp=[i for i in range(1, 21)],
    record_name="Eval_record.txt",
)

eg3 = ExpGroup(
    exp_name="2023.06.18_10vq_Zc[2]_Zs[0]_edim8_plus0.02_switch1dig_assocIn_[0-20]_noAssoc",
    exp_alias='w/0 assoc, [0, 20], digit-switch',
    sub_exp=[i for i in range(1, 21)],
    record_name="Eval_record.txt",
)




# exp_groups = [eg1, eg2]
exp_groups = [eg1, eg2, eg3]


YTICKS = [i * 0.01 for i in range(0, 11)]
COMPARE_KEYS = ['loss_ED', 'plus_recon']
COMPARE_KEYS_NAME = ['Self-recon', 'Plus Recon Loss']
BEST_MEAN = 5
IS_MAX_BETTER = True
OUTPUT_PATH = "train_test_summary/"
EXTREME_NUM = 5
ITER_AFTER = 10000
Y_NAME = "Recon Loss ↓"
RESULT_NAME = f"{'.'.join(COMPARE_KEYS)}_{eg1.exp_name}(plus_recon).png"


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
    plot_graph(cg_list, save_path, YTICKS)
