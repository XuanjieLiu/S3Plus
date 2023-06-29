import sys
import os

sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from typing import List
from load_batch_record import ExpGroup
from epoch_compare import EpochCompareGroup, plot_graph


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
    exp_name="2023.06.29_100vq_Zc[1]_Zs[0]_edim8_[0-20]",
    exp_alias='w/ associative loss',
    sub_exp=[i for i in range(1, 21)],
    record_name="plus_eval.txt",
)

eg2 = ExpGroup(
    exp_name="2023.06.29_100vq_Zc[1]_Zs[0]_edim8_[0-20]_noAssoc",
    exp_alias='w/o associative loss',
    sub_exp=[i for i in range(1, 21)],
    record_name="plus_eval.txt",
)


# eg1 = ExpGroup(
#     exp_name="2023.06.18_10vq_Zc[2]_Zs[0]_edim8_plus0.02_switchOff_assocIn_[1-20]",
#     exp_alias='w/ assoc, [1, 20], no-switch',
#     sub_exp=[i for i in range(1, 21)],
#     record_name="plus_eval.txt",
# )
#
# eg2 = ExpGroup(
#     exp_name="2023.06.18_10vq_Zc[2]_Zs[0]_edim8_plus0.02_switch1dig_assocIn_[1-20]",
#     exp_alias='w/ assoc, [1, 20], digit-switch',
#     sub_exp=[i for i in range(1, 21)],
#     record_name="plus_eval.txt",
# )
#
# eg3 = ExpGroup(
#     exp_name="2023.06.18_10vq_Zc[2]_Zs[0]_edim8_plus0.02_switch1dig_assocIn_[1-20]_noAssoc",
#     exp_alias='w/o assoc, [1, 20], digit-switch',
#     sub_exp=[i for i in range(1, 21)],
#     record_name="plus_eval.txt",
# )


exp_groups = [eg1, eg2]
# exp_groups = [eg1, eg2, eg3]


# KEYS = ['train_accu', 'eval_accu']
# KEYS_NAME = ['TrainSet', 'TestSet']
# KEYS_STYLE = ['dotted', 'solid']
KEYS = ['eval_accu']
KEYS_NAME = ['TestSet']
KEYS_STYLE = ['solid']
OUTPUT_PATH = "train_test_epoch/"
Y_NAME = "Plus Accuracy (max=1.0) ↑"
RESULT_NAME = f"{'.'.join(KEYS)}_{eg1.exp_name}.png"

def gen_compare_groups(exp_groups: List[ExpGroup]):
    compare_groups = []
    for eg in exp_groups:
        epoch_group = EpochCompareGroup(
            title=eg.exp_alias,
            y_name=Y_NAME,
            sub_record=eg.sub_record,
            keys=KEYS,
            key_labels=KEYS_NAME,
            keys_linestyle=KEYS_STYLE
        )
        compare_groups.append(epoch_group)
    return compare_groups


if __name__ == '__main__':
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    cg_list = gen_compare_groups(exp_groups)
    save_path = os.path.join(OUTPUT_PATH, RESULT_NAME)
    plot_graph(cg_list, save_path)
