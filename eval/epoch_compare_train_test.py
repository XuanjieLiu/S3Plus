import sys
import os

sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from typing import List
from load_batch_record import ExpGroup
from epoch_compare import EpochCompareGroup, plot_graph


eg1 = ExpGroup(
    exp_name="2023.03.19_10vq_Zc[2]_Zs[0]_edim1_singleS",
    exp_alias='w/ associative loss',
    sub_exp=[1, 2, 3, 4, 5],
    record_name="plus_eval.txt",
)

eg2 = ExpGroup(
    exp_name="2023.03.19_10vq_Zc[2]_Zs[0]_edim1_singleS_noAssoc",
    exp_alias='w/o associative loss',
    sub_exp=[1, 2, 3, 4, 5],
    record_name="plus_eval.txt",
)

exp_groups = [eg1, eg2]
# exp_groups_2 = [egd1, egd8, egd64]

KEYS = ['train_accu', 'eval_accu']
KEYS_NAME = ['TrainSet', 'TestSet']
KEYS_STYLE = ['dotted', 'solid']
OUTPUT_PATH = "train_test_epoch/"
Y_NAME = "Plus Accuracy (max=1.0)"
RESULT_NAME = "Assoc_vs_noAssoc.png"

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