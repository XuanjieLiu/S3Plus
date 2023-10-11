import sys
import os
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from typing import List
from load_batch_record import ExpGroup
from multi_key_compare import MultiKeyCompareGroup, plot_graph
import numpy as np



# eg1 = ExpGroup(
#     exp_name="2023.09.18_ceilingTest_book10_fc8",
#     exp_alias='fc 8',
#     sub_exp=[i for i in range(1, 21)],
#     record_name="book10_minus_8_1_linear_eval_record.txt",
# )
#
# eg2 = ExpGroup(
#     exp_name="2023.09.18_ceilingTest_book10_fc16",
#     exp_alias='fc 16',
#     sub_exp=[i for i in range(1, 21)],
#     record_name="book10_minus_16_1_linear_eval_record.txt",
# )
#
# eg3 = ExpGroup(
#     exp_name="2023.09.18_ceilingTest_book10_fc32",
#     exp_alias='fc 32',
#     sub_exp=[i for i in range(1, 21)],
#     record_name="book10_minus_32_1_linear_eval_record.txt",
# )
#
# eg4 = ExpGroup(
#     exp_name="2023.09.18_ceilingTest_book10_fc128",
#     exp_alias='fc 128',
#     sub_exp=[i for i in range(1, 21)],
#     record_name="book10_minus_128_1_linear_eval_record.txt",
# )
#
# eg5 = ExpGroup(
#     exp_name="2023.09.18_ceilingTest_book10_fc1024",
#     exp_alias='fc 1024',
#     sub_exp=[i for i in range(1, 21)],
#     record_name="book10_minus_1024_1_linear_eval_record.txt",
# )

eg1 = ExpGroup(
    exp_name="2023.09.24_10vq_Zc[2]_Zs[0]_edim1_[0-20]_noPlus",
    exp_alias='No Plus (book_size=10, emb_num=2, emb_dim=1)',
    sub_exp=[i for i in range(1, 21)],
    record_name="minus_16_1_eval_record.txt",
)

eg2 = ExpGroup(
    exp_name="2023.09.24_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_noAssoc",
    exp_alias='w/o assoc (book_size=10, emb_num=2, emb_dim=1)',
    sub_exp=[i for i in range(1, 21)],
    record_name="minus_16_1_eval_record.txt",
)

eg3 = ExpGroup(
    exp_name="2023.09.24_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1",
    exp_alias='w/ (book_size=10, emb_num=2, emb_dim=1)',
    sub_exp=[i for i in range(1, 21)],
    record_name="minus_16_1_eval_record.txt",
)

eg4 = ExpGroup(
    exp_name="2023.09.24_100vq_Zc[1]_Zs[0]_edim2_[0-20]_noPlus",
    exp_alias='No Plus (book_size=100, emb_num=1, emb_dim=2)',
    sub_exp=[i for i in range(1, 21)],
    record_name="minus_16_1_eval_record.txt",
)

eg5 = ExpGroup(
    exp_name="2023.09.25_100vq_Zc[1]_Zs[0]_edim2_[0-20]_plus1024_1_noAssoc",
    exp_alias='w/o assoc (book_size=100, emb_num=1, emb_dim=2)',
    sub_exp=[i for i in range(1, 21)],
    record_name="minus_16_1_eval_record.txt",
)

eg6 = ExpGroup(
    exp_name="2023.09.25_100vq_Zc[1]_Zs[0]_edim2_[0-20]_plus1024_1",
    exp_alias='w/ assoc (book_size=100, emb_num=1, emb_dim=2)',
    sub_exp=[i for i in range(1, 21)],
    record_name="minus_16_1_eval_record.txt",
)
exp_groups = [eg1, eg2, eg3, eg4, eg5, eg6]


# eg1 = ExpGroup(
#     exp_name="2023.09.18_ceilingTest_book10_fc16",
#     exp_alias='FC: 16×1, Linear Repr',
#     sub_exp=[i for i in range(1, 21)],
#     record_name="book10_minus_16_1_linear_eval_record.txt",
# )
#
# eg2 = ExpGroup(
#     exp_name="2023.09.18_ceilingTest_book10_fc16",
#     exp_alias='FC: 16×1, Random Repr',
#     sub_exp=[i for i in range(1, 21)],
#     record_name="book10_minus_16_1_random_eval_record.txt",
# )
# exp_groups = [eg1, eg2]

COMPARE_KEYS = ['accu', 'loss_recon']
COMPARE_KEYS_NAME = ['Accuracy', 'Repr_pred_loss']
IS_MAX_BETTER = [True, False]
OUTPUT_PATH = "train_test_summary/"
EXTREME_NUM = 1
ITER_AFTER = 2000
Y_NAME = "Plus Accuracy (max=1.0) ↑"
RESULT_NAME = f"{'.'.join(COMPARE_KEYS)}_{eg1.exp_name}_{len(exp_groups)}.png"


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
            print(f'{cg.keys[i]}: {sum([v[i] for v in cg.values])/len(cg.values)}')
        print('\n')

