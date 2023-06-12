import os
import sys
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from eval.load_batch_record import ExpGroup
from eval.record_vis_impl import run_eval

eg1 = ExpGroup(
    exp_name='2023.06.09_10vq_Zc[2]_Zs[0]_edim8_plusUnit128.2_encFc128.2_plusS0.3',
    exp_alias='高维度(8 dim),\n 两位数,\n 多冗余(10 tokens),\n (plusS0.3)',
    sub_exp=[19],
    is_load_record=False
)

eg2 = ExpGroup(
    exp_name='2023.06.09_10vq_Zc[2]_Zs[0]_edim8_plusUnit128.2_encFc128.2_plusS1',
    exp_alias='高维度(8 dim),\n 两位数,\n 多冗余(10 tokens),\n (plusS1)',
    sub_exp=[14],
    is_load_record=False
)

eg3 = ExpGroup(
    exp_name='2023.05.26_10vq_Zc[2]_Zs[0]_edim8_plusUnit128.2_encFc128.2_singleS',
    exp_alias='高维度(8 dim),\n 两位数,\n 多冗余(10 tokens),\n (baseline)',
    sub_exp=[10],
    is_load_record=False
)

eg_group = [eg1, eg2, eg3]
if __name__ == '__main__':
    run_eval(eg_group)
