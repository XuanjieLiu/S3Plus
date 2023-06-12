import os
import sys
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from eval.load_batch_record import ExpGroup
from eval.record_vis_impl import run_eval

eg1 = ExpGroup(
    exp_name='2023.05.26_10vq_Zc[2]_Zs[0]_edim8_plusUnit128.2_encFc128.2_singleS',
    exp_alias='Normal result',
    sub_exp=[16],
    is_load_record=False
)

eg2 = ExpGroup(
    exp_name='2023.05.26_10vq_Zc[2]_Zs[0]_edim8_plusUnit128.2_encFc128.2_singleS',
    exp_alias='Bad result',
    sub_exp=[12],
    is_load_record=False
)

eg3 = ExpGroup(
    exp_name='2023.05.26_10vq_Zc[2]_Zs[0]_edim8_plusUnit128.2_encFc128.2_singleS',
    exp_alias='Good result',
    sub_exp=[10],
    is_load_record=False
)

eg_group = [eg1, eg2, eg3]
if __name__ == '__main__':
    run_eval(eg_group)
