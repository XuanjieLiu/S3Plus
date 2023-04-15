import sys
import os
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from dec_vis_eval_2digit import ExpGroup, plot_dec_img

eg1 = ExpGroup(
    exp_name="2023.04.06_5vq_Zc[2]_Zs[0]_edim8_singleS_plusUnit128_noAssoc",
    sub_exp="2",
    result_path="2023.04.06_5vq_Zc[2]_Zs[0]_edim8_singleS_plusUnit128_noAssoc.png",
    check_point="checkpoint_36000.pt"
)


if __name__ == '__main__':
    eg1.run_eval()