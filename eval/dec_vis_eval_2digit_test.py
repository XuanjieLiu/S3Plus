import sys
import os
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from dec_vis_eval_2digit import ExpGroup, plot_dec_img

eg1 = ExpGroup(
    exp_name="2023.03.25_10vq_Zc[2]_Zs[0]_edim1_singleS_noAssoc",
    sub_exp="13",
    result_path="2023.03.25_10vq_Zc[2]_Zs[0]_edim1_singleS_noAssoc.png",
    check_point="checkpoint_30000.pt"
)


if __name__ == '__main__':
    eg1.run_eval()