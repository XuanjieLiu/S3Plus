import sys
import os
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from importlib import reload
from two_dim_num_plus_vis import MumEval, gen_k_list
from common_func import load_config_from_exp_name, EXP_ROOT, find_optimal_checkpoint_num_by_train_config

DATASET_PATH = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../dataset/(0,20)-FixedPos-oneStyle')
RESULT_DIR_NAME = 'two_dim_z_vis'
EXP_NUM_LIST = [str(i) for i in range(1, 21)]
EXP_NAME_LIST = [
    # "2024.02.03_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_AssocSymmCommuAll",
    # "2024.03.30_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_oneColSet_Fullsymm_Out",
    # "2024.03.30_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_oneShotSet_Fullsymm_Out",
    # "2024.03.30_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_Fullsymm_Out",
    # "2024.03.28_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_oneShotSet_AssocFullsymmCommuAll",
    # "2024.03.06_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_oneShotSet_Nothing",
    # "2024.02.28_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_Nothing",

    # "2024.03.06_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_oneShotSet_AssocSymmCommuAll",
    # "2024.03.28_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_oneColSet_AssocFullsymmCommuAll",
    # "2024.02.03_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_oneColSet_AssocSymmCommuAll",
    # "2024.02.08_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_oneColSet_nothing",

    "2024.04.07_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_AssocFullsymm",
    "2024.04.07_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_AssocFullsymmCommu",
]


def batch_eval():
    for exp_name in EXP_NAME_LIST:
        exp_path = os.path.join(EXP_ROOT, exp_name)
        result_dir = os.path.join(exp_path, RESULT_DIR_NAME)
        os.makedirs(result_dir, exist_ok=True)
        config = load_config_from_exp_name(exp_name)
        evaler = MumEval(config, None, DATASET_PATH)
        for sub_exp in EXP_NUM_LIST:
            sub_exp_path = os.path.join(exp_path, sub_exp)
            optimal_checkpoint_num = find_optimal_checkpoint_num_by_train_config(sub_exp_path, config)
            check_point_name = f'checkpoint_{optimal_checkpoint_num}.pt'
            checkpoint_path = os.path.join(exp_path, sub_exp, check_point_name)
            evaler.reload_model(checkpoint_path)
            result_path = os.path.join(result_dir, f'{sub_exp}_{check_point_name}')
            evaler.num_eval_two_dim_int_plus(result_path)
            k_lists = [gen_k_list(0.2, 4, i) for i in range(0, 6)]
            evaler.eval_multi_decimal_plus(k_lists, result_path)


if __name__ == "__main__":
    batch_eval()
