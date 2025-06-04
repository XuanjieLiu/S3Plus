import sys
import os
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from importlib import reload
from two_dim_num_plus_vis import MumEval, gen_k_list
from common_func import load_config_from_exp_name, EXP_ROOT, find_optimal_checkpoint_num_by_train_config

RESULT_DIR_NAME = 'two_dim_z_vis'
EXP_NUM_LIST = [str(i) for i in range(1, 21)]
EXP_NAME_LIST = [
    "2025.06.03_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_Fullsymm_noise0.4",
    "2025.06.03_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_Nothing_noise0.4",
    "2025.06.03_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_Nothing_trainAll_noise0.4",
]


def batch_eval():
    for exp_name in EXP_NAME_LIST:
        exp_path = os.path.join(EXP_ROOT, exp_name)
        result_dir = os.path.join(exp_path, RESULT_DIR_NAME)
        os.makedirs(result_dir, exist_ok=True)
        config = load_config_from_exp_name(exp_name)
        dataset_path = config['single_img_eval_set_path']
        evaler = MumEval(config, None, dataset_path)
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
