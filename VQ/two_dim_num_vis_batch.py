import sys
import os
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from importlib import reload
from two_dim_num_vis import MumEval
from common_func import load_config_from_exp_name, EXP_ROOT, find_optimal_checkpoint_num_by_train_config

DATASET_PATH = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../dataset/(1,20)-FixedPos-oneStyle')
RESULT_DIR_NAME = 'two_dim_z_vis'
EXP_NUM_LIST = [str(i) for i in range(1, 21)]
EXP_NAME_LIST = [
    "2024.02.03_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_AssocCommuAll",
    "2024.02.03_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_AssocSymmCommuAll",
    "2024.02.03_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_commuAll",
    "2024.02.03_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_SymmCommuAll",
    "2024.02.28_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_Assoc",
    "2024.02.28_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_AssocSymm",
    "2024.02.28_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_Symm",
    "2024.02.28_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_Nothing",
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
            result_path = os.path.join(result_dir, f'{sub_exp}_{check_point_name}.png')
            evaler.num_eval_two_dim(result_path)


if __name__ == "__main__":
    batch_eval()
