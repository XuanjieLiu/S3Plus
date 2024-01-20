import sys
import os
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from importlib import reload
from num_eval import MumEval
from common_func import load_config_from_exp_name, EXP_ROOT

DATASET_PATH = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../dataset/(0,20)-FixedPos-oneStyle')
RESULT_DIR_NAME = 'two_dim_z_vis'
CHECK_POINT = 'checkpoint_40000.pt'
EXP_NUM_LIST = [str(i) for i in range(1, 21)]
EXP_NAME_LIST = [
    "2024.01.18_2023.11.23_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_assocCommu",
    "2024.01.18_2023.11.23_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_symmAssoc",
    "2024.01.17_2023.11.23_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_symmCommu",
    "2023.11.23_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet",
    "2024.01.17_2023.11.23_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_symm",
    "2024.01.17_2023.11.23_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_commu",
    "2023.11.23_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_noAssoc",
]


def batch_eval():
    for exp_name in EXP_NAME_LIST:
        exp_path = os.path.join(EXP_ROOT, exp_name)
        result_dir = os.path.join(exp_path, RESULT_DIR_NAME)
        os.makedirs(result_dir, exist_ok=True)
        config = load_config_from_exp_name(exp_name)
        evaler = MumEval(config, None, DATASET_PATH)
        for sub_exp in EXP_NUM_LIST:
            checkpoint_path = os.path.join(exp_path, sub_exp, CHECK_POINT)
            evaler.reload_model(checkpoint_path)
            result_path = os.path.join(result_dir, f'{sub_exp}_{CHECK_POINT}.png')
            evaler.num_eval_two_dim(result_path)


if __name__ == "__main__":
    batch_eval()
