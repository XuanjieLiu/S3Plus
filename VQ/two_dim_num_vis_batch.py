import sys
import os
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from importlib import reload
from two_dim_num_vis import MumEval
from common_func import load_config_from_exp_name, EXP_ROOT, find_optimal_checkpoint_num_by_train_config, record_num_list
from torch.utils.data import DataLoader
from dataloader import SingleImgDataset

RESULT_DIR_NAME = 'two_dim_z_vis'
EXP_NUM_LIST = [str(i) for i in range(1, 21)]
EXP_NAME_LIST = [
    "2024.05.31_5vq_Zc[3]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_AssocFullsymmCommu",
    "2024.05.31_10vq_Zc[3]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_AssocFullsymmCommu",
]

RESULT_NAME = 'near_neighbour_score.txt'

def batch_eval():
    for exp_name in EXP_NAME_LIST:
        exp_path = os.path.join(EXP_ROOT, exp_name)
        result_dir = os.path.join(exp_path, RESULT_DIR_NAME)
        os.makedirs(result_dir, exist_ok=True)
        config = load_config_from_exp_name(exp_name)
        dataset_path = config['single_img_eval_set_path']
        single_img_eval_set = SingleImgDataset(dataset_path)
        single_img_eval_loader = DataLoader(single_img_eval_set, batch_size=256)
        evaler = MumEval(config, None)
        nna_score_lists = []
        for sub_exp in EXP_NUM_LIST:
            sub_exp_path = os.path.join(exp_path, sub_exp)
            optimal_checkpoint_num = find_optimal_checkpoint_num_by_train_config(sub_exp_path, config)
            check_point_name = f'checkpoint_{optimal_checkpoint_num}.pt'
            checkpoint_path = os.path.join(exp_path, sub_exp, check_point_name)
            evaler.reload_model(checkpoint_path)
            result_path = os.path.join(result_dir, f'{sub_exp}_{check_point_name}')
            nna_score = evaler.num_eval_two_dim(single_img_eval_loader, result_path)
            nna_score_lists.append(nna_score)
            record_num_list(os.path.join(exp_path, f'{RESULT_NAME}'), nna_score_lists, EXP_NUM_LIST)


if __name__ == "__main__":
    batch_eval()
