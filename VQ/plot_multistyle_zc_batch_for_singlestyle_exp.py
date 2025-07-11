import sys
import os
import numpy as np
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from importlib import reload
from plot_multistyle_zc import MultiStyleZcEvaler
from common_func import load_config_from_exp_name, record_num_list, EXP_ROOT, find_optimal_checkpoint_num_by_train_config
from dataloader import SingleImgDataset
from torch.utils.data import DataLoader


EXP_NUM_LIST = [str(i) for i in range(1, 21)]
EXP_NAME_LIST = [
    "2025.06.12_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_Nothing_train0.1",
    "2025.06.12_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_Nothing_train0.2",
    "2025.06.12_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_Nothing_train0.3",
    "2025.06.12_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_Nothing_train0.4",
    "2025.06.12_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_Nothing_train0.5",
    "2025.06.12_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_Nothing_train0.6",
    "2025.06.12_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_Nothing_train0.7",
    "2025.06.12_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_Nothing_train0.8",
    "2025.06.12_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_Nothing_train0.9",
    "2025.06.12_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_symm_train0.1",
    "2025.06.12_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_symm_train0.2",
    "2025.06.12_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_symm_train0.3",
    "2025.06.12_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_symm_train0.4",
]


RESULT_PATH = 'EMB_match_rate_vis'
MATCH_RATE_PATH = 'EMB_match_rate.txt'


if __name__ == "__main__":
    for exp_name in EXP_NAME_LIST:
        exp_path = os.path.join(EXP_ROOT, exp_name)
        config = load_config_from_exp_name(exp_name)
        evaler = MultiStyleZcEvaler(config)
        match_rate_lists = []
        result_path = os.path.join(exp_path, RESULT_PATH)
        os.makedirs(result_path, exist_ok=True)
        for sub_exp in EXP_NUM_LIST:
            sub_exp_path = os.path.join(exp_path, sub_exp)
            optimal_checkpoint_num = find_optimal_checkpoint_num_by_train_config(sub_exp_path, config)
            # optimal_checkpoint_num = 10000
            checkpoint_path = os.path.join(exp_path, sub_exp, f'checkpoint_{optimal_checkpoint_num}.pt')
            evaler.reload_model(checkpoint_path)
            single_img_eval_set = SingleImgDataset(config['single_img_eval_set_path'])
            single_img_eval_loader = DataLoader(single_img_eval_set, batch_size=256)
            save_path = os.path.join(result_path, f'{sub_exp}_{optimal_checkpoint_num}')
            match_rate = evaler.eval(single_img_eval_loader, save_path, figure_title=f'Exp: {sub_exp}', is_shift=False, is_plot_graph=False)
            match_rate_lists.append(match_rate)
            record_num_list(os.path.join(exp_path, f'{MATCH_RATE_PATH}'), match_rate_lists, EXP_NUM_LIST)


