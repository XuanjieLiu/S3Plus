import sys
import os
import numpy as np
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from importlib import reload
from plot_multistyle_zc import MultiStyleZcEvaler
from common_func import load_config_from_exp_name, record_num_list, EXP_ROOT, find_optimal_checkpoint_num_by_train_config
from common_func import load_config_from_exp_name, record_num_list, DATASET_ROOT, EXP_ROOT, dict_switch_key_value
from dataloader import SingleImgDataset, load_enc_eval_data_with_style
from torch.utils.data import DataLoader


EXP_NUM_LIST = [str(i) for i in range(1, 21)]
EXP_NAME_LIST = [
    "2024.04.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyle_AssocFullsymmCommu",
    "2024.04.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyle_AssocFullsymm",
    "2024.04.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyle_Fullsymm",
    "2024.04.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyle_Nothing",
]

EVAL_SETS = [
    {
        'name': 'Train_style',
        'path': os.path.join(DATASET_ROOT, 'multi_style_eval_(0,20)_FixedPos_TrainStyle')
    }, {
        'name': 'New_color',
        'path': os.path.join(DATASET_ROOT, 'multi_style_eval_(0,20)_FixedPos_newColor')
    }, {
        'name': 'New_shape',
        'path': os.path.join(DATASET_ROOT, 'multi_style_eval_(0,20)_FixedPos_newShape')
    }, {
        'name': 'New_shape_color',
        'path': os.path.join(DATASET_ROOT, 'multi_style_eval_(0,20)_FixedPos_newShapeColor')
    },
]

RESULT_PATH = 'multi_style_zc_eval'
MATCH_RATE_PATH = 'EMB_match_rate.txt'


if __name__ == "__main__":
    for eval_set in EVAL_SETS:
        single_img_eval_set = SingleImgDataset(eval_set['path'])
        single_img_eval_loader = DataLoader(single_img_eval_set, batch_size=256)
        eval_set['loader'] = single_img_eval_loader

    for exp_name in EXP_NAME_LIST:
        exp_path = os.path.join(EXP_ROOT, exp_name)
        config = load_config_from_exp_name(exp_name)
        evaler = MultiStyleZcEvaler(config)
        match_rate_lists = [[] for _ in range(len(EVAL_SETS))]
        result_path = os.path.join(exp_path, RESULT_PATH)
        os.makedirs(result_path, exist_ok=True)
        for sub_exp in EXP_NUM_LIST:
            sub_exp_path = os.path.join(exp_path, sub_exp)
            optimal_checkpoint_num = find_optimal_checkpoint_num_by_train_config(sub_exp_path, config)
            checkpoint_path = os.path.join(exp_path, sub_exp, f'checkpoint_{optimal_checkpoint_num}.pt')
            evaler.reload_model(checkpoint_path)
            for i in range(len(EVAL_SETS)):
                save_path = os.path.join(result_path, f'{sub_exp}_{optimal_checkpoint_num}_{EVAL_SETS[i]["name"]}')
                match_rate = evaler.eval(EVAL_SETS[i]['loader'], save_path, figure_title=f'Exp: {sub_exp} {EVAL_SETS[i]["name"]}')
                match_rate_lists[i].append(match_rate)
        for i in range(len(EVAL_SETS)):
            record_num_list(os.path.join(result_path, f'{EVAL_SETS[i]["name"]}_{MATCH_RATE_PATH}'), match_rate_lists[i], EXP_NUM_LIST)


