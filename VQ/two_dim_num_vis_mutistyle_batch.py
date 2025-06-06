import sys
import os
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from importlib import reload
from two_dim_num_vis import MumEval
from common_func import load_config_from_exp_name, EXP_ROOT, find_optimal_checkpoint_num_by_train_config, DATASET_ROOT
from torch.utils.data import DataLoader
from dataloader import SingleImgDataset


RESULT_DIR_NAME = 'two_dim_z_vis_multi_style'

EXP_NUM_LIST = [str(i) for i in range(1, 21)]
# EXP_NUM_LIST = ['6']
EXP_NAME_LIST = [
    "2025.05.19_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyle_Nothing_trainAll",
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

def batch_eval():
    for eval_set in EVAL_SETS:
        single_img_eval_set = SingleImgDataset(eval_set['path'])
        single_img_eval_loader = DataLoader(single_img_eval_set, batch_size=256)
        eval_set['loader'] = single_img_eval_loader
    for exp_name in EXP_NAME_LIST:
        exp_path = os.path.join(EXP_ROOT, exp_name)
        result_dir = os.path.join(exp_path, RESULT_DIR_NAME)
        os.makedirs(result_dir, exist_ok=True)
        config = load_config_from_exp_name(exp_name)
        evaler = MumEval(config, None)
        for sub_exp in EXP_NUM_LIST:
            sub_exp_path = os.path.join(exp_path, sub_exp)
            optimal_checkpoint_num = find_optimal_checkpoint_num_by_train_config(sub_exp_path, config)
            # optimal_checkpoint_num=9500
            check_point_name = f'checkpoint_{optimal_checkpoint_num}.pt'
            checkpoint_path = os.path.join(exp_path, sub_exp, check_point_name)
            evaler.reload_model(checkpoint_path)
            for i in range(len(EVAL_SETS)):
                result_path = os.path.join(result_dir, f'{sub_exp}_{optimal_checkpoint_num}_{EVAL_SETS[i]["name"]}')
                evaler.num_eval_two_dim(EVAL_SETS[i]['loader'], result_path)


if __name__ == "__main__":
    batch_eval()
