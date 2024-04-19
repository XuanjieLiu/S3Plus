import sys
import os
import numpy as np
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from importlib import reload
from eval_multi_style import AccuEval
from common_func import EXP_ROOT, DATASET_ROOT, load_config_from_exp_name, record_num_list



EXP_ROOT_PATH = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/exp')
CHECK_POINT = 'checkpoint_10000.pt'
EXP_NUM_LIST = [str(i) for i in range(1, 21)]
EXP_NAME_LIST = [
    "2024.04.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyle_AssocFullsymm",
    "2024.04.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyle_AssocFullsymmCommu",
    "2024.04.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyle_Fullsymm",
    "2024.04.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyle_Nothing",
]

TRAIN_SET_PATH = os.path.join(DATASET_ROOT, 'multi_style_eval_(0,20)_FixedPos_TrainStyle')
EVAL_COLOR_SET_PATH = os.path.join(DATASET_ROOT, 'multi_style_eval_(0,20)_FixedPos_newColor')
EVAL_SHAPE_SET_PATH = os.path.join(DATASET_ROOT, 'multi_style_eval_(0,20)_FixedPos_newShape')
EVAL_ALL_PATH = os.path.join(DATASET_ROOT, 'multi_style_eval_(0,20)_FixedPos_newShapeColor')
RECORD_EVAL_COLOR_PATH = 'new_color_eval.txt'
RECORD_EVAL_SHAPE_PATH = 'new_shape_eval.txt'
RECORD_EVAL_ALL_PATH = 'new_style_eval.txt'


if __name__ == "__main__":
    for exp_name in EXP_NAME_LIST:
        exp_path = os.path.join(EXP_ROOT, exp_name)
        config = load_config_from_exp_name(exp_name)
        color_evaler = AccuEval(config, TRAIN_SET_PATH, EVAL_COLOR_SET_PATH)
        shape_evaler = AccuEval(config, TRAIN_SET_PATH, EVAL_SHAPE_SET_PATH)
        all_evaler = AccuEval(config, TRAIN_SET_PATH, EVAL_ALL_PATH)
        list_color_accu = []
        list_shape_accu = []
        list_all_accu = []
        for sub_exp in EXP_NUM_LIST:
            checkpoint_path = os.path.join(exp_path, sub_exp, CHECK_POINT)
            list_color_accu.append(color_evaler.eval_check_point(checkpoint_path))
            list_shape_accu.append(shape_evaler.eval_check_point(checkpoint_path))
            list_all_accu.append(all_evaler.eval_check_point(checkpoint_path))
        record_num_list(os.path.join(exp_path, RECORD_EVAL_COLOR_PATH), list_color_accu, EXP_NUM_LIST)
        record_num_list(os.path.join(exp_path, RECORD_EVAL_SHAPE_PATH), list_shape_accu, EXP_NUM_LIST)
        record_num_list(os.path.join(exp_path, RECORD_EVAL_ALL_PATH), list_all_accu, EXP_NUM_LIST)



            




