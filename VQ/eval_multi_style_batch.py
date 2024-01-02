import sys
import os
import numpy as np
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from importlib import reload
from eval_multi_style import AccuEval


EXP_ROOT_PATH = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/exp')
CHECK_POINT = 'checkpoint_10000.pt'
EXP_NUM_LIST = [str(i) for i in range(1, 21)]
EXP_NAME_LIST = [
    '2023.12.17_multiStyle_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_2_realPair',
    '2023.12.17_multiStyle_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_2_realPair_noAssoc',
]
DATASET_ROOT = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../dataset/')
TRAIN_SET_PATH = os.path.join(DATASET_ROOT, 'multi_style_eval_(0,20)_FixedPos_TrainStyle')
EVAL_COLOR_SET_PATH = os.path.join(DATASET_ROOT, 'multi_style_eval_(0,20)_FixedPos_newColor')
EVAL_SHAPE_SET_PATH = os.path.join(DATASET_ROOT, 'multi_style_eval_(0,20)_FixedPos_newShape')
EVAL_ALL_PATH = os.path.join(DATASET_ROOT, 'multi_style_eval_(0,20)_FixedPos_newShapeColor')
RECORD_EVAL_COLOR_PATH = 'new_color_eval.txt'
RECORD_EVAL_SHAPE_PATH = 'new_shape_eval.txt'
RECORD_EVAL_ALL_PATH = 'new_style_eval.txt'



def init_an_evaler(exp_name, train_set_path, eval_set_path):
    exp_path = os.path.join(EXP_ROOT_PATH, exp_name)
    sys.path.append(exp_path)
    os.chdir(exp_path)
    print(f'Exp path: {exp_path}')
    t_config = __import__('train_config')
    reload(t_config)
    sys.path.pop()
    evaler = AccuEval(t_config.CONFIG, train_set_path, eval_set_path)
    return evaler


def record(record_path, accu_list):
    mean_accu = sum(accu_list) / len(accu_list)
    std_accu = np.std(accu_list)
    with open(record_path, 'w') as f:
        f.write(f'Mean accu: {mean_accu}\n')
        f.write(f'Std accu: {std_accu}\n')
        for i in range(len(accu_list)):
            f.write(f'Exp {EXP_NUM_LIST[i]}: {accu_list[i]}\n')


if __name__ == "__main__":
    for exp_name in EXP_NAME_LIST:
        exp_path = os.path.join(EXP_ROOT_PATH, exp_name)
        color_evaler = init_an_evaler(exp_name, TRAIN_SET_PATH, EVAL_COLOR_SET_PATH)
        shape_evaler = init_an_evaler(exp_name, TRAIN_SET_PATH, EVAL_SHAPE_SET_PATH)
        all_evaler = init_an_evaler(exp_name, TRAIN_SET_PATH, EVAL_ALL_PATH)
        list_color_accu = []
        list_shape_accu = []
        list_all_accu = []
        for sub_exp in EXP_NUM_LIST:
            checkpoint_path = os.path.join(exp_path, sub_exp, CHECK_POINT)
            list_color_accu.append(color_evaler.eval_check_point(checkpoint_path))
            list_shape_accu.append(shape_evaler.eval_check_point(checkpoint_path))
            list_all_accu.append(all_evaler.eval_check_point(checkpoint_path))
        record(os.path.join(exp_path, RECORD_EVAL_COLOR_PATH), list_color_accu)
        record(os.path.join(exp_path, RECORD_EVAL_SHAPE_PATH), list_shape_accu)
        record(os.path.join(exp_path, RECORD_EVAL_ALL_PATH), list_all_accu)



            




