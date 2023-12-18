import sys
import os
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from importlib import reload
from num_eval import MumEval

DATASET_PATH = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../dataset/(1,20)-FixedPos-oneStyle')
EXP_ROOT_PATH = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/exp')
RESULT_DIR_NAME = 'two_dim_z_vis'
CHECK_POINT = 'checkpoint_9000.pt'
EXP_NUM_LIST = [str(i) for i in range(1, 11)]
EXP_NAME_LIST = [
    '2023.12.17_multiStyle_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_2_realPair',
    '2023.12.17_multiStyle_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_2_realPair_noAssoc',
]


def init_an_evaler(exp_name):
    exp_path = os.path.join(EXP_ROOT_PATH, exp_name)
    sys.path.append(exp_path)
    os.chdir(exp_path)
    print(f'Exp path: {exp_path}')
    t_config = __import__('train_config')
    reload(t_config)
    sys.path.pop()
    evaler = MumEval(t_config.CONFIG, None, DATASET_PATH)
    return evaler


def batch_eval():
    for exp_name in EXP_NAME_LIST:
        exp_path = os.path.join(EXP_ROOT_PATH, exp_name)
        result_dir = os.path.join(exp_path, RESULT_DIR_NAME)
        os.makedirs(result_dir, exist_ok=True)
        evaler = init_an_evaler(exp_name)
        for sub_exp in EXP_NUM_LIST:
            checkpoint_path = os.path.join(exp_path, sub_exp, CHECK_POINT)
            evaler.reload_model(checkpoint_path)
            result_path = os.path.join(result_dir, f'{sub_exp}_{CHECK_POINT}.png')
            evaler.num_eval_two_dim(result_path)


if __name__ == "__main__":
    batch_eval()
