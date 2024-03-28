import sys
import os
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from importlib import reload
from other_task_eval import OtherTask, is_need_train
from common_func import find_optimal_checkpoint_num_by_train_config, EXP_ROOT

from other_task_eval import OtherTask


EXP_NAME = '2024.02.03_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_AssocSymmCommuAll'
SUB_EXP = 2
PRETRAIN_CHECK_POINT = 'checkpoint_50000.pt'
SUB_SUB_EXP = 1
OTHER_TASK_EXP_NUM = 0


if __name__ == "__main__":
    exp_path = os.path.join(EXP_ROOT, EXP_NAME)
    os.chdir(exp_path)
    sys.path.append(exp_path)
    print(f'Exp path: {exp_path}')
    pretrained_config = __import__('train_config')
    other_task_config = __import__('other_tasks_config')
    reload(pretrained_config)
    reload(other_task_config)
    sys.path.pop()
    sub_exp_path = os.path.join(exp_path, str(SUB_EXP))
    os.chdir(sub_exp_path)
    config = other_task_config.CONFIG[OTHER_TASK_EXP_NUM]
    config['pretrained_path'] = PRETRAIN_CHECK_POINT
    other_task = OtherTask(pretrained_config.CONFIG, config)
    sub_sub_exp_path = os.path.join(sub_exp_path, str(SUB_SUB_EXP))
    os.chdir(sub_sub_exp_path)

    data_path = config['train_data_path']
    result_name = f'plus_train_{OTHER_TASK_EXP_NUM}_recon_result'
    other_task.eval_dec_view(data_path, result_name)
