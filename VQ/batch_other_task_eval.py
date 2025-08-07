import sys
import os
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from importlib import reload
from other_task_eval import OtherTask, is_need_train
from common_func import find_optimal_checkpoint_num_by_train_config, EXP_ROOT
import torch.multiprocessing as mp

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python myscript.py arg1 arg2 ...")
        sys.exit()

    mp.set_start_method('spawn', force=True)
    sys.path.append(EXP_ROOT)
    EXP_NAME_LIST = sys.argv[1:]
    EXP_NUM_LIST = [str(i) for i in range(1, 21)]

    # Batch eval minus

    for exp_num in EXP_NUM_LIST:
        for exp_name in EXP_NAME_LIST:
            exp_path = os.path.join(EXP_ROOT, exp_name)
            os.chdir(exp_path)
            sys.path.append(exp_path)
            print(f'Exp path: {exp_path}')
            pretrained_config = __import__('train_config')
            other_task_config = __import__('other_tasks_config')
            reload(pretrained_config)
            reload(other_task_config)
            sys.path.pop()
            print(pretrained_config.CONFIG)
            print(other_task_config.CONFIG)
            sub_exp_path = os.path.join(exp_path, exp_num)
            for i in range(0, len(other_task_config.CONFIG)):
                print(f'Sub-Exp path: {sub_exp_path}')
                os.chdir(sub_exp_path)
                config = other_task_config.CONFIG[i]
                optimal_checkpoint_finding_config = config.get('optimal_checkpoint_finding_config', None)
                optimal_check_point = find_optimal_checkpoint_num_by_train_config(sub_exp_path, pretrained_config.CONFIG, optimal_checkpoint_finding_config)
                config['pretrained_path'] = f'checkpoint_{optimal_check_point}.pt'
                print(f'Optimal checkpoint: {optimal_check_point}')
                other_task = OtherTask(pretrained_config.CONFIG, config)
                num_sub_exp = config.get('num_sub_exp', 20)
                other_task_exp_num_list = [str(i) for i in range(1, num_sub_exp + 1)]
                for j in other_task_exp_num_list:
                    sub_sub_exp_path = os.path.join(sub_exp_path, j)
                    os.makedirs(sub_sub_exp_path, exist_ok=True)
                    os.chdir(sub_sub_exp_path)
                    if is_need_train(other_task_config.CONFIG[i]):
                        other_task.train()

