import sys
import os
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from importlib import reload
from other_task_eval import OtherTask, is_need_train
from common_func import find_optimal_checkpoint_num_by_train_config, EXP_ROOT

if len(sys.argv) < 2:
    print("Usage: python myscript.py arg1 arg2 ...")
    sys.exit()

sys.path.append(EXP_ROOT)
EXP_NAME_LIST = sys.argv[1:]
EXP_NUM_LIST = [str(i) for i in range(1, 21)]
OTHER_TASK_EXP_NUM_LIST = [str(i) for i in range(1, 21)]


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
        optimal_check_point = find_optimal_checkpoint_num_by_train_config(sub_exp_path, pretrained_config.CONFIG)
        for i in range(0, len(other_task_config.CONFIG)):
            print(f'Sub-Exp path: {sub_exp_path}')
            os.chdir(sub_exp_path)
            config = other_task_config.CONFIG[i]
            config['pretrained_path'] = f'checkpoint_{optimal_check_point}.pt'
            other_task = OtherTask(pretrained_config.CONFIG, config)
            for j in OTHER_TASK_EXP_NUM_LIST:
                sub_sub_exp_path = os.path.join(sub_exp_path, j)
                os.makedirs(sub_sub_exp_path, exist_ok=True)
                os.chdir(sub_sub_exp_path)
                if is_need_train(other_task_config.CONFIG[i]):
                    other_task.train()

