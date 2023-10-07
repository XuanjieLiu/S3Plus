import sys
import os
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from importlib import reload
from other_task_eval import OtherTask, is_need_train

if len(sys.argv) < 2:
    print("Usage: python myscript.py arg1 arg2 ...")
    sys.exit()

EXP_ROOT_PATH = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/exp')
sys.path.append(EXP_ROOT_PATH)
EXP_NAME_LIST = sys.argv[1:]
EXP_NUM_LIST = [str(i) for i in range(1, 21)]
OTHER_TASK_EXP_NUM_LIST = [str(i) for i in range(1, 21)]

for exp_num in EXP_NUM_LIST:
    for exp_name in EXP_NAME_LIST:
        exp_path = os.path.join(EXP_ROOT_PATH, exp_name)
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
        print(f'Sub-Exp path: {sub_exp_path}')
        os.chdir(sub_exp_path)
        for j in OTHER_TASK_EXP_NUM_LIST:
            sub_sub_exp_path = os.path.join(sub_exp_path, j)
            os.chdir(sub_sub_exp_path)
            for i in range(0, len(other_task_config.CONFIG)):
                if is_need_train(other_task_config.CONFIG[i]):
                    other_task = OtherTask(pretrained_config.CONFIG, other_task_config.CONFIG[i])
                    other_task.train()

