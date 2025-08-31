import sys
import os
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../../'))
from importlib import reload
from train_align import AlignTrain

if len(sys.argv) < 2:
    print("Usage: python myscript.py arg1 arg2 ...")
    sys.exit()

EXP_ROOT_PATH = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/exp')
sys.path.append(EXP_ROOT_PATH)
EXP_NAME_LIST = sys.argv[1:]
# EXP_NUM_LIST = [str(i) for i in range(1, 21)]
if __name__ == '__main__':
    print(f'Experiment names: {EXP_NAME_LIST}')
    for exp_name in EXP_NAME_LIST:
        exp_path = os.path.join(EXP_ROOT_PATH, exp_name)
        os.chdir(exp_path)
        sys.path.append(exp_path)
        print(f'Exp path: {exp_path}')
        t_config = __import__('config')
        reload(t_config)
        sys.path.pop()
        print(t_config.CONFIG)
        num_sub_exp = t_config.CONFIG.get('num_sub_exp', 20)
        print(f'Number of sub-experiments: {num_sub_exp}')
        exp_num_list = [str(i) for i in range(1, num_sub_exp + 1)]
        for exp_num in exp_num_list:
            os.makedirs(exp_num, exist_ok=True)
            sub_exp_path = os.path.join(exp_path, exp_num)
            print(f'Sub-Exp path: {sub_exp_path}')
            os.chdir(sub_exp_path)
            t_config.CONFIG['sub_exp_id'] = exp_num
            trainer = AlignTrain(t_config.CONFIG)
            trainer.train()
            os.chdir(exp_path)
