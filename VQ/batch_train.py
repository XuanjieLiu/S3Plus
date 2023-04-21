import sys
import os
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from importlib import reload
from train import is_need_train, PlusTrainer


EXP_ROOT_PATH = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/exp')
sys.path.append(EXP_ROOT_PATH)
EXP_NAME_LIST = [
    "2023.04.20_10vq_Zc[2]_Zs[0]_edim8_plusUnit128.1_encFc128.1_singleS_plusOnE",
]
EXP_NUM_LIST = [str(i) for i in range(1, 21)]

for exp_num in EXP_NUM_LIST:
    for exp_name in EXP_NAME_LIST:
        exp_path = os.path.join(EXP_ROOT_PATH, exp_name)
        os.chdir(exp_path)
        sys.path.append(exp_path)
        print(f'Exp path: {exp_path}')
        t_config = __import__('train_config')
        reload(t_config)
        sys.path.pop()
        print(t_config.CONFIG)
        os.makedirs(exp_num, exist_ok=True)
        sub_exp_path = os.path.join(exp_path, exp_num)
        print(f'Sub-Exp path: {sub_exp_path}')
        os.chdir(sub_exp_path)
        if is_need_train(t_config.CONFIG):
            trainer = PlusTrainer(t_config.CONFIG)
            trainer.train()
