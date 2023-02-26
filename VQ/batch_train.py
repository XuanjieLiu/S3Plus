from importlib import reload
import os
from train import is_need_train, PlusTrainer
import sys

EXP_ROOT_PATH = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/exp')
sys.path.append(EXP_ROOT_PATH)
EXP_NAME_LIST = [
  "2023.02.21_20vq_Zc[1]_Zs(2)_edim64",
]
EXP_NUM_LIST = ['1', '2', '3', '4', '5']

for exp_num in EXP_NUM_LIST:
  for exp_name in EXP_NAME_LIST:
    exp_path = os.path.join(EXP_ROOT_PATH, exp_name)
    os.chdir(exp_path)
    sys.path.append(exp_path)
    print(f'Exp path: {exp_path}')
    t_config = __import__('train_config')
    reload(t_config)
    print(t_config.CONFIG)
    os.makedirs(exp_num, exist_ok=True)
    sub_exp_path = os.path.join(exp_path, exp_num)
    print(f'Sub-Exp path: {sub_exp_path}')
    os.chdir(sub_exp_path)
    if is_need_train(t_config.CONFIG):
      trainer = PlusTrainer(t_config.CONFIG)
      trainer.train()