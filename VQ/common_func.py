import sys
import os
import numpy as np
from loss_counter import read_record, find_optimal_checkpoint
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from importlib import reload

DATASET_ROOT = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../dataset/')
EXP_ROOT = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/exp')
DEFAULT_CHECKPOINTS_NUM = [n*2000 for n in range(50)]
DEFAULT_KEYS = ['plus_recon', 'plus_z', 'loss_oper', 'loss_ED']
DEFAULT_RECORD_NAME = 'Train_record.txt'


def find_optimal_checkpoint_num(
    sub_exp_path,
    record_name=DEFAULT_RECORD_NAME,
    keys=None,
    check_points_num=None
):
    if check_points_num is None:
        check_points_num = DEFAULT_CHECKPOINTS_NUM
    if keys is None:
        keys = DEFAULT_KEYS
    path = os.path.join(sub_exp_path, record_name)
    record = read_record(path)
    optimal_checkpoint_num = find_optimal_checkpoint(record, keys, check_points_num)
    return optimal_checkpoint_num


def find_optimal_checkpoint_num_by_train_config(
    sub_exp_path,
    train_config,
    keys=None,
):
    record_name = train_config['train_record_path']
    checkpoint_interval = train_config['checkpoint_interval']
    max_iter_num = train_config['max_iter_num']
    check_points_num = int(max_iter_num / checkpoint_interval)
    check_points = [checkpoint_interval * i for i in range(check_points_num)]
    return find_optimal_checkpoint_num(sub_exp_path, record_name, keys, check_points)


def load_config_from_exp_name(exp_name, exp_root=EXP_ROOT, config_name='train_config'):
    exp_path = os.path.join(exp_root, exp_name)
    sys.path.append(exp_path)
    os.chdir(exp_path)
    print(f'Exp path: {exp_path}')
    t_config = __import__(config_name)
    reload(t_config)
    sys.path.pop()
    return t_config.CONFIG


def record_num_list(record_path, accu_list, exp_num_list=None):
    if exp_num_list is None:
        exp_num_list = [str(i) for i in range(1, 21)]
    mean_accu = sum(accu_list) / len(accu_list)
    std_accu = np.std(accu_list)
    with open(record_path, 'w') as f:
        f.write(f'Mean accu: {mean_accu}\n')
        f.write(f'Std accu: {std_accu}\n')
        for i in range(len(accu_list)):
            f.write(f'Exp {exp_num_list[i]}: {accu_list[i]}\n')

def parse_label(label):
    return int(label.split('.')[0].split('-')[1])

