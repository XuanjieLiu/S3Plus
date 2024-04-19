import sys
import os
from typing import List

import numpy as np
from loss_counter import read_record, find_optimal_checkpoint
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from importlib import reload

DATASET_ROOT = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../dataset/')
EXP_ROOT = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/exp')
DEFAULT_CHECKPOINTS_NUM = [n*2000 for n in range(50)]
DEFAULT_KEYS = ['plus_recon', 'plus_z', 'loss_oper', 'loss_ED']
DEFAULT_RECORD_NAME = 'Train_record.txt'
SPECIFIC_CHECKPOINT_TXT_PATH = 'specific_checkpoint.txt'


def read_specific_checkpoint(sub_exp_path):
    path = os.path.join(sub_exp_path, SPECIFIC_CHECKPOINT_TXT_PATH)
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        lines = f.readlines()
    first_line = lines[0].split('\n')[0]
    return int(first_line)


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
    specific_checkpoint = read_specific_checkpoint(sub_exp_path)
    if specific_checkpoint is not None:
        return specific_checkpoint
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


def sorted_idx(nums: List[float]):
    return sorted(range(len(nums)), key=lambda k: nums[k])


def sort_X_by_Y(X: List, Y: List[float]):
    return [X[i] for i in sorted_idx(Y)]


def add_element_at_index(arr: List, element, index):
    return arr[:index] + [element] + arr[index:]

def add_prefix_0_for_int_small_than_10(nums: List[int]):
    return [f'0{num}' if num < 10 else str(num) for num in nums]


def calc_tensor_seq_limits(tensor_seq, margin=0.1):
    limits = []
    for i in range(tensor_seq.size(-1)):
        max_val = tensor_seq[..., i].max().item()
        min_val = tensor_seq[..., i].min().item()
        interval = (max_val - min_val) * margin
        limits.append((min_val - interval, max_val + interval))
    return limits


if __name__ == "__main__":
    a = [1,0,1.5,3,2.5,2]
    print(sorted_idx(a))
    print(sort_X_by_Y(a, a))
    print(sorted(a))

