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


def dict_switch_key_value(d):
    return {v: k for k, v in d.items()}


'''
reference: https://www.topcoder.com/community/competitive-programming/tutorials/assignment-problem-and-hungarian-algorithm/
'''
#max weight assignment
class KMMatcher:

    ## weights : nxm weight matrix (numpy , float), n <= m
    def __init__(self, weights):
        weights = np.array(weights).astype(np.float32)
        self.weights = weights
        self.n, self.m = weights.shape
        assert self.n <= self.m
        # init label
        self.label_x = np.max(weights, axis=1)
        self.label_y = np.zeros((self.m, ), dtype=np.float32)

        self.max_match = 0
        self.xy = -np.ones((self.n,), dtype=np.int_)
        self.yx = -np.ones((self.m,), dtype=np.int_)

    def do_augment(self, x, y):
        self.max_match += 1
        while x != -2:
            self.yx[y] = x
            ty = self.xy[x]
            self.xy[x] = y
            x, y = self.prev[x], ty

    def find_augment_path(self):
        self.S = np.zeros((self.n,), np.bool_)
        self.T = np.zeros((self.m,), np.bool_)

        self.slack = np.zeros((self.m,), dtype=np.float32)
        self.slackyx = -np.ones((self.m,), dtype=np.int_)  # l[slackyx[y]] + l[y] - w[slackx[y], y] == slack[y]

        self.prev = -np.ones((self.n,), np.int_)

        queue, st = [], 0
        root = -1

        for x in range(self.n):
            if self.xy[x] == -1:
                queue.append(x);
                root = x
                self.prev[x] = -2
                self.S[x] = True
                break

        self.slack = self.label_y + self.label_x[root] - self.weights[root]
        self.slackyx[:] = root

        while True:
            while st < len(queue):
                x = queue[st]; st+= 1

                is_in_graph = np.isclose(self.weights[x], self.label_x[x] + self.label_y)
                nonzero_inds = np.nonzero(np.logical_and(is_in_graph, np.logical_not(self.T)))[0]

                for y in nonzero_inds:
                    if self.yx[y] == -1:
                        return x, y
                    self.T[y] = True
                    queue.append(self.yx[y])
                    self.add_to_tree(self.yx[y], x)

            self.update_labels()
            queue, st = [], 0
            is_in_graph = np.isclose(self.slack, 0)
            nonzero_inds = np.nonzero(np.logical_and(is_in_graph, np.logical_not(self.T)))[0]

            for y in nonzero_inds:
                x = self.slackyx[y]
                if self.yx[y] == -1:
                    return x, y
                self.T[y] = True
                if not self.S[self.yx[y]]:
                    queue.append(x)
                    self.add_to_tree(self.yx[y], x)

    def solve(self, verbose = False):
        while self.max_match < self.n:
            x, y = self.find_augment_path()
            self.do_augment(x, y)

        sum = 0.
        for x in range(self.n):
            if verbose:
                print('match {} to {}, weight {:.4f}'.format(x, self.xy[x], self.weights[x, self.xy[x]]))
            sum += self.weights[x, self.xy[x]]
        self.best = sum
        if verbose:
            print('ans: {:.4f}'.format(sum))
        return sum


    def add_to_tree(self, x, prevx):
        self.S[x] = True
        self.prev[x] = prevx

        better_slack_idx = self.label_x[x] + self.label_y - self.weights[x] < self.slack
        self.slack[better_slack_idx] = self.label_x[x] + self.label_y[better_slack_idx] - self.weights[x, better_slack_idx]
        self.slackyx[better_slack_idx] = x

    def update_labels(self):
        delta = self.slack[np.logical_not(self.T)].min()
        self.label_x[self.S] -= delta
        self.label_y[self.T] += delta
        self.slack[np.logical_not(self.T)] -= delta


def keep_max_in_matrix_rows(matrix):
    # 找到每行的最大值的索引
    max_values_idx = np.argmax(matrix, axis=1)
    # 创建一个全零矩阵
    result = np.zeros_like(matrix)
    # 将每行的最大值赋值给对应的索引
    for i, idx in enumerate(max_values_idx):
        result[i, idx] = matrix[i, idx]
    return result


def keep_max_in_matrix_colum(matrix):
    # 找到每列的最大值的索引
    max_values_idx = np.argmax(matrix, axis=0)
    # 创建一个全零矩阵
    result = np.zeros_like(matrix)
    # 将每列的最大值赋值给对应的索引
    for i, idx in enumerate(max_values_idx):
        result[idx, i] = matrix[idx, i]
    return result


if __name__ == "__main__":
    matrix = np.array([[1, 2, 3, 3], [4, 5, 5, 6], [7, 8, 9, 9]])
    new_matrix_rows = keep_max_in_matrix_colum(matrix)
    print("Matrix with only row maxes:", new_matrix_rows)

