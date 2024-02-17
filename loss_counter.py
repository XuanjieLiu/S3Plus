from typing import List

import numpy as np
import os

RECORD_PATH_DEFAULT = 'path-default'


class LossCounter:
    def __init__(self, name_list, record_path=None):
        self.name_list = name_list
        self.values_list = []  # iter X values
        self.record_path = record_path

    def add_values(self, values):
        self.values_list.append(values)

    def calc_values_mean(self):
        return np.mean(self.values_list, axis=0)

    def clear_values_list(self):
        self.values_list = []

    def make_record(self, num, round_idx=6):
        means = self.calc_values_mean()
        str_list = [f'{self.name_list[i]}:{round(means[i], round_idx)}' for i in range(len(self.name_list))]
        split_str = ','
        start_str = str(num) + '-'
        final_str = start_str + split_str.join(str_list) + '\n'
        return final_str

    def record_and_clear(self, record_path=RECORD_PATH_DEFAULT, num=0, round_idx=6):
        r_path = self.choose_path(record_path)
        final_str = self.make_record(num, round_idx)
        fo = open(r_path, "a")
        fo.writelines(final_str)
        fo.close()
        self.clear_values_list()

    def choose_path(self, record_path):
        if record_path == RECORD_PATH_DEFAULT:
            assert self.record_path is not None, "Record_path should not be None"
            r_path = self.record_path
        else:
            r_path = record_path
        return r_path

    def load_iter_num(self, record_path=RECORD_PATH_DEFAULT):
        r_path = self.choose_path(record_path)
        if os.path.exists(r_path):
            f = open(r_path, "r")
            lines = f.readlines()
            t_list = [int(a.split('-')[0]) for a in lines]
            f.close()
            return t_list[-1] + 1
        else:
            return 0


class Records:
    def __init__(self, name: str = None, X: List[float] = None, Y: List[float] = None):
        self.name = name,
        self.X = X
        self.Y = Y

    def find_iter_idx(self, iter_num):
        for i in range(len(self.X)):
            if self.X[i] > iter_num:
                return i

    def find_topMin_after_iter(self, num_extreme, iter_after=0, reverse=False):
        iter_idx = self.find_iter_idx(iter_after)
        search_list = self.Y[iter_idx:]
        search_list.sort(reverse=reverse)
        return search_list[0:num_extreme]

    def filter_Y_by_X_nums(self, X_nums):
        idx_list = []
        filtered_X = []
        for i in range(len(self.X)):
            if self.X[i] in X_nums:
                idx_list.append(i)
                filtered_X.append(self.X[i])
        filtered_Y = [self.Y[i] for i in idx_list]
        return filtered_X, filtered_Y


def read_record(path: str):
    with open(path, 'r') as f:
        lines = f.readlines()
    X = [int(line.split('-')[0]) for line in lines]
    items = ['-'.join(line.split('-')[1:]).split('\n')[0].split(',') for line in lines]
    records = {}
    for i in range(len(items[0])):
        name = items[0][i].split(':')[0]
        Y = [float(item[i].split(':')[1]) for item in items]
        records[name] = Records(name, X, Y)
    return records


def find_min_nums_idxs(list: List[float]):
    min_num = min(list)
    min_idxs = [i for i in range(len(list)) if list[i] == min_num]
    return min_idxs


def find_optimal_checkpoint(record, keys, check_points_num=None):
    filtered_idx = []
    filtered_checkpoints = []
    for key in keys:
        if check_points_num is not None and len(check_points_num) != 0:
            filtered_X, filtered_Y = record[key].filter_Y_by_X_nums(check_points_num)
        else:
            filtered_X, filtered_Y = record[key].X, record[key].Y
        if len(filtered_idx) != 0:
            filtered_Y = [filtered_Y[i] for i in filtered_idx]
        min_idxs = find_min_nums_idxs(filtered_Y)
        if len(filtered_idx) == 0:
            filtered_idx = min_idxs
        else:
            filtered_idx = [filtered_idx[i] for i in min_idxs]
            filtered_checkpoints = [filtered_X[i] for i in filtered_idx]
    optimal_checkpoint_num = filtered_checkpoints[-1]
    return optimal_checkpoint_num


if __name__ == '__main__':
    check_points_num = [n*2000 for n in range(50)]
    keys = ['plus_recon', 'plus_z', 'loss_oper', 'loss_ED']
    path = "VQ/exp/2024.02.03_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_oneColSet_AssocSymmCommuAll/2/Train_record.txt"
    record = read_record(path)
    filtered_checkpoints = find_optimal_checkpoint(record, keys, check_points_num)
    print(filtered_checkpoints)