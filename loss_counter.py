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

if __name__ == '__main__':
    path = "VQ/exp/2023.03.19_10vq_Zc[2]_Zs[0]_edim1_singleS/1/plus_eval.txt"
    read_record(path)