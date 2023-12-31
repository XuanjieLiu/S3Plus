import random
import matplotlib
import matplotlib.markers
import matplotlib.pyplot as plt
from typing import List, Iterable
from dataMaker_commonFunc import *
from tqdm import tqdm
import torch
from dataMaker_fixedPosition_plusPair import DATA_ROOT, PairData, render_dataset, comp_plus

matplotlib.use('AGG')
import os


def load_data_pairs_from_dataset(dataset_name):
    dataset_path = os.path.join(DATA_ROOT, dataset_name)
    files_name = os.listdir(dataset_path)
    tuples = [(int(name.split('-')[0]), int(name.split('-')[1])) for name in files_name]
    no_duplicate_tuples = remove_duplicate_binary_tuple(tuples)
    print(len(no_duplicate_tuples))
    return no_duplicate_tuples


def remove_duplicate_binary_tuple(tuple_list):
    new_list = []
    for t in tuple_list:
        if t not in new_list:
            new_list.append(t)
    return new_list


def make_dataset_datapair_from_tuple_list(tuple_list, markers, colors, data_root: str, compositional_func):
    datapairs = []
    for t in tuple_list:
        for mar in markers:
            for color in colors:
                datapairs.append(PairData(t[0], t[1], mar, color))
    render_dataset(datapairs, data_root, compositional_func)




if __name__ == "__main__":
    a = (2, 1)
    b = (1, 2)
    c = (2, 1)
    print(a==b)
    print(a==c)
    dataset_path = "multi_style_(4,4)_realPairs_plus(0,20)/test"
    print(load_data_pairs_from_dataset(dataset_path))

