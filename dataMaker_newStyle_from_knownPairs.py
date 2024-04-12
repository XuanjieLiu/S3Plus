import random
import matplotlib
import matplotlib.markers
import matplotlib.pyplot as plt
from typing import List, Iterable
from dataMaker_commonFunc import *
from tqdm import tqdm
import torch
from dataMaker_fixedPosition_plusPair import DATA_ROOT, PairData, render_dataset, comp_plus, render_arabic_num_dataset, comp_minus

matplotlib.use('AGG')
import os


TRAIN_MARKERS = ['s', 'o', 'v', 'd']
NEW_MARKERS = ['^', 'X', 'p', 'D']

TRAIN_COLORS = ['purple', 'salmon',  'olive', 'blue']
NEW_COLORS = ['red', 'green', 'black', 'yellow']

# data_set_path = "multi_style_(4,4)_realPairs_plus(0,20)"
NEW_SHAPE_TRAIN_CONF = [
    f'{DATA_ROOT}/multi_style_realPairs_plus_eval_newShape/train',
    NEW_MARKERS,
    TRAIN_COLORS,
]

# data_set_path = "multi_style_(4,4)_realPairs_plus(0,20)"
NEW_SHAPE_TEST_CONF = [
    f'{DATA_ROOT}/multi_style_realPairs_plus_eval_newShape/test',
    NEW_MARKERS,
    TRAIN_COLORS,
]

# data_set_path = "multi_style_(4,4)_realPairs_plus(0,20)"
NEW_COLOR_TRAIN_CONF = [
    f'{DATA_ROOT}/multi_style_realPairs_plus_eval_newColor/train',
    TRAIN_MARKERS,
    NEW_COLORS,
]

# data_set_path = "multi_style_(4,4)_realPairs_plus(0,20)"
NEW_COLOR_TEST_CONF = [
    f'{DATA_ROOT}/multi_style_realPairs_plus_eval_newColor/test',
    TRAIN_MARKERS,
    NEW_COLORS,
]

# data_set_path = "multi_style_(4,4)_realPairs_plus(0,20)"
NEW_SHAPE_COLOR_TRAIN_CONF = [
    f'{DATA_ROOT}/multi_style_realPairs_plus_eval_newShapeColor/train',
    NEW_MARKERS,
    NEW_COLORS,
]

# data_set_path = "multi_style_(4,4)_realPairs_plus(0,20)"
NEW_SHAPE_COLOR_TEST_CONF = [
    f'{DATA_ROOT}/multi_style_realPairs_plus_eval_newShapeColor/test',
    NEW_MARKERS,
    NEW_COLORS,
]


def load_data_pairs_from_dataset(dataset_path):
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


def make_dataset_datapair_from_tuple_list(tuple_list, data_root: str, markers, colors,  compositional_func=comp_plus):
    datapairs = []
    for t in tuple_list:
        for mar in markers:
            for color in colors:
                datapairs.append(PairData(t[0], t[1], mar, color))
    render_dataset(datapairs, data_root, compositional_func)

def make_arabic_dataset_datapair_from_tuple_list(tuple_list, data_root: str, markers, colors,  compositional_func=comp_plus):
    datapairs = []
    for t in tuple_list:
        for mar in markers:
            for color in colors:
                datapairs.append(PairData(t[0], t[1], mar, color))
    render_arabic_num_dataset(datapairs, data_root, compositional_func)



if __name__ == "__main__":
    data_set_path = "single_style_pairs_minus(0,20)"
    test_1_path = os.path.join(DATA_ROOT, data_set_path, 'test')
    # test_2_path = os.path.join(DATA_ROOT, data_set_path, 'test_2')
    train_path = os.path.join(DATA_ROOT, data_set_path, 'train')
    test_1_tuples = load_data_pairs_from_dataset(test_1_path)
    # test_2_tuples = load_data_pairs_from_dataset(test_2_path)
    train_tuples = load_data_pairs_from_dataset(train_path)
    target_set_name = f'{DATA_ROOT}/single_style_pairs_minus(0,20)_arabic'
    make_arabic_dataset_datapair_from_tuple_list(
        train_tuples,
        os.path.join(target_set_name, 'train'),
        ['default'],
        ['blue'],
        comp_minus
    )
    make_arabic_dataset_datapair_from_tuple_list(
        test_1_tuples,
        os.path.join(target_set_name, 'test'),
        ['default'],
        ['blue'],
        comp_minus
    )
    # make_arabic_dataset_datapair_from_tuple_list(
    #     test_2_tuples,
    #     os.path.join(target_set_name, 'test_2'),
    #     ['default'],
    #     ['blue'],
    # )
    # make_dataset_datapair_from_tuple_list(train_tuples, *NEW_SHAPE_TRAIN_CONF)
    # make_dataset_datapair_from_tuple_list(test_tuples, *NEW_SHAPE_TEST_CONF)
    # make_dataset_datapair_from_tuple_list(train_tuples, *NEW_COLOR_TRAIN_CONF)
    # make_dataset_datapair_from_tuple_list(test_tuples, *NEW_COLOR_TEST_CONF)
    # make_dataset_datapair_from_tuple_list(train_tuples, *NEW_SHAPE_COLOR_TRAIN_CONF)
    # make_dataset_datapair_from_tuple_list(test_tuples, *NEW_SHAPE_COLOR_TEST_CONF)

