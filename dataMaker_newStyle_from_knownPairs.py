import random
import matplotlib
import matplotlib.markers
import matplotlib.pyplot as plt
from typing import List, Iterable
from dataMaker_commonFunc import *
from tqdm import tqdm
import torch
from dataMaker_fixedPosition_plusPair import DATA_ROOT, PairData, render_dataset, comp_plus, render_arabic_num_dataset, \
    comp_minus, render_lines_num_dataset, render_EU_tally_mark_num_dataset, render_ZHENG_num_dataset

matplotlib.use('AGG')
import os


TRAIN_MARKERS = ['s', 'o', 'v', 'd']
NEW_MARKERS = ['^', 'X', 'p', 'D']

TRAIN_COLORS = ['purple', 'salmon',  'olive', 'blue']
NEW_COLORS = ['red', 'green', 'black', 'yellow']



# data_set_path = "multi_style_(4,4)_realPairs_plus(0,20)"
NEW_SHAPE_TEST_CONF = [
    f'{DATA_ROOT}/multi_style_realPairs_plus_eval_newShape/shot_16_test',
    NEW_MARKERS,
    TRAIN_COLORS,
]



# data_set_path = "multi_style_(4,4)_realPairs_plus(0,20)"
NEW_COLOR_TEST_CONF = [
    f'{DATA_ROOT}/multi_style_realPairs_plus_eval_newColor/shot_16_test',
    TRAIN_MARKERS,
    NEW_COLORS,
]



# data_set_path = "multi_style_(4,4)_realPairs_plus(0,20)"
NEW_SHAPE_COLOR_TEST_CONF = [
    f'{DATA_ROOT}/multi_style_realPairs_plus_eval_newShapeColor/shot_16_test',
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


def make_dataset_datapair_from_tuple_list(
        tuple_list, data_root: str,
        markers, colors,
        compositional_func=comp_plus,
        data_render_func: callable = render_dataset,
):
    datapairs = []
    for t in tuple_list:
        for mar in markers:
            for color in colors:
                datapairs.append(PairData(t[0], t[1], mar, color))
    data_render_func(datapairs, data_root, compositional_func)


MANUAL_DATAPAIRS_TUPLE = [
    (0, 3), (3, 1), (1, 4), (6, 0), (2, 5), (8, 1), (4, 6), (2, 9), (5, 7), (3, 10),
    (2, 12), (11, 4), (12, 8), (15, 3), (2, 14), (17, 2)
]


def render_new_data_from_old_pairs(
        source_data_set_path: str,
        target_data_set_path: str,
        markers: List[str],
        colors: List[str],
        compositional_func=comp_plus,
        data_render_func: callable = render_dataset,
):
    data_tuple = load_data_pairs_from_dataset(source_data_set_path)
    make_dataset_datapair_from_tuple_list(
        data_tuple,
        target_data_set_path,
        markers,
        colors,
        compositional_func,
        data_render_func
    )


def render_new_dataset_from_old_dataset(
        seb_set_list: List[str],
        source_data_set_path: str,
        target_data_set_path: str,
        markers: List[str],
        colors: List[str],
        compositional_func=comp_plus,
        data_render_func: callable = render_dataset,
):
    for name in seb_set_list:
        render_new_data_from_old_pairs(
            os.path.join(source_data_set_path, name),
            os.path.join(target_data_set_path, name),
            markers,
            colors,
            compositional_func,
            data_render_func
        )

if __name__ == "__main__":

    render_new_dataset_from_old_dataset(
        ['train', 'test_1', 'test_2'],
        os.path.join(DATA_ROOT, 'single_style_pairs(0,20)_tripleSet'),
        os.path.join(DATA_ROOT, 'multi_style_pairs(0,20)_tripleSet_mahjong'),
        ['default'],
        ['b'],
        comp_plus,  # remember to change the comp_plus to comp_minus or reverse
        render_EU_tally_mark_num_dataset
    )

    """
    Make EU tally mark dataset from single style pairs dataset.
    """
    # render_new_dataset_from_old_dataset(
    #     ['train', 'test_1', 'test_2'],
    #     os.path.join(DATA_ROOT, 'single_style_pairs(0,20)_tripleSet_ZHENG'),
    #     os.path.join(DATA_ROOT, 'single_style_pairs(0,20)_tripleSet_EU_tally'),
    #     ['default'],
    #     ['b'],
    #     comp_plus,  # remember to change the comp_plus to comp_minus or reverse
    #     render_EU_tally_mark_num_dataset
    # )

    # source_dataset = os.path.join(DATA_ROOT, 'single_style_pairs(0,20)_tripleSet')
    # train_plus_tuple = load_data_pairs_from_dataset(os.path.join(source_dataset, 'train'))
    # test_1_plus_tuple = load_data_pairs_from_dataset(os.path.join(source_dataset, 'test_1'))
    # test_2_plus_tuple = load_data_pairs_from_dataset(os.path.join(source_dataset, 'test_2'))
    # target_data_set_path = "single_style_pairs(0,20)_tripleSet_ZHENG"
    # test_1_path = os.path.join(DATA_ROOT, target_data_set_path, 'test_1')
    # test_2_path = os.path.join(DATA_ROOT, target_data_set_path, 'test_2')
    # train_path = os.path.join(DATA_ROOT, target_data_set_path, 'train')
    # make_dataset_datapair_from_tuple_list(
    #     train_plus_tuple, train_path, ['default'], ['b'], comp_plus, render_ZHENG_num_dataset
    # )
    # make_dataset_datapair_from_tuple_list(
    #     test_1_plus_tuple, test_1_path, ['default'], ['b'], comp_plus, render_ZHENG_num_dataset
    # )
    # make_dataset_datapair_from_tuple_list(
    #     test_2_plus_tuple, test_2_path, ['default'], ['b'], comp_plus, render_ZHENG_num_dataset
    # )

    print('aaa')


    # target_set_name = f'{DATA_ROOT}/single_style_pairs_minus(0,20)_arabic'
    # make_arabic_dataset_datapair_from_tuple_list(
    #     train_tuples,
    #     os.path.join(target_set_name, 'train'),
    #     ['default'],
    #     ['blue'],
    #     comp_minus
    # )
    # make_arabic_dataset_datapair_from_tuple_list(
    #     test_1_tuples,
    #     os.path.join(target_set_name, 'test'),
    #     ['default'],
    #     ['blue'],
    #     comp_minus
    # )


