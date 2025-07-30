import random
import matplotlib
import matplotlib.markers
import matplotlib.pyplot as plt
from typing import List, Iterable
from dataMaker_commonFunc import *
from tqdm import tqdm
import torch

matplotlib.use('AGG')
from scipy.spatial import distance
import os


NUMBERS = range(0, 21)
MARKERS = ['o', 'v', '*', 'd']
DATA_ROOT = 'dataset'
DATA_PATH = f'{DATA_ROOT}/PlusPair-({NUMBERS[0]},{NUMBERS[-1]})-FixedPos'
COLORS_TRAIN = ['purple', 'salmon', 'olive', 'blue']

NUM_RAN = (1, 20)
SINGLE_STYLE_DATA_ROOT_PLUS = f'dataset/single_style_pairs({NUM_RAN[0]},{NUM_RAN[1]})'
SINGLE_STYLE_DATA_ROOT_MOD = f'dataset/single_style_pairs_mod({NUM_RAN[0]},{NUM_RAN[1]})'
SINGLE_STYLE_DATA_ROOT_DIVISION = f'dataset/single_style_pairs_division({NUM_RAN[0]},{NUM_RAN[1]})'
SINGLE_MARKERS = ['o']
SINGLE_COLOR = ['blue']


def draw_data(i, j, mar, color, data_path, compositional_func):
    os.makedirs(data_path, exist_ok=True)
    plot_a_scatter(
        DOT_POSITIONS[i],
        save_dir=os.path.join(data_path, f'a-{i}'),
        marker=mar, color=color, is_fill=i != 0
    )
    plot_a_scatter(
        DOT_POSITIONS[j],
        save_dir=os.path.join(data_path, f'b-{j}'),
        marker=mar, color=color, is_fill=j != 0
    )
    result = compositional_func(i, j)
    print(f'i={i}, j={j}, k={result}')
    plot_a_scatter(
        DOT_POSITIONS[result],
        save_dir=os.path.join(data_path, f'c-{result}'),
        marker=mar, color=color, is_fill=result != 0
    )


def draw_arabic_data(i, j, color, data_path, compositional_func):
    os.makedirs(data_path, exist_ok=True)
    plot_arabic_numbers(
        i,
        save_dir=os.path.join(data_path, f'a-{i}'),
        color=color
    )
    plot_arabic_numbers(
        j,
        save_dir=os.path.join(data_path, f'b-{j}'),
        color=color
    )
    result = compositional_func(i, j)
    print(f'i={i}, j={j}, k={result}')
    plot_arabic_numbers(
        result,
        save_dir=os.path.join(data_path, f'c-{result}'),
        color=color
    )

def draw_lines_data(i, j, color, data_path, compositional_func, line_positions, line_width=2.):
    os.makedirs(data_path, exist_ok=True)
    plot_lines(
        line_positions[i],
        save_dir=os.path.join(data_path, f'a-{i}'),
        color=color,
        line_width=line_width
    )
    plot_lines(
        line_positions[j],
        save_dir=os.path.join(data_path, f'b-{j}'),
        color=color,
        line_width = line_width
    )
    result = compositional_func(i, j)
    print(f'i={i}, j={j}, k={result}')
    plot_lines(
        line_positions[result],
        save_dir=os.path.join(data_path, f'c-{result}'),
        color=color,
        line_width=line_width
    )


def data_name_2_labels(data: Iterable):
    return torch.LongTensor([data_name_2_num(x) for x in data])


def data_name_2_num(name: str):
    return int(name.split('.')[0].split('-')[1])


def data_name_2_one_hot(data: Iterable, num_class=-1):
    labels = data_name_2_labels(data)
    return torch.nn.functional.one_hot(labels, num_class)


def comp_plus(a, b):
    return a + b


def comp_minus(a, b):
    return a - b


def comp_mod(a, b):
    return a % b


def comp_division(a, b):
    return int(a / b)


def gen_pairs_list(min_number, max_number, pair_func):
    all_idx = []
    for i in range(min_number, max_number + 1):
        pair_list = []
        for a, b in pair_func(i):
            pair_list.append((a, b))
        if len(pair_list) != 0:
            all_idx.append(pair_list)
    return all_idx


def pair_list2set(pair_list, markers, colors):
    pair_set = []
    for mar in markers:
        for color in colors:
            for pair in pair_list:
                pair_set.append(PairData(pair[0], pair[1], mar, color))
    return pair_set



class PairData:
    def __init__(self, a: int, b: int, marker: str = None, color: str = None, path: str = None):
        self.a = a
        self.b = b
        self.marker = marker
        self.color = color
        self.path = path


def make_train_test_datapair_maxN(min_number, max_number, min_sample_num, sample_rate, markers, colors, pair_func):
    train_set = []
    test_set = []
    for mar in markers:
        for color in colors:
            for i in range(min_number, max_number+1):
                data = []
                for a, b in pair_func(i):
                    data.append(PairData(a, b, mar, color))
                if len(data) == 0:
                    continue
                all_idx = range(0, len(data))
                sample_num = min(len(data), max(min_sample_num, int(round(i * sample_rate, 0))))
                train_idx = random.sample(all_idx, sample_num)
                for x in all_idx:
                    if x in train_idx:
                        train_set.append(data[x])
                    else:
                        test_set.append(data[x])
    return train_set, test_set


def make_train_test_datapair_maxN_no_leak(min_number, max_number, min_sample_num, sample_rate, markers, colors, pair_func):
    train_set = []
    test_set = []
    for i in range(min_number, max_number+1):
        all_idx = []
        base = 0
        for a, b in pair_func(i):
            all_idx.append(base)
            base += 1
        sample_num = min(i+1, max(min_sample_num, int(round(i * sample_rate, 0))))
        train_idx = random.sample(all_idx, sample_num)
        for mar in markers:
            for color in colors:
                data = []
                for a, b in pair_func(i):
                    data.append(PairData(a, b, mar, color))
                for x in all_idx:
                    if x in train_idx:
                        train_set.append(data[x])
                    else:
                        test_set.append(data[x])
    return train_set, test_set


def make_plusone_triple_datapair_maxN_no_leak(min_number, max_number, markers, colors, pair_func):
    train_set = []
    test_set_1 = []
    test_set_2 = []
    for i in range(min_number, max_number+1):
        for mar in markers:
            for color in colors:
                for a, b in pair_func(i):
                    if a == 1:
                        train_set.append(PairData(a, b, mar, color))
                    elif b == 1:
                        test_set_2.append(PairData(a, b, mar, color))
                    else:
                        test_set_1.append(PairData(a, b, mar, color))
    return train_set, test_set_1, test_set_2


def make_random_one_shot_triple_datapair_no_leak(min_number, max_number, markers, colors, pair_func):
    assert min_number >= 1, 'min_number should be greater than 1'
    train_pairs = []
    test_pairs_1 = []
    test_pairs_2 = []
    all_idx = gen_pairs_list(min_number, max_number, pair_func)
    for pair_list in all_idx:
        train_idx = random.randint(0, len(pair_list) - 1)
        sample = pair_list[train_idx]
        a, b = sample
        if a == b:
            test_2_idx = -1
        else:
            test_2_idx = len(pair_list) - 1 - train_idx
        for i in range(0, len(pair_list)):
            if i == train_idx:
                train_pairs.append(pair_list[i])
            elif i == test_2_idx:
                test_pairs_2.append(pair_list[i])
            else:
                test_pairs_1.append(pair_list[i])
    train_set = pair_list2set(train_pairs, markers, colors)
    test_set_1 = pair_list2set(test_pairs_1, markers, colors)
    test_set_2 = pair_list2set(test_pairs_2, markers, colors)
    return train_set, test_set_1, test_set_2


def make_plusone_double_datapair_maxN_no_leak(min_number, max_number, markers, colors, pair_func):
    train_set = []
    test_set = []
    for i in range(min_number, max_number+1):
        for mar in markers:
            for color in colors:
                for a, b in pair_func(i):
                    if a == 1 or b == 1:
                        train_set.append(PairData(a, b, mar, color))
                    else:
                        test_set.append(PairData(a, b, mar, color))
    return train_set, test_set


def make_plus_half_double_datapair_maxN_no_leak(min_number, max_number, markers, colors, pair_func):
    train_set = []
    test_set = []
    for i in range(min_number, max_number+1):
        for mar in markers:
            for color in colors:
                for a, b in pair_func(i):
                    half = int(i / 2)
                    if a == half or b == half:
                        train_set.append(PairData(a, b, mar, color))
                    else:
                        test_set.append(PairData(a, b, mar, color))
    return train_set, test_set


def make_train_test_datapair_division(min_number, max_number, sample_rate, markers, colors, pair_func):
    train_set = []
    test_set = []
    for mar in markers:
        for color in colors:
            for i in range(min_number, max_number+1):
                data = []
                for a, b in pair_func(i):
                    data.append(PairData(a, b, mar, color))
                all_idx = range(0, len(data))
                sample_num = int(round(len(all_idx) * sample_rate, 0))
                train_idx = random.sample(all_idx, sample_num)
                for x in all_idx:
                    if x in train_idx:
                        train_set.append(data[x])
                    else:
                        test_set.append(data[x])
    return train_set, test_set


def render_dataset(data_list: List[PairData], data_root: str, compositional_func):
    for data in tqdm(data_list, desc=data_root):
        a = data.a
        b = data.b
        color = data.color
        marker = data.marker
        data_name = f'{a}-{b}-{MARK_NAME_SPACE[marker]}-{color}'
        data_path = os.path.join(data_root, data_name)
        os.makedirs(data_path, exist_ok=True)
        draw_data(a, b, marker, color, data_path, compositional_func)


def render_arabic_num_dataset(data_list: List[PairData], data_root: str, compositional_func):
    for data in tqdm(data_list, desc=data_root):
        a = data.a
        b = data.b
        color = data.color
        data_name = f'{a}-{b}-default-{color}'
        data_path = os.path.join(data_root, data_name)
        os.makedirs(data_path, exist_ok=True)
        draw_arabic_data(a, b, color, data_path, compositional_func)


def render_lines_num_dataset(data_list: List[PairData], data_root: str, compositional_func,
                             line_positions, line_width=2., marker_name='default'):
    for data in tqdm(data_list, desc=data_root):
        a = data.a
        b = data.b
        color = data.color
        data_name = f'{a}-{b}-{marker_name}-{color}'
        data_path = os.path.join(data_root, data_name)
        os.makedirs(data_path, exist_ok=True)
        draw_lines_data(a, b, color, data_path, compositional_func, line_positions, line_width)

def render_ZHENG_num_dataset(data_list: List[PairData], data_root: str, compositional_func):
    return render_lines_num_dataset(data_list, data_root, compositional_func, ZHENG_POSITIONS, marker_name='zheng')


def render_EU_tally_mark_num_dataset(data_list: List[PairData], data_root: str, compositional_func):
    return render_lines_num_dataset(data_list, data_root, compositional_func, EU_tally_mark_POSITIONS,
                                    line_width=1.5, marker_name='eutally')


def sum_pairs(min_number):
    def func(max_number):
        for a in range(min_number, max_number + 1):
            b = max_number - a
            if b < min_number:
                continue
            yield a, b
    return func


def minus_pairs(min_number):
    def func(max_number):
        for a in range(min_number, max_number + 1):
            if max_number - a < min_number:
                continue
            yield max_number, a
    return func


def mod_pairs(a):
    min_div_num = 1
    max_div_num = NUM_RAN[1] + 1
    for i in range(min_div_num, max_div_num):
        yield a, i


def make_dataset_single_style_plus():
    data_root = SINGLE_STYLE_DATA_ROOT_PLUS
    os.makedirs(data_root, exist_ok=True)
    train_root = os.path.join(data_root, 'train')
    test_root = os.path.join(data_root, 'test')
    train_set, test_set = make_train_test_datapair_maxN(
        NUM_RAN[0],
        NUM_RAN[1],
        3,
        0.33,
        SINGLE_MARKERS,
        SINGLE_COLOR,
        sum_pairs(NUM_RAN[0]),
    )
    render_dataset(train_set, train_root, comp_plus)
    render_dataset(test_set, test_root, comp_plus)


def make_dataset_single_style_minus():
    min_num = 1
    max_num = 20
    data_root = f'dataset/single_style_pairs_minus({min_num},{max_num})'
    os.makedirs(data_root, exist_ok=True)
    train_root = os.path.join(data_root, 'train')
    test_root = os.path.join(data_root, 'test')
    train_set, test_set = make_train_test_datapair_maxN(
        min_num,
        max_num,
        3,
        0.6,
        SINGLE_MARKERS,
        SINGLE_COLOR,
        minus_pairs(min_num),
    )
    render_dataset(train_set, train_root, comp_minus)
    render_dataset(test_set, test_root, comp_minus)


def make_dataset_single_style_mod():
    data_root = SINGLE_STYLE_DATA_ROOT_MOD
    os.makedirs(data_root, exist_ok=True)
    train_root = os.path.join(data_root, 'train')
    test_root = os.path.join(data_root, 'test')
    train_set, test_set = make_train_test_datapair_division(
        NUM_RAN[0],
        NUM_RAN[1],
        0.7,
        SINGLE_MARKERS,
        SINGLE_COLOR,
        mod_pairs,
    )
    render_dataset(train_set, train_root, comp_mod)
    render_dataset(test_set, test_root, comp_mod)


def make_dataset_single_style_division():
    data_root = SINGLE_STYLE_DATA_ROOT_DIVISION
    os.makedirs(data_root, exist_ok=True)
    train_root = os.path.join(data_root, 'train')
    test_root = os.path.join(data_root, 'test')
    train_set, test_set = make_train_test_datapair_division(
        NUM_RAN[0],
        NUM_RAN[1],
        0.7,
        SINGLE_MARKERS,
        SINGLE_COLOR,
        mod_pairs,
    )
    render_dataset(train_set, train_root, comp_division)
    render_dataset(test_set, test_root, comp_division)


def make_dataset_multi_style_plus():
    marks = ['s', 'o', 'v', 'd']
    colors = ['purple', 'salmon', 'olive', 'blue']
    data_root = f'dataset/multi_style_({len(marks)},{len(colors)})_realPairs_plus({NUM_RAN[0]},{NUM_RAN[1]})'
    os.makedirs(data_root, exist_ok=True)
    train_root = os.path.join(data_root, 'train')
    test_root = os.path.join(data_root, 'test')
    train_set, test_set = make_train_test_datapair_maxN_no_leak(
        NUM_RAN[0],
        NUM_RAN[1],
        3,
        0.33,
        marks,
        colors,
        sum_pairs(NUM_RAN[0]),
    )
    render_dataset(train_set, train_root, comp_plus)
    render_dataset(test_set, test_root, comp_plus)


def make_hundred_dataset_plus_arabic():
    marks = ['default']
    colors = ['blue']
    start = 0
    end = 99
    data_root = f'dataset/single_style_pairs({start},{end})_arabic'
    os.makedirs(data_root, exist_ok=True)
    train_root = os.path.join(data_root, 'train')
    test_root = os.path.join(data_root, 'test')
    train_set, test_set = make_train_test_datapair_maxN_no_leak(
        start,
        end,
        3,
        0.33,
        marks,
        colors,
        sum_pairs(start),
    )
    render_arabic_num_dataset(train_set, train_root, comp_plus)
    render_arabic_num_dataset(test_set, test_root, comp_plus)


def make_dataset_single_style_plus_one_triple_set():
    marks = ['o']
    colors = ['blue']
    start = 1
    end = 20
    data_root = f'dataset/single_style_plus_one_triple_set({start},{end})'
    os.makedirs(data_root, exist_ok=True)
    train_root = os.path.join(data_root, 'train')
    test_root_1 = os.path.join(data_root, 'test_1')
    test_root_2 = os.path.join(data_root, 'test_2')
    train_set, test_set_1, test_set_2 = make_plusone_triple_datapair_maxN_no_leak(
        start,
        end,
        marks,
        colors,
        sum_pairs(start),
    )
    render_dataset(train_set, train_root, comp_plus)
    render_dataset(test_set_1, test_root_1, comp_plus)
    render_dataset(test_set_2, test_root_2, comp_plus)


def make_dataset_single_style_plus_one_double_set():
    marks = ['o']
    colors = ['blue']
    start = 1
    end = 20
    data_root = f'dataset/single_style_plus_one_double_set({start},{end})'
    os.makedirs(data_root, exist_ok=True)
    train_root = os.path.join(data_root, 'train')
    test_root = os.path.join(data_root, 'test')
    train_set, test_set_1,  = make_plusone_double_datapair_maxN_no_leak(
        start,
        end,
        marks,
        colors,
        sum_pairs(start),
    )
    render_dataset(train_set, train_root, comp_plus)
    render_dataset(test_set_1, test_root, comp_plus)


def make_dataset_single_style_plus_half_double_set():
    marks = ['o']
    colors = ['blue']
    start = 1
    end = 20
    data_root = f'dataset/single_style_plus_half_double_set({start},{end})'
    os.makedirs(data_root, exist_ok=True)
    train_root = os.path.join(data_root, 'train')
    test_root = os.path.join(data_root, 'test')
    train_set, test_set_1,  = make_plus_half_double_datapair_maxN_no_leak(
        start,
        end,
        marks,
        colors,
        sum_pairs(start),
    )
    render_dataset(train_set, train_root, comp_plus)
    render_dataset(test_set_1, test_root, comp_plus)


def make_dataset_single_style_plus_random_one_shot_triple():
    marks = ['o']
    colors = ['blue']
    start = 1
    end = 20
    data_root = f'dataset/single_style_plus_random_one_shot_triple_set({start},{end})'
    train_root = os.path.join(data_root, 'train')
    test_root_1 = os.path.join(data_root, 'test_1')
    test_root_2 = os.path.join(data_root, 'test_2')
    train_set, test_set_1, test_set_2 = make_random_one_shot_triple_datapair_no_leak(
        start,
        end,
        marks,
        colors,
        sum_pairs(start),
    )
    render_dataset(train_set, train_root, comp_plus)
    render_dataset(test_set_1, test_root_1, comp_plus)
    render_dataset(test_set_2, test_root_2, comp_plus)


if __name__ == "__main__":
    # make_dataset_single_style_plus_one_double_set()
    # make_dataset_multi_style_plus()
    # make_train_dataset_n2(NUMBERS, MARKERS, DATA_PATH)
    # make_dataset_single_style_plus_random_one_shot_triple()
    make_hundred_dataset_plus_arabic()