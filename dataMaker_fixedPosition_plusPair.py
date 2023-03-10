import random
import matplotlib
import matplotlib.markers
import matplotlib.pyplot as plt
from typing import List
from dataMaker_commonFunc import *
from tqdm import tqdm

matplotlib.use('AGG')
from scipy.spatial import distance
import os


NUMBERS = range(1, 11)
MARKERS = ['o', 'v', '*', 'd']
DATA_ROOT = 'dataset'
DATA_PATH = f'{DATA_ROOT}/PlusPair-({NUMBERS[0]},{NUMBERS[-1]})-FixedPos'
COLORS_TRAIN = ['purple', 'salmon', 'olive', 'blue']

SINGLE_STYLE_DATA_ROOT = 'dataset/single_style_pairs'
SINGLE_MARKERS = ['o']
SINGLE_COLOR = ['blue']


def draw_plus_data(i, j, mar, color, data_path):
    os.makedirs(data_path, exist_ok=True)
    plot_a_scatter(
        POSITIONS[i],
        save_dir=os.path.join(data_path, f'a-{i}'),
        marker=mar, color=color, is_fill=i != 0
    )
    plot_a_scatter(
        POSITIONS[j],
        save_dir=os.path.join(data_path, f'b-{j}'),
        marker=mar, color=color, is_fill=j != 0
    )
    plot_a_scatter(
        POSITIONS[i + j],
        save_dir=os.path.join(data_path, f'c-{i + j}'),
        marker=mar, color=color, is_fill=i+j != 0
    )


def make_train_dataset_n2(numbers, markers, colors, dataset_path):
    data_root = os.path.join(dataset_path, 'train')
    os.makedirs(data_root, exist_ok=True)
    for i in numbers:
        for j in range(i, numbers[-1]+1):
            for mar in markers:
                for color in colors:
                    data_name = f'{i}-{j}-{MARK_NAME_SPACE[mar]}-{color}'
                    data_path = os.path.join(data_root, data_name)
                    os.makedirs(data_path, exist_ok=True)
                    draw_plus_data(i, j, mar, color, data_path)


class PairData:
    def __init__(self, a: int, b: int, marker: str = None, color: str = None, path: str = None):
        self.a = a
        self.b = b
        self.marker = marker
        self.color = color
        self.path = path


def make_train_test_dataset_maxN(max_number, min_sample_num, sample_rate, markers, colors, data_root):
    train_root = os.path.join(data_root, 'train')
    test_root = os.path.join(data_root, 'test')
    train_set = []
    test_set = []
    for mar in markers:
        for color in colors:
            for i in range(2, max_number+1):
                data = []
                all_idx = [x for x in range(0, i-1)]
                sample_num = min(i-1, max(min_sample_num, int(round(i * sample_rate, 0))))
                train_idx = random.sample(all_idx, sample_num)
                for a, b in sum_pairs(i):
                    data.append(PairData(a, b, mar, color))
                for x in all_idx:
                    if x in train_idx:
                        train_set.append(data[x])
                    else:
                        test_set.append(data[x])
    render_dataset(train_set, train_root)
    render_dataset(test_set, test_root)


def render_dataset(data_list: List[PairData], data_root: str):
    for data in tqdm(data_list, desc=data_root):
        a = data.a
        b = data.b
        color = data.color
        marker = data.marker
        data_name = f'{a}-{b}-{marker}-{color}'
        data_path = os.path.join(data_root, data_name)
        os.makedirs(data_path, exist_ok=True)
        draw_plus_data(a, b, marker, color, data_path)



def sum_pairs(max_number):
    for a in range(1, max_number):
        b = max_number - a
        yield a, b



if __name__ == "__main__":
    # make_train_dataset_n2(NUMBERS, MARKERS, DATA_PATH)
    make_train_test_dataset_maxN(20, 2, 0.33, SINGLE_MARKERS, SINGLE_COLOR, DATA_ROOT)