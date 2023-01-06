import random
import matplotlib
import matplotlib.markers
import matplotlib.pyplot as plt
from dataMaker_commonFunc import *

matplotlib.use('AGG')
from scipy.spatial import distance
import os


NUMBERS = range(1, 11)
MARKERS = ['o', 'v', '*', 'd']
# MARKERS = ['o']
DATA_ROOT = 'dataset'
DATA_PATH = f'{DATA_ROOT}/PlusPair-({NUMBERS[0]},{NUMBERS[-1]})-FixedPos'
COLORS_TRAIN = ['purple', 'salmon', 'olive', 'blue']
# COLORS_TRAIN = ['blue']


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


def make_train_dataset():
    data_root = os.path.join(DATA_PATH, 'train')
    os.makedirs(data_root, exist_ok=True)
    for i in NUMBERS:
        for j in range(i, NUMBERS[-1]+1):
            for mar in MARKERS:
                for color in COLORS_TRAIN:
                    data_name = f'{i}-{j}-{MARK_NAME_SPACE[mar]}-{color}'
                    data_path = os.path.join(data_root, data_name)
                    os.makedirs(data_path, exist_ok=True)
                    draw_plus_data(i, j, mar, color, data_path)


if __name__ == "__main__":
    make_train_dataset()
