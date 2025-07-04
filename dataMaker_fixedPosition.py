import random
import matplotlib
import matplotlib.markers
import matplotlib.pyplot as plt
from dataMaker_commonFunc import *

matplotlib.use('AGG')
from scipy.spatial import distance
import os

NUMBERS = range(0, 21)
TRAIN_MARKERS = ['s', 'o', 'v', 'd']
NEW_MARKERS = ['^', 'X', 'p', 'D']
# MARKERS = ['o']
DATA_ROOT = 'dataset'

TRAIN_COLORS = ['purple', 'salmon',  'olive', 'blue']
NEW_COLORS = ['red', 'green', 'black', 'yellow']


TRAIN_SET_CONF = [
    f'{DATA_ROOT}/multi_style_eval_({NUMBERS[0]},{NUMBERS[-1]})_FixedPos_trainStyle',
    NUMBERS,
    TRAIN_MARKERS,
    TRAIN_COLORS,
]

NEW_SHAPE_CONF = [
    f'{DATA_ROOT}/multi_style_eval_({NUMBERS[0]},{NUMBERS[-1]})_FixedPos_newShape',
    NUMBERS,
    NEW_MARKERS,
    TRAIN_COLORS,
]

NEW_COLOR_CONF = [
    f'{DATA_ROOT}/multi_style_eval_({NUMBERS[0]},{NUMBERS[-1]})_FixedPos_newColor',
    NUMBERS,
    TRAIN_MARKERS,
    NEW_COLORS,
]

NEW_SHAPE_COLOR_CONF = [
    f'{DATA_ROOT}/multi_style_eval_({NUMBERS[0]},{NUMBERS[-1]})_FixedPos_newShapeColor',
    NUMBERS,
    NEW_MARKERS,
    NEW_COLORS,
]


def make_dataset_1(data_path, numbers, markers, colors):
    os.makedirs(data_path, exist_ok=True)
    for i in numbers:
        for mar in markers:
            for color in colors:
                position = DOT_POSITIONS[i]
                fig_path = os.path.join(data_path, f'{i}-{MARK_NAME_SPACE[mar]}-{color}')
                plot_a_scatter(position, fig_path, marker=mar, color=color, is_fill=i != 0)


def make_arabic_num_dataset(data_path, numbers, colors):
    os.makedirs(data_path, exist_ok=True)
    for i in numbers:
        for color in colors:
            fig_path = os.path.join(data_path, f'{i}-default-{color}')
            plot_arabic_numbers(i, fig_path, color)


def make_ZHENG_num_dataset(data_path, numbers, colors, marker_name='zheng'):
    os.makedirs(data_path, exist_ok=True)
    for i in numbers:
        for color in colors:
            fig_path = os.path.join(data_path, f'{i}-{marker_name}-{color}')
            plot_lines(ZHENG_POSITIONS[i], fig_path, color)

def make_EU_tally_mark_num_dataset(data_path, numbers, colors, maker_name='eutally'):
    os.makedirs(data_path, exist_ok=True)
    for i in numbers:
        for color in colors:
            fig_path = os.path.join(data_path, f'{i}-{maker_name}-{color}')
            plot_lines(EU_tally_mark_POSITIONS[i], fig_path, color, line_width=1.5)

if __name__ == "__main__":
    """
    make FixedPos-oneStyle_EU_tally
    """
    # data_path = f'{DATA_ROOT}/(0,20)-FixedPos-oneStyle_EU_tally'
    # make_EU_tally_mark_num_dataset(data_path, range(0, 21), ['blue'])

    data_path = f'{DATA_ROOT}/(0,20)-FixedPos-mahjong'
    make_EU_tally_mark_num_dataset(data_path, range(0, 21), ['blue'])
    make_ZHENG_num_dataset(data_path, range(0, 21), ['blue'])
