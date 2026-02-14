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


def make_dataset_1(data_path, numbers, markers, colors, size=18):
    os.makedirs(data_path, exist_ok=True)
    for i in numbers:
        for mar in markers:
            for color in colors:
                position = DOT_POSITIONS[i]
                fig_path = os.path.join(data_path, f'{i}-{MARK_NAME_SPACE[mar]}-{color}')
                plot_a_scatter(position, fig_path, marker=mar, color=color, is_fill=i != 0, size=size)


def make_arabic_num_dataset(data_path, numbers, colors):
    os.makedirs(data_path, exist_ok=True)
    for i in numbers:
        for color in colors:
            fig_path = os.path.join(data_path, f'{i}-default-{color}')
            plot_arabic_numbers(i, fig_path, color)


def make_ZHENG_num_dataset(data_path, numbers, colors, marker_name='zheng', fig_size=0.64, lim=0.5):
    os.makedirs(data_path, exist_ok=True)
    for i in numbers:
        for color in colors:
            fig_path = os.path.join(data_path, f'{i}-{marker_name}-{color}')
            plot_lines(ZHENG_POSITIONS[i], fig_path, color, fig_size=fig_size, lim=lim)

def make_EU_tally_mark_num_dataset(data_path, numbers, colors, maker_name='eutally', fig_size=0.64, lim=0.5):
    os.makedirs(data_path, exist_ok=True)
    for i in numbers:
        for color in colors:
            fig_path = os.path.join(data_path, f'{i}-{maker_name}-{color}')
            plot_lines(EU_tally_mark_POSITIONS[i], fig_path, color, line_width=1.5, fig_size=fig_size, lim=lim)

if __name__ == "__main__":
    """
    make FixedPos-oneStyle_EU_tally
    """
    # data_path = f'{DATA_ROOT}/(0,20)-FixedPos-oneStyle_EU_tally'
    # make_EU_tally_mark_num_dataset(data_path, range(0, 21), ['blue'])

    # big_EU_data_path = f'data_plot/(0,20)-FixedPos-EU-big'
    # make_EU_tally_mark_num_dataset(big_EU_data_path, range(0, 21), ['blue'], fig_size=0.4, lim=0.25)
    # big_Zheng_data_path = f'data_plot/(0,20)-FixedPos-ZHENG-big'
    # make_ZHENG_num_dataset(big_Zheng_data_path, range(0, 21), ['blue'], fig_size=0.4, lim=0.25)

    # data_path = f'data_plot/(0,20)-FixedPos-multiStyle-big'
    # make_dataset_1(data_path, NUMBERS, TRAIN_MARKERS, TRAIN_COLORS, size=48)

    # arabic_colorful_data_path = f'{DATA_ROOT}/(0,20)-FixedPos-arabic-blue'
    # arabic_colorful_data_colors = ['blue']
    # make_arabic_num_dataset(arabic_colorful_data_path, NUMBERS, arabic_colorful_data_colors)

    data_path = f'{DATA_ROOT}/(0,20)-FixedPos-multiStyle'
    make_dataset_1(data_path, NUMBERS, TRAIN_MARKERS, TRAIN_COLORS)