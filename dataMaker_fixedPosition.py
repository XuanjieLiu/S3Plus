import random
import matplotlib
import matplotlib.markers
import matplotlib.pyplot as plt
from dataMaker_commonFunc import *
matplotlib.use('AGG')
from scipy.spatial import distance
import os

NUMBERS = range(1, 17)
MARKERS = ['o', 'v', '*', 'd']
DATA_ROOT = 'dataset'
DATA_PATH = f'{DATA_ROOT}/({NUMBERS[0]},{NUMBERS[-1]})-FixedPos-4Color'

# COLORS_TRAIN = ['purple', 'brown', 'salmon', 'chocolate', 'olive', 'blue', 'deeppink', 'teal']
COLORS_TRAIN = ['purple', 'salmon',  'olive', 'blue']


def make_dataset_1():
    os.makedirs(DATA_PATH, exist_ok=True)
    for i in NUMBERS:
        for mar in MARKERS:
            for color in COLORS_TRAIN:
                position = POSITIONS[i]
                fig_path = os.path.join(DATA_PATH, f'{i}-{MARK_NAME_SPACE[mar]}-{color}')
                plot_a_scatter(position, fig_path, marker=mar, color=color)


if __name__ == "__main__":
    make_dataset_1()
