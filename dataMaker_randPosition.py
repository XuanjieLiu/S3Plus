import random
import matplotlib
import matplotlib.markers
import matplotlib.pyplot as plt
matplotlib.use('AGG')
from scipy.spatial import distance
import os

MARK_NAME_SPACE = {
        '.': 'point',
        ',': 'pixel',
        'o': 'circle',
        'v': 'triangle_down',
        '^': 'triangle_up',
        '<': 'triangle_left',
        '>': 'triangle_right',
        '1': 'tri_down',
        '2': 'tri_up',
        '3': 'tri_left',
        '4': 'tri_right',
        '8': 'octagon',
        's': 'square',
        'p': 'pentagon',
        '*': 'star',
        'h': 'hexagon1',
        'H': 'hexagon2',
        '+': 'plus',
        'x': 'x',
        'D': 'diamond',
        'd': 'thin_diamond',
        '|': 'vline',
        '_': 'hline',
        'P': 'plus_filled',
        'X': 'x_filled',
}

POINT_RANGE = (0, 20)
SAMPLE_RANGE = (0.1, 0.90)
MIN_DIST = 0.15

NUMBERS = range(1, 11)
NUM_NUMS = 20
# MARKERS = ['o', 'v', '*', 'd']
MARKERS = ['v']
DATA_ROOT = 'dataset'
DATA_PATH = f'{DATA_ROOT}/RandPos_({NUMBERS[0]},{NUMBERS[-1]})-{NUM_NUMS}-{str([MARK_NAME_SPACE[name] for name in MARKERS])}'
aaa = matplotlib.markers.CARETDOWNBASE


def make_dataset_1():
    os.makedirs(DATA_PATH, exist_ok=True)
    for i in NUMBERS:
        for mar in MARKERS:
            for j in range(0, NUM_NUMS):
                position = gen_positions(i, SAMPLE_RANGE, MIN_DIST)
                fig_path = os.path.join(DATA_PATH, f'{i}-{MARK_NAME_SPACE[mar]}-{j}')
                plot_a_scatter(position, fig_path, marker=mar, color='black')


def plot_a_scatter(position_list, save_dir, marker: str, color: str):
    x = [n[0] for n in position_list]
    y = [n[1] for n in position_list]
    fig = plt.figure(figsize=(0.64, 0.64))
    a1 = fig.add_axes([0, 0, 1, 1])
    a1.scatter(x, y, c=color, marker=marker, s=18)
    a1.set_ylim(0, 1)
    a1.set_xlim(0, 1)
    # plt.show()
    plt.savefig(save_dir)
    plt.cla()
    plt.clf()
    plt.close(fig)
    return


def gen_positions(num, sample_range, min_dist):
    def gen_a_point_keep_apart(apart_list):
        point = (random.uniform(*sample_range), random.uniform(*sample_range))
        for exist_p in apart_list:
            dist = distance.euclidean(exist_p, point)
            if dist < min_dist:
                return gen_a_point_keep_apart(apart_list)
        return point

    points = []
    for i in range(0, num):
        points.append(gen_a_point_keep_apart(points))
    return points


make_dataset_1()
