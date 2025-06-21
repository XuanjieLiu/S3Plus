import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

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

MARK_NAME_SPACE_MAHJONG = {
    'o': 'circle',
    's': 'zheng',
    '+': 'eutally',
}


def plot_arabic_numbers(num, save_dir, color: str):
    fig = plt.figure(figsize=(0.64, 0.64))
    a1 = fig.add_axes([0, 0, 1, 1])
    a1.scatter(0, 0,
               c=color,
               marker=f'${num}$',
               s=800,
               )
    a1.set_ylim(-0.5, 0.5)
    a1.set_xlim(-0.5, 0.5)
    plt.savefig(save_dir)
    plt.cla()
    plt.clf()
    plt.close()
    return


def plot_a_scatter(position_list, save_dir, marker: str, color: str, is_fill=True):
    x = [n[0] for n in position_list]
    y = [n[1] for n in position_list]
    fig = plt.figure(figsize=(0.64, 0.64))
    a1 = fig.add_axes([0, 0, 1, 1])
    a1.scatter(x, y,
               c=color if is_fill else 'none',
               facecolors=color if is_fill else 'none',
               edgecolors=color,
               marker=marker,
               s=18,
               )
    a1.set_ylim(-0.5, 0.5)
    a1.set_xlim(-0.5, 0.5)
    # plt.show()
    plt.savefig(save_dir)
    plt.cla()
    plt.clf()
    plt.close(fig)
    return


def plot_lines(line_list, save_dir = './a.png', color: str='b', line_width: float=2.):
    # 创建一个新的图形和轴
    fig, ax = plt.subplots()
    #设置figuresize
    fig.set_size_inches(0.64, 0.64)
    # 添加多条线段
    for l in line_list:
        x = [n[0] for n in l]
        y = [n[1] for n in l]
        line = Line2D(x, y, linewidth=line_width, color=color)
        ax.add_line(line)
    # 设置轴的范围
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlim(-0.5, 0.5)
    # 隐藏轴
    ax.axis('off')
    # 保存图形
    plt.savefig(save_dir)
    plt.cla()
    plt.clf()
    plt.close(fig)
    return


def Zero(center: tuple):
    # return [(center[0]+10, center[1]+10)]
    return [center]


def One(center: tuple):
    return [center]


def Two(center: tuple, interval: float = 0.125):
    return [(center[0], center[1] - interval), (center[0], center[1] + interval)]


def Three(center: tuple, interval: float = 0.125):
    return [
        (center[0] - interval, center[1] - interval),
        (center[0] - interval, center[1] + interval),
        (center[0] + interval, center[1] + interval),
    ]


def Four(center: tuple, interval: float = 0.125):
    return [
        (center[0] - interval, center[1] - interval),
        (center[0] - interval, center[1] + interval),
        (center[0] + interval, center[1] + interval),
        (center[0] + interval, center[1] - interval),
    ]


def Five(center: tuple, interval: float = 0.125):
    return [
        (center[0] - interval, center[1] - interval),
        (center[0] - interval, center[1] + interval),
        (center[0] + interval, center[1] + interval),
        (center[0] + interval, center[1] - interval),
        center
    ]


def zheng_0(center: tuple = (0, 0), scalar: float = 1.):
    return np.array([[[-0.03, 0.], [0.03, 0.0]]]) * scalar + np.array(center)


def zheng_1(center: tuple = (0, 0), scalar: float = 1.):
    return np.array([[[-0.16, 0.16], [0.16, 0.16]]]) * scalar + np.array(center)


def zheng_2(center: tuple = (0, 0), scalar: float = 1.):
    return np.vstack((
        zheng_1(center, scalar),
        np.array([[[0., 0.16], [0., -0.16]]]) * scalar + np.array(center)
    ))


def zheng_3(center: tuple = (0, 0), scalar: float = 1.):
    return np.vstack((
        zheng_2(center, scalar),
        np.array([[[0., 0.], [0.14, 0.]]]) * scalar + np.array(center)
    ))


def zheng_4(center: tuple = (0, 0), scalar: float = 1.):
    return np.vstack((
        zheng_3(center, scalar),
        np.array([[[-0.12, 0.01], [-0.12, -0.16]]]) * scalar + np.array(center)
    ))


def zheng_5(center: tuple = (0, 0), scalar: float = 1.):
    return np.vstack((
        zheng_4(center, scalar),
        np.array([[[-0.16, -0.16], [0.16, -0.16]]]) * scalar + np.array(center)
    ))


def EU_tally_mark_0(center: tuple = (0, 0), scalar: float = 1.):
    return np.array([[[0.0, -0.03], [0.0, 0.03]]]) * scalar + np.array(center)


def EU_tally_mark_1(center: tuple = (0, 0), scalar: float = 1.):
    return np.array([[[-0.15, 0.16], [-0.15, -0.16]]]) * scalar + np.array(center)


def EU_tally_mark_2(center: tuple = (0, 0), scalar: float = 1.):
    return np.vstack((
        EU_tally_mark_1(center, scalar),
        np.array([[[-0.05, 0.16], [-0.05, -0.16]]]) * scalar + np.array(center)
    ))

def EU_tally_mark_3(center: tuple = (0, 0), scalar: float = 1.):
    return np.vstack((
        EU_tally_mark_2(center, scalar),
        np.array([[[0.05, 0.16], [0.05, -0.16]]]) * scalar + np.array(center)
    ))

def EU_tally_mark_4(center: tuple = (0, 0), scalar: float = 1.):
    return np.vstack((
        EU_tally_mark_3(center, scalar),
        np.array([[[0.15, 0.16], [0.15, -0.16]]]) * scalar + np.array(center)
    ))

def EU_tally_mark_5(center: tuple = (0, 0), scalar: float = 1.):
    return np.vstack((
        EU_tally_mark_4(center, scalar),
        np.array([[[-0.2, -0.05], [0.2, 0.05]]]) * scalar + np.array(center)
    ))



ANCHORS = {
    1: [(0.0, 0.0)],
    2: [(-0.25, 0.0), (0.25, 0.0)],
    3: [(-0.25, 0.25), (0.25, 0.25), (0.0, -0.25)],
    4: [(-0.25, 0.25), (0.25, 0.25), (0.25, -0.25), (-0.25, -0.25)]
}

DOT_POSITIONS = {
    0:    Zero(ANCHORS[1][0]),
    1:    One(ANCHORS[1][0]),
    2:    Two(ANCHORS[1][0], interval=0.2),
    3:    Three(ANCHORS[1][0], interval=0.2),
    4:    Four(ANCHORS[1][0], interval=0.2),
    5:    Five(ANCHORS[1][0], interval=0.2),
    6:  [*Five(ANCHORS[2][0]), *One(ANCHORS[2][1])],
    7:  [*Five(ANCHORS[2][0]), *Two(ANCHORS[2][1])],
    8:  [*Five(ANCHORS[2][0]), *Three(ANCHORS[2][1])],
    9:  [*Five(ANCHORS[2][0]), *Four(ANCHORS[2][1])],
    10: [*Five(ANCHORS[2][0]), *Five(ANCHORS[2][1])],
    11: [*Five(ANCHORS[3][0]), *Five(ANCHORS[3][1]), *One(ANCHORS[3][2])],
    12: [*Five(ANCHORS[3][0]), *Five(ANCHORS[3][1]), *Two(ANCHORS[3][2])],
    13: [*Five(ANCHORS[3][0]), *Five(ANCHORS[3][1]), *Three(ANCHORS[3][2])],
    14: [*Five(ANCHORS[3][0]), *Five(ANCHORS[3][1]), *Four(ANCHORS[3][2])],
    15: [*Five(ANCHORS[3][0]), *Five(ANCHORS[3][1]), *Five(ANCHORS[3][2])],
    16: [*Five(ANCHORS[4][0]), *Five(ANCHORS[4][1]), *Five(ANCHORS[4][2]), *One(ANCHORS[4][3])],
    17: [*Five(ANCHORS[4][0]), *Five(ANCHORS[4][1]), *Five(ANCHORS[4][2]), *Two(ANCHORS[4][3])],
    18: [*Five(ANCHORS[4][0]), *Five(ANCHORS[4][1]), *Five(ANCHORS[4][2]), *Three(ANCHORS[4][3])],
    19: [*Five(ANCHORS[4][0]), *Five(ANCHORS[4][1]), *Five(ANCHORS[4][2]), *Four(ANCHORS[4][3])],
    20: [*Five(ANCHORS[4][0]), *Five(ANCHORS[4][1]), *Five(ANCHORS[4][2]), *Five(ANCHORS[4][3])],
}

ZHENG_POSITIONS = {
    0: zheng_0(scalar=1.3),
    1: zheng_1(),
    2: zheng_2(),
    3: zheng_3(),
    4: zheng_4(),
    5: zheng_5(),
    6: [*zheng_5(ANCHORS[2][0]), *zheng_1(ANCHORS[2][1])],
    7: [*zheng_5(ANCHORS[2][0]), *zheng_2(ANCHORS[2][1])],
    8: [*zheng_5(ANCHORS[2][0]), *zheng_3(ANCHORS[2][1])],
    9: [*zheng_5(ANCHORS[2][0]), *zheng_4(ANCHORS[2][1])],
    10: [*zheng_5(ANCHORS[2][0]), *zheng_5(ANCHORS[2][1])],
    11: [*zheng_5(ANCHORS[3][0]), *zheng_5(ANCHORS[3][1]), *zheng_1(ANCHORS[3][2])],
    12: [*zheng_5(ANCHORS[3][0]), *zheng_5(ANCHORS[3][1]), *zheng_2(ANCHORS[3][2])],
    13: [*zheng_5(ANCHORS[3][0]), *zheng_5(ANCHORS[3][1]), *zheng_3(ANCHORS[3][2])],
    14: [*zheng_5(ANCHORS[3][0]), *zheng_5(ANCHORS[3][1]), *zheng_4(ANCHORS[3][2])],
    15: [*zheng_5(ANCHORS[3][0]), *zheng_5(ANCHORS[3][1]), *zheng_5(ANCHORS[3][2])],
    16: [*zheng_5(ANCHORS[4][0]), *zheng_5(ANCHORS[4][1]), *zheng_5(ANCHORS[4][2]), *zheng_1(ANCHORS[4][3])],
    17: [*zheng_5(ANCHORS[4][0]), *zheng_5(ANCHORS[4][1]), *zheng_5(ANCHORS[4][2]), *zheng_2(ANCHORS[4][3])],
    18: [*zheng_5(ANCHORS[4][0]), *zheng_5(ANCHORS[4][1]), *zheng_5(ANCHORS[4][2]), *zheng_3(ANCHORS[4][3])],
    19: [*zheng_5(ANCHORS[4][0]), *zheng_5(ANCHORS[4][1]), *zheng_5(ANCHORS[4][2]), *zheng_4(ANCHORS[4][3])],
    20: [*zheng_5(ANCHORS[4][0]), *zheng_5(ANCHORS[4][1]), *zheng_5(ANCHORS[4][2]), *zheng_5(ANCHORS[4][3])],
}

EU_tally_mark_POSITIONS = {
    0: EU_tally_mark_0(scalar=1.3),
    1: EU_tally_mark_1(),
    2: EU_tally_mark_2(),
    3: EU_tally_mark_3(),
    4: EU_tally_mark_4(),
    5: EU_tally_mark_5(),
    6: [*EU_tally_mark_5(ANCHORS[2][0]), *EU_tally_mark_1(ANCHORS[2][1])],
    7: [*EU_tally_mark_5(ANCHORS[2][0]), *EU_tally_mark_2(ANCHORS[2][1])],
    8: [*EU_tally_mark_5(ANCHORS[2][0]), *EU_tally_mark_3(ANCHORS[2][1])],
    9: [*EU_tally_mark_5(ANCHORS[2][0]), *EU_tally_mark_4(ANCHORS[2][1])],
    10: [*EU_tally_mark_5(ANCHORS[2][0]), *EU_tally_mark_5(ANCHORS[2][1])],
    11: [*EU_tally_mark_5(ANCHORS[3][0]), *EU_tally_mark_5(ANCHORS[3][1]), *EU_tally_mark_1(ANCHORS[3][2])],
    12: [*EU_tally_mark_5(ANCHORS[3][0]), *EU_tally_mark_5(ANCHORS[3][1]), *EU_tally_mark_2(ANCHORS[3][2])],
    13: [*EU_tally_mark_5(ANCHORS[3][0]), *EU_tally_mark_5(ANCHORS[3][1]), *EU_tally_mark_3(ANCHORS[3][2])],
    14: [*EU_tally_mark_5(ANCHORS[3][0]), *EU_tally_mark_5(ANCHORS[3][1]), *EU_tally_mark_4(ANCHORS[3][2])],
    15: [*EU_tally_mark_5(ANCHORS[3][0]), *EU_tally_mark_5(ANCHORS[3][1]), *EU_tally_mark_5(ANCHORS[3][2])],
    16: [*EU_tally_mark_5(ANCHORS[4][0]), *EU_tally_mark_5(ANCHORS[4][1]), *EU_tally_mark_5(ANCHORS[4][2]), *EU_tally_mark_1(ANCHORS[4][3])],
    17: [*EU_tally_mark_5(ANCHORS[4][0]), *EU_tally_mark_5(ANCHORS[4][1]), *EU_tally_mark_5(ANCHORS[4][2]), *EU_tally_mark_2(ANCHORS[4][3])],
    18: [*EU_tally_mark_5(ANCHORS[4][0]), *EU_tally_mark_5(ANCHORS[4][1]), *EU_tally_mark_5(ANCHORS[4][2]), *EU_tally_mark_3(ANCHORS[4][3])],
    19: [*EU_tally_mark_5(ANCHORS[4][0]), *EU_tally_mark_5(ANCHORS[4][1]), *EU_tally_mark_5(ANCHORS[4][2]), *EU_tally_mark_4(ANCHORS[4][3])],
    20: [*EU_tally_mark_5(ANCHORS[4][0]), *EU_tally_mark_5(ANCHORS[4][1]), *EU_tally_mark_5(ANCHORS[4][2]), *EU_tally_mark_5(ANCHORS[4][3])],
}

if "__main__" == __name__:

    print(zheng_4())
    plot_a_scatter(DOT_POSITIONS[17], './17.png', marker='o', color='b', is_fill=True)