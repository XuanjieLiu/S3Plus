import matplotlib.pyplot as plt

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


def Zero(center: tuple):
    return [(center[0]+10, center[1]+10)]


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


ANCHORS = {
    1: [(0.0, 0.0)],
    2: [(-0.25, 0.0), (0.25, 0.0)],
    3: [(-0.25, 0.25), (0.25, 0.25), (0.0, -0.25)],
    4: [(-0.25, 0.25), (0.25, 0.25), (0.25, -0.25), (-0.25, -0.25)]
}

POSITIONS = {
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

