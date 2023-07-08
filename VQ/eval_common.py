import matplotlib
import matplotlib.pyplot as plt
from matplotlib import collections as matcoll


def calc_real_digit_num(latent_embedding_1, multi_num_embeddings):
    if multi_num_embeddings is None:
        return latent_embedding_1
    else:
        return len(multi_num_embeddings)


class EvalHelper:
    def __init__(self, config):
        self.config = config
        self.isVQStyle = config['isVQStyle']
        self.digit_num = calc_real_digit_num(config['latent_embedding_1'], config['multi_num_embeddings'])
        self.embedding_dim = config['embedding_dim']
        self.latent_code_1 = config['latent_embedding_1'] * self.embedding_dim
        self.latent_code_2 = config['latent_embedding_2'] * self.embedding_dim if \
            self.isVQStyle else config['latent_code_2']

        self.VQ_dim = self.find_VQ_dim()

    def calc_subfigures_row_col(self):
        n_row = self.digit_num
        n_col = self.embedding_dim
        while 2 * n_row <= n_col / 2 and n_col % 2 == 0:
            n_row = n_row * 2
            n_col = n_col / 2
        return int(n_row), int(n_col)

    def find_VQ_dim(self):
        if self.isVQStyle:
            return range(2)
        else:
            return range(1)

    def set_axis(self, ax: plt.Axes, dim_idx):
        if dim_idx in self.VQ_dim:
            ax.set(ylabel='Embedding index')
            if dim_idx == 0:
                ax.set_title("Content embedding")
            else:
                ax.set_title("Style embedding")
        else:
            ax.set(ylabel='z value')
            ax.set_title(f"z_style_{dim_idx}")

    def draw_scatter_point_line_or_grid(self, ax: plt.Axes, dim_idx, x, y):
        if dim_idx in self.VQ_dim:
            draw_scatter_point_line(ax, x, y)
        else:
            ax.grid(True)


def draw_scatter_point_line(ax: plt.Axes, x, y):
    lines = []
    for j in range(len(x)):
        pair = [(0, y[j]), (x[j], y[j])]
        lines.append(pair)
    linecoll = matcoll.LineCollection(lines, linewidths=0.2)
    ax.add_collection(linecoll)
    ax.grid(True, axis='x')