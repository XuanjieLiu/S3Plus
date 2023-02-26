import matplotlib
import matplotlib.pyplot as plt
from matplotlib import collections as matcoll


class EvalHelper:
    def __init__(self, config):
        self.config = config
        self.isVQStyle = config['isVQStyle']
        self.embeddings_num = config['embeddings_num']
        self.embedding_dim = config['embedding_dim']
        self.latent_code_1 = config['latent_embedding_1'] * self.embedding_dim
        self.latent_code_2 = config['latent_embedding_2'] * self.embedding_dim if \
            self.isVQStyle else config['latent_code_2']
        self.VQ_dim = self.find_VQ_dim()

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

    def draw_scatter_point_line(self, ax: plt.Axes, dim_idx, x, y):
        if dim_idx in self.VQ_dim:
            lines = []
            for j in range(len(x)):
                pair = [(0, y[j]), (x[j], y[j])]
                lines.append(pair)
            linecoll = matcoll.LineCollection(lines, linewidths=0.2)
            ax.add_collection(linecoll)
            ax.grid(True, axis='x')
        else:
            ax.grid(True)