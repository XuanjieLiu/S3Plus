import matplotlib
import matplotlib.pyplot as plt
from matplotlib import collections as matcoll


class EvalHelper:
    def __init__(self, config):
        self.config = config
        self.VQ_dim = self.find_VQ_dim()

    def find_VQ_dim(self):
        isVQStyle = self.config['isVQStyle']
        latent_code_1 = self.config['latent_code_1']
        latent_code_2 = self.config['latent_code_2']
        if isVQStyle:
            return range(latent_code_1 + latent_code_2)
        else:
            return range(latent_code_1)

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