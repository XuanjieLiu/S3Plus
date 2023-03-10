import os
from typing import List

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.optim as optim
from dataloader import SingleImgDataset
from loss_counter import LossCounter
from VQVAE import VQVAE
from shared import *
import matplotlib
import matplotlib.markers
import matplotlib.pyplot as plt
from eval_common import EvalHelper

matplotlib.use('AGG')

MODEL_PATH = 'curr_model.pt'
EVAL_PATH = f'num_eval-{MODEL_PATH.split(".")[0]}/'


def plot_z_against_label(num_z, num_labels, eval_path, eval_helper: EvalHelper = None):
    fig, axs = plt.subplots(1, num_z.size(1), figsize=(num_z.size(1) * 7, 5))
    if num_z.size(1) == 1:
        axs = [axs]
    for i in range(0, num_z.size(1)):
        x = num_labels
        y = num_z[:, i].detach().cpu()
        axs[i].scatter(x, y)
        axs[i].set_title(f'z{i + 1}')
        axs[i].set(xlabel='Num of Points on the card', xticks=range(0, 18))
        if eval_helper is not None:
            eval_helper.draw_scatter_point_line(axs[i], i, x, y)
            eval_helper.set_axis(axs[i], i)
        else:
            axs[i].grid(True)

    # for ax in axs.flat:
    #     ax.label_outer()
    plt.savefig(eval_path)
    plt.cla()
    plt.clf()
    plt.close()


def load_enc_eval_data(loader, encode_func):
    num_labels = []
    num_z = None
    for batch_ndx, sample in enumerate(loader):
        data, labels = sample
        data = data.to(DEVICE)
        num = [int(label.split('-')[0]) for label in labels]
        z = encode_func(data)
        num_labels.extend(num)
        if num_z is None:
            num_z = z
        else:
            num_z = torch.cat((num_z, z), dim=0)
    return num_z, num_labels


class MumEval:
    def __init__(self, config, is_train=True):
        self.config = config
        dataset = SingleImgDataset('../dataset/(1,16)-FixedPos-4Color')
        self.batch_size = config['batch_size']
        self.loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model = VQVAE(config).to(DEVICE)
        self.model.load_state_dict(self.model.load_tensor(MODEL_PATH))
        self.isVAE = config['kld_loss_scalar'] > 0.00001
        self.model.eval()

    def eval(self):
        os.makedirs(EVAL_PATH, exist_ok=True)
        num_z, num_labels = load_enc_eval_data(
                                    self.loader,
                                    lambda x: self.model.find_indices(
                                          self.model.batch_encode_to_z(x)[0], True
                                    )
        )
        eval_path = os.path.join(EVAL_PATH, f'z.png')
        eval_helper = EvalHelper(self.config)
        plot_z_against_label(num_z, num_labels, eval_path, eval_helper)

#
# if __name__ == "__main__":
#     evaler = MumEval(CONFIG)
#     evaler.eval()
