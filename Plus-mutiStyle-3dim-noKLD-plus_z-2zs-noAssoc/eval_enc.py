import os
from typing import List

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.optim as optim
from dataloader import SingleImgDataset
from loss_counter import LossCounter
from model import S3Plus
from shared import *
from train_config import CONFIG
import matplotlib
import matplotlib.markers
import matplotlib.pyplot as plt

matplotlib.use('AGG')

MODEL_PATH = 'curr_model.pt'
EVAL_PATH = f'num_eval-{MODEL_PATH.split(".")[0]}/'


def plot_z_against_label(num_z, num_labels, eval_path):
    z_dim = num_z.size(1)
    if z_dim == 1:
        fig, ax = plt.subplots(1, z_dim, sharey='all', figsize=(num_z.size(1) * 5, 5))
        axs = [ax]
    else:
        fig, axs = plt.subplots(1, z_dim, sharey='all', figsize=(num_z.size(1) * 5, 5))
    for i in range(0, num_z.size(1)):
        x = num_labels
        y = num_z[:, i].detach().cpu()
        axs[i].scatter(x, y)
        axs[i].set_title(f'z{i + 1}')
        axs[i].grid(True)
        # axs[i].xticks(range(0, 18))
        axs[i].set(ylabel='z value', xlabel='Num of Points on the card', xticks=range(0, max(num_labels)+1))
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
        z, mu, logvar = encode_func(data)
        num_labels.extend(num)
        if num_z is None:
            num_z = mu
        else:
            num_z = torch.cat((num_z, mu), dim=0)
    return num_z, num_labels





class MumEval:
    def __init__(self, config, is_train=True):
        dataset = SingleImgDataset('../dataset/(1,16)-FixedPos-4Color')
        self.batch_size = config['batch_size']
        self.loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model = S3Plus(config).to(DEVICE)
        self.model.load_state_dict(self.model.load_tensor(MODEL_PATH))
        self.isVAE = config['kld_loss_scalar'] > 0.00001
        self.model.eval()

    def eval(self):
        os.makedirs(EVAL_PATH, exist_ok=True)
        num_z, num_labels = load_enc_eval_data(self.loader,
                                               lambda x: self.model.batch_encode_to_z(x, is_VAE=self.isVAE))
        eval_path = os.path.join(EVAL_PATH, f'z.png')
        plot_z_against_label(num_z, num_labels, eval_path)


if __name__ == "__main__":
    evaler = MumEval(CONFIG)
    evaler.eval()
