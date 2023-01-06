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

MODEL_PATH = 'checkpoint_8000.pt'
EVAL_PATH = f'num_eval-{MODEL_PATH.split(".")[0]}/'


def plot_z_against_label(num_z, num_labels):
    for i in range(0, num_z.size(1)):
        x = num_labels
        y = num_z[:, i].detach().cpu()
        plt.scatter(x, y)
        plt.xlabel('Num of Points')
        plt.ylabel('z value')
        plt.xticks(range(0, 18))
        fig_name = f'z_{i+1}.png'
        plt.savefig(os.path.join(EVAL_PATH, fig_name))
        plt.cla()
        plt.clf()
        plt.close()


class MumEval:
    def __init__(self, config, is_train=True):
        dataset = SingleImgDataset(config['train_data_path'])
        self.batch_size = config['batch_size']
        self.loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model = S3Plus(config).to(DEVICE)
        self.model.load_state_dict(self.model.load_tensor(MODEL_PATH))
        self.isVAE = config['kld_loss_scalar'] > 0.00001
        self.model.eval()

    def eval(self):
        os.makedirs(EVAL_PATH, exist_ok=True)
        num_z, num_labels = self.load_data()
        plot_z_against_label(num_z, num_labels)

    def load_data(self):
        num_labels = []
        num_z = None
        for batch_ndx, sample in enumerate(self.loader):
            data, labels = sample
            data = data.to(DEVICE)
            num = [int(label.split('-')[0]) for label in labels]
            z, mu, logvar = self.model.batch_encode_to_z(data, is_VAE=self.isVAE)
            num_labels.extend(num)
            if num_z is None:
                num_z = mu
            else:
                num_z = torch.cat((num_z, mu), dim=0)
        return num_z, num_labels

if __name__ == "__main__":
    evaler = MumEval(CONFIG)
    evaler.eval()
