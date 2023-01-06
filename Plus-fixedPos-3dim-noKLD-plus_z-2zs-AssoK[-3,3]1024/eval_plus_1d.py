import os
from typing import List

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.optim as optim
from dataloader_plus import Dataset
from loss_counter import LossCounter
from model import S3Plus
from shared import *
from train_config import CONFIG
import matplotlib
import matplotlib.markers
import matplotlib.pyplot as plt

matplotlib.use('AGG')
MODEL_PATH = 'checkpoint_30000.pt'
EVAL_ROOT = 'eval_z'


class EncZ:
    def __init__(self, label, z):
        self.label = label
        self.z = z


class PlusZ:
    def __init__(self, label_a, label_b, plus_c_z):
        self.plus_c_z = plus_c_z
        self.label_c = label_a + label_b


def parse_label(label):
    return int(label.split('.')[0].split('-')[1])


def load_z(loader, encode_func, plus_func, zc_dim):
    all_enc_z = []
    all_plus_z = []
    for batch_ndx, sample in enumerate(loader):
        enc_z_list = []
        plus_z_list = []
        data, labels = sample
        _, za, _ = encode_func(data[0])
        _, zb, _ = encode_func(data[1])
        _, zc, _ = encode_func(data[2])
        za = za[..., 0:zc_dim]
        zb = zb[..., 0:zc_dim]
        zc = zc[..., 0:zc_dim]
        label_a = [parse_label(x) for x in labels[0]]
        label_b = [parse_label(x) for x in labels[1]]
        label_c = [parse_label(x) for x in labels[2]]
        plus_c = plus_func(za, zb)
        for i in range(0, za.size(0)):
            enc_z_list.append(EncZ(label_a[i], za[i]))
            enc_z_list.append(EncZ(label_b[i], zb[i]))
            enc_z_list.append(EncZ(label_c[i], zc[i]))
            plus_z_list.append(PlusZ(label_a[i], label_b[i], plus_c[i]))
        all_enc_z.extend(enc_z_list)
        all_plus_z.extend(plus_z_list)
    return all_enc_z, all_plus_z


def plot_z_against_label(all_enc_z, all_plus_z, eval_path):
    dim_z = all_enc_z[0].z.size(0)
    if dim_z == 1:
        axs = []
        fig, ax = plt.subplots(1, dim_z, sharey='all', figsize=(dim_z * 5, 5))
        axs.append(ax)
    else:
        fig, axs = plt.subplots(1, dim_z, sharey='all', figsize=(dim_z * 5, 5))
    enc_x = [ob.label for ob in all_enc_z]
    plus_x = [ob.label_c for ob in all_plus_z]
    for i in range(0, dim_z):
        enc_y = [ob.z.cpu()[i].item() for ob in all_enc_z]
        plus_y = [ob.plus_c_z.cpu()[i].item() for ob in all_plus_z]
        axs[i].scatter(enc_x, enc_y, edgecolors='blue', label='z by encoder', facecolors='none')
        axs[i].scatter(plus_x, plus_y, edgecolors='red', label='z by plus', facecolors='none')
    #for ax in axs.flat:
        axs[i].set(ylabel='z value', xlabel='Num of Points on the card', xticks=range(0, 18))
        axs[i].grid()
    # for ax in axs.flat:
    #     ax.label_outer()
    plt.legend()
    plt.savefig(eval_path)
    plt.cla()
    plt.clf()
    plt.close()


class PlusEval:
    def __init__(self, config, is_train=True):
        dataset = Dataset(config['train_data_path'])
        self.batch_size = config['batch_size']
        self.loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model = S3Plus(config).to(DEVICE)
        self.model.load_state_dict(self.model.load_tensor(MODEL_PATH))
        self.isVAE = config['kld_loss_scalar'] > 0.00001
        self.latent_code_1 = config['latent_code_1']
        self.model.eval()

    def eval(self):
        all_enc_z, all_plus_z = load_z(
            self.loader,
            lambda x: self.model.batch_encode_to_z(x, is_VAE=self.isVAE),
            self.model.plus,
            self.latent_code_1
        )
        eval_path = os.path.join(EVAL_ROOT, f'plus_eval_{MODEL_PATH}.png')
        plot_z_against_label(all_enc_z, all_plus_z, eval_path)
        print('loaded')


if __name__ == "__main__":
    os.makedirs(EVAL_ROOT, exist_ok=True)
    evaler = PlusEval(CONFIG)
    evaler.eval()