import sys
from os import path

sys.path.append(path.join(path.dirname(path.abspath(__file__)), '../../'))

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import gzip
from shared import *

# todo: make these parameters configurable
BATCH_SIZE = 32
log_interval = 10
IMG_CHANNEL = 3

LAST_H = 4
LAST_W = 4

FIRST_CH_NUM = 64
LAST_CN_NUM = FIRST_CH_NUM * 4


class S3Plus(nn.Module):
    def __init__(self, config):
        super(S3Plus, self).__init__()
        self.latent_code_1 = config['latent_code_1']
        self.latent_code_2 = config['latent_code_2']
        self.latent_code_num = self.latent_code_1 + self.latent_code_2

        self.encoder = nn.Sequential(
            nn.Conv2d(IMG_CHANNEL, FIRST_CH_NUM, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(FIRST_CH_NUM, FIRST_CH_NUM, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(FIRST_CH_NUM, FIRST_CH_NUM * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(FIRST_CH_NUM * 2, LAST_CN_NUM, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.fc11 = nn.Linear(LAST_CN_NUM * LAST_H * LAST_W, self.latent_code_num)
        self.fc12 = nn.Linear(LAST_CN_NUM * LAST_H * LAST_W, self.latent_code_num)

        self.fc3 = nn.Sequential(
            nn.Linear(self.latent_code_num, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, LAST_CN_NUM * LAST_H * LAST_W)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(LAST_CN_NUM, FIRST_CH_NUM * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(FIRST_CH_NUM * 2, FIRST_CH_NUM, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(FIRST_CH_NUM, FIRST_CH_NUM, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(FIRST_CH_NUM, IMG_CHANNEL, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def save_tensor(self, tensor, path):
        """保存 tensor 对象到文件"""
        torch.save(tensor, path)

    def load_tensor(self, path):
        """从文件读取 tensor 对象"""
        return torch.load(path)

    def reparameterize(self, mu, logvar):
        eps = Variable(torch.randn(mu.size(0), mu.size(1))).to(DEVICE)
        z = mu + eps * torch.exp(logvar) * 0.5
        return z

    def batch_decode_from_z(self, z):
        out3 = self.fc3(z).view(z.size(0), LAST_CN_NUM, LAST_H, LAST_W)
        frames = self.decoder(out3)
        return frames

    def batch_encode_to_z(self, x, is_VAE=True):
        out = self.encoder(x)
        mu = self.fc11(out.view(out.size(0), -1))
        logvar = self.fc12(out.view(out.size(0), -1))
        z1 = self.reparameterize(mu, logvar)
        if is_VAE:
            return z1, mu, logvar
        else:
            return mu, mu, mu
