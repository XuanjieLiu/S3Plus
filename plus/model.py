import sys
from os import path

sys.path.append(path.join(path.dirname(path.abspath(__file__)), '../../'))

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import gzip
from shared import *


class S3Plus(nn.Module):
    def __init__(self, config):
        super(S3Plus, self).__init__()
        self.latent_code_1 = config['latent_code_1']
        self.latent_code_2 = config['latent_code_2']
        self.latent_code_num = self.latent_code_1 + self.latent_code_2
        self.enc_dec_config = config['network_config']['enc_dec']
        self.plus_unit = config['network_config']['plus']['plus_unit']
        self.first_ch_num = self.enc_dec_config['first_ch_num']
        self.last_ch_num = self.first_ch_num * 4
        self.last_H = self.enc_dec_config['last_H']
        self.last_W = self.enc_dec_config['last_W']
        self.img_channel = self.enc_dec_config['img_channel']

        self.encoder = nn.Sequential(
            nn.Conv2d(self.img_channel, self.first_ch_num, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.first_ch_num, self.first_ch_num, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.first_ch_num, self.first_ch_num * 2, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.first_ch_num * 2, self.last_ch_num, kernel_size=(4, 4), stride=(2, 2), padding=1),
        )

        self.fc11 = nn.Linear(self.last_ch_num * self.last_H * self.last_W, self.latent_code_num)
        self.fc12 = nn.Linear(self.last_ch_num * self.last_H * self.last_W, self.latent_code_num)

        self.fc3 = nn.Sequential(
            nn.Linear(self.latent_code_num, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.last_ch_num * self.last_H * self.last_W)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.last_ch_num, self.first_ch_num * 2, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.first_ch_num * 2, self.first_ch_num, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.first_ch_num, self.first_ch_num, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.first_ch_num, self.img_channel, kernel_size=(4, 4), stride=(2, 2), padding=1),
        )

        self.plus_net = nn.Sequential(
            nn.Linear(self.latent_code_1 * 2, self.plus_unit),
            nn.ReLU(),
            nn.Linear(self.plus_unit, self.plus_unit),
            nn.ReLU(),
            nn.Linear(self.plus_unit, self.latent_code_1),
        )

    def plus(self, z_a, z_b):
        comb = torch.cat((z_a, z_b), dim=-1)
        return self.plus_net(comb)

    def minus(self, z_a, z_b):
        comb = torch.cat((z_a, z_b), dim=-1)
        return self.minus_net(comb)

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
        out3 = self.fc3(z).view(z.size(0), self.last_ch_num, self.last_H, self.last_W)
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
