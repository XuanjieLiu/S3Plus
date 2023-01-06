import sys
from os import path

sys.path.append(path.join(path.dirname(path.abspath(__file__)), '../../'))

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import gzip
from shared import *


class VectorQuantizer(nn.Module):
    """
    VQ-VAE layer: Input any tensor to be quantized.
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
          quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss terms (see
          equation 4 in the paper - this variable is Beta).
    """

    def __init__(self, embedding_dim, num_embeddings, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # initialize embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.mse_loss = nn.MSELoss()

    def forward(self, x):
        # [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1).contiguous()
        # [B, H, W, C] -> [BHW, C]
        flat_x = x.reshape(-1, self.embedding_dim)

        encoding_indices = self.get_code_indices(flat_x)
        quantized = self.quantize(encoding_indices)
        quantized = quantized.view_as(x)  # [B, H, W, C]

        if not self.training:
            quantized = quantized.permute(0, 3, 1, 2).contiguous()
            return quantized

        # embedding loss: move the embeddings towards the encoder's output
        q_latent_loss = self.mse_loss(quantized, x.detach())
        # commitment loss
        e_latent_loss = self.mse_loss(x, quantized.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = x + (quantized - x).detach()

        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return quantized, loss

    def get_code_indices(self, flat_x):
        # compute L2 distance
        distances = (
                torch.sum(flat_x ** 2, dim=1, keepdim=True) +
                torch.sum(self.embeddings.weight ** 2, dim=1) -
                2. * torch.matmul(flat_x, self.embeddings.weight.t())
        )  # [N, M]
        encoding_indices = torch.argmin(distances, dim=1)  # [N,]
        return encoding_indices

    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return self.embeddings(encoding_indices)


IMG_CHANNEL = 3
LAST_H = 4
LAST_W = 4

FIRST_CH_NUM = 64
LAST_CN_NUM = FIRST_CH_NUM * 4

OPERATOR_NET_NUM = 4

class Encoder(nn.Module):
    """Encoder of VQ-VAE"""
    def __init__(self, latent_code_num):
        super().__init__()
        self.latent_code_num = latent_code_num

        self.encoder = nn.Sequential(
            nn.Conv2d(IMG_CHANNEL, FIRST_CH_NUM, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(FIRST_CH_NUM, FIRST_CH_NUM, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(FIRST_CH_NUM, FIRST_CH_NUM * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(FIRST_CH_NUM * 2, self.latent_code_num, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    """Decoder of VQ-VAE"""
    def __init__(self, latent_code_num):
        super().__init__()
        self.latent_code_num = latent_code_num
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.latent_code_num, FIRST_CH_NUM * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(FIRST_CH_NUM * 2, FIRST_CH_NUM, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(FIRST_CH_NUM, FIRST_CH_NUM, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(FIRST_CH_NUM, IMG_CHANNEL, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z):
        return self.decoder(z)


class VQVAE(nn.Module):
    """VQ-VAE"""
    def __init__(self, config):
        super().__init__()
        self.latent_code_1 = config['latent_code_1']
        self.latent_code_2 = config['latent_code_2']
        self.embeddings_num = config['embeddings_num']
        self.commitment_scalar = config['commitment_scalar']
        self.latent_code_num = self.latent_code_1 + self.latent_code_2
        self.encoder = Encoder(self.latent_code_num)
        self.vq_layer = VectorQuantizer(self.latent_code_num, self.embeddings_num, self.commitment_scalar)
        self.decoder = Decoder(self.latent_code_num)
        self.mse_loss = nn.MSELoss(reduction='mean')

    def batch_encode_to_z(self, x):
        z = self.encoder(x)
        e, e_q_loss = self.vq_layer(z)
        return e, e_q_loss

    def batch_decode_from_z(self, e):
        x_recon = self.decoder(e)
        return x_recon


    def save_tensor(self, tensor, path):
        """保存 tensor 对象到文件"""
        torch.save(tensor, path)

    def load_tensor(self, path):
        """从文件读取 tensor 对象"""
        return torch.load(path)


    def forward(self, x):
        z = self.encoder(x)

        e, e_q_loss = self.vq_layer(z)
        x_recon = self.decoder(e)

        recon_loss = self.mse_loss(x_recon, x)

        return e_q_loss + recon_loss

