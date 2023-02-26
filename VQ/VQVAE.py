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

    def __init__(self, embedding_dim, num_embeddings, commitment_cost, embedding_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embedding_cost = embedding_cost

        # initialize embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.mse_loss = nn.MSELoss()

    def forward(self, x):
        encoding_indices = self.get_code_flat_indices(x)
        quantized = self.quantize(encoding_indices)
        quantized = quantized.view_as(x)

        # embedding loss: move the embeddings towards the encoder's output
        q_latent_loss = self.mse_loss(quantized, x.detach()) * self.embedding_cost
        # commitment loss
        e_latent_loss = self.mse_loss(x, quantized.detach()) * self.commitment_cost
        loss = q_latent_loss + e_latent_loss

        # Straight Through Estimator
        quantized = x + (quantized - x).detach()

        return quantized, loss

    def get_code_flat_indices(self, x):
        flat_x = x.reshape(-1, self.embedding_dim)
        # compute L2 distance
        distances = (
                torch.sum(flat_x ** 2, dim=1, keepdim=True) +
                torch.sum(self.embeddings.weight ** 2, dim=1) -
                2. * torch.matmul(flat_x, self.embeddings.weight.t())
        )  # [N, M]
        encoding_indices = torch.argmin(distances, dim=1)  # [N,]
        return encoding_indices

    def get_code_indices(self, x):
        flat_indices = self.get_code_flat_indices(x)
        return flat_indices.reshape(x.size(0), -1)

    def flat_decimal_indices(self, indices):
        base = 1
        total = torch.zeros(indices.size(0)).to(DEVICE)
        for i in reversed(range(indices.size(-1))):
            total += indices[..., i] * base
            base *= self.num_embeddings
        return total.unsqueeze(-1)


    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return self.embeddings(encoding_indices)


class Encoder(nn.Module):
    """Encoder of VQ-VAE"""
    def __init__(self, latent_code_num, encoder_config):
        super().__init__()
        self.latent_code_num = latent_code_num
        self.img_channel = encoder_config['img_channel']
        self.first_ch_num = encoder_config['first_ch_num']
        self.last_ch_num = self.first_ch_num * 4
        self.last_H = encoder_config['last_H']
        self.last_W = encoder_config['last_W']
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

    def forward(self, x):
        out = self.encoder(x)
        return self.fc11(out.view(out.size(0), -1))


class Decoder(nn.Module):
    """Decoder of VQ-VAE"""
    def __init__(self, latent_code_num, decoder_config):
        super().__init__()
        self.latent_code_num = latent_code_num
        self.img_channel = decoder_config['img_channel']
        self.first_ch_num = decoder_config['first_ch_num']
        self.last_ch_num = self.first_ch_num * 4
        self.last_H = decoder_config['last_H']
        self.last_W = decoder_config['last_W']

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.last_ch_num, self.first_ch_num * 2, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.first_ch_num * 2, self.first_ch_num, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.first_ch_num, self.first_ch_num, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.first_ch_num, self.img_channel, kernel_size=(4, 4), stride=(2, 2), padding=1),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(self.latent_code_num, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.last_ch_num * self.last_H * self.last_W)
        )

    def forward(self, z):
        out3 = self.fc3(z).view(z.size(0), self.last_ch_num, self.last_H, self.last_W)
        frames = self.decoder(out3)
        return frames


class VQVAE(nn.Module):
    """VQ-VAE"""
    def __init__(self, config):
        super().__init__()
        self.embeddings_num = config['embeddings_num']
        self.embedding_dim = config['embedding_dim']
        self.isVQStyle = config['isVQStyle']
        self.latent_code_1 = config['latent_embedding_1'] * self.embedding_dim
        self.latent_code_2 = config['latent_embedding_2'] * self.embedding_dim if \
            self.isVQStyle else config['latent_code_2']
        self.commitment_scalar = config['commitment_scalar']
        self.embedding_scalar = config['embedding_scalar']
        self.latent_code_num = self.latent_code_1 + self.latent_code_2
        self.encoder = Encoder(self.latent_code_num, config['network_config']['enc_dec'])
        self.vq_layer = VectorQuantizer(self.embedding_dim, self.embeddings_num, self.commitment_scalar, self.embedding_scalar)
        self.decoder = Decoder(self.latent_code_num, config['network_config']['enc_dec'])
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.plus_unit = config['network_config']['plus']['plus_unit']
        self.plus_net = nn.Sequential(
            nn.Linear(self.latent_code_1 * 2, self.plus_unit),
            nn.ReLU(),
            nn.Linear(self.plus_unit, self.plus_unit),
            nn.ReLU(),
            nn.Linear(self.plus_unit, self.latent_code_1),
        )

    def plus(self, z_a, z_b):
        comb = torch.cat((z_a, z_b), dim=-1)
        z_plus = self.plus_net(comb)
        e_plus, e_q_loss = self.vq_layer(z_plus)
        return e_plus, e_q_loss, z_plus

    def batch_encode_to_z(self, x):
        z = self.encoder(x)
        if self.isVQStyle:
            e, e_q_loss = self.vq_layer(z)
            return e, e_q_loss, z
        else:
            z_c = z[..., 0:self.latent_code_1]
            z_s = z[..., self.latent_code_1:]
            e_c, e_q_loss = self.vq_layer(z_c)
            return torch.cat((e_c, z_s), -1), e_q_loss, z

    def batch_decode_from_z(self, e):
        x_recon = self.decoder(e)
        return x_recon

    def find_indices(self, z, need_split_style):
        if not need_split_style:
            return self.vq_layer.flat_decimal_indices(self.vq_layer.get_code_indices(z))
        z_c = z[..., 0:self.latent_code_1]
        z_s = z[..., self.latent_code_1:]
        z_c_idx = self.vq_layer.flat_decimal_indices(self.vq_layer.get_code_indices(z_c))
        if self.isVQStyle:
            z_s_idx = self.vq_layer.flat_decimal_indices(self.vq_layer.get_code_indices(z_s))
            return torch.cat((z_c_idx, z_s_idx), -1)
        else:
            return torch.cat((z_c_idx, z_s), -1)

    def save_tensor(self, tensor, path):
        """保存 tensor 对象到文件"""
        torch.save(tensor, path)

    def load_tensor(self, path):
        """从文件读取 tensor 对象"""
        return torch.load(path)


    def forward(self, x):
        print("forward not implemented")
        return 0

