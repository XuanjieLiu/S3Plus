import sys
from os import path
from typing import List
sys.path.append(path.join(path.dirname(path.abspath(__file__)), '../../'))
sys.path.append(path.join(path.dirname(path.abspath(__file__)), '../'))
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
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
        flat_quantized = self.quantize_flat(encoding_indices)
        quantized = flat_quantized.view_as(x)

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


    def quantize_flat(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return self.embeddings(encoding_indices)


class MultiVectorQuantizer(nn.Module):
    """
    VQ-VAE layer: Input any tensor to be quantized.
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
          quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss terms (see
          equation 4 in the paper - this variable is Beta).
    """

    def __init__(self, embedding_dim, num_embeddings, commitment_cost, embedding_cost, multi_num_embeddings=None, init_embs: torch.Tensor=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embedding_cost = embedding_cost
        self.is_multiVQ = multi_num_embeddings is not None
        self.num_embedding_list = multi_num_embeddings
        # initialize embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim).to(DEVICE)
        if init_embs is not None:
            assert init_embs.shape == self.embeddings.weight.shape, "init_embs shape not match"
            self.embeddings.weight.data = torch.tensor(init_embs, dtype=torch.float)
        if self.is_multiVQ:
            self.codebooks = [nn.Embedding(n, self.embedding_dim).to(DEVICE) for n in self.num_embedding_list]
        self.mse_loss = nn.MSELoss()

    def forward(self, x):
        # encoding_indices = self.get_code_flat_indices(x)
        # flat_quantized = self.quantize_flat(encoding_indices)
        # quantized = flat_quantized.view_as(x)

        indices = self.get_code_indices(x)
        quantized = self.quantize(indices)
        # embedding loss: move the embeddings towards the encoder's output
        q_latent_loss = self.mse_loss(quantized, x.detach()) * self.embedding_cost
        # commitment loss
        e_latent_loss = self.mse_loss(x, quantized.detach()) * self.commitment_cost
        loss = q_latent_loss + e_latent_loss

        # Straight Through Estimator
        quantized = x + (quantized - x).detach()

        return quantized, loss

    def get_code_flat_indices(self, x, codebook_idx=-1):
        if codebook_idx == -1:
            codebook = self.embeddings
        else:
            codebook = self.codebooks[codebook_idx]
        flat_x = x.reshape(-1, self.embedding_dim)
        # compute L2 distance
        distances = (
                torch.sum(flat_x ** 2, dim=1, keepdim=True) +
                torch.sum(codebook.weight ** 2, dim=1) -
                2. * torch.matmul(flat_x, codebook.weight.t())
        )  # [N, M]
        encoding_indices = torch.argmin(distances, dim=1)  # [N,]
        return encoding_indices

    def get_code_indices(self, x):
        if self.is_multiVQ:
            indices_list = []
            for i in range(len(self.num_embedding_list)):
                x_slice = x[..., i*self.embedding_dim:(i+1)*self.embedding_dim]
                indices = self.get_code_flat_indices(x_slice, i)
                indices_list.append(indices)
            return torch.stack(indices_list, dim=1)
        else:
            flat_indices = self.get_code_flat_indices(x)
            return flat_indices.reshape(x.size(0), -1)

    def quantize_flat(self, encoding_indices, codebook_idx=-1):
        if codebook_idx == -1:
            codebook = self.embeddings
        else:
            codebook = self.codebooks[codebook_idx]
        """Returns embedding tensor for a batch of indices."""
        return codebook(encoding_indices)

    def quantize(self, encoding_indices):
        if self.is_multiVQ:
            emb_list = []
            for i in range(len(self.num_embedding_list)):
                quantized = self.quantize_flat(encoding_indices[..., i], i)
                emb_list.append(quantized)
            return torch.cat(emb_list, dim=-1)
        else:
            batch_size = encoding_indices.size(0)
            total_dim = encoding_indices.size(1) * self.embedding_dim
            flat_indices = encoding_indices.reshape(-1)
            flat_quantize = self.quantize_flat(flat_indices)
            return flat_quantize.reshape(batch_size, total_dim)



class Encoder(nn.Module):
    """Encoder of VQ-VAE"""
    def __init__(self, latent_code_num, encoder_config, enc_fc_config):
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
        fc_n_input = self.last_ch_num * self.last_H * self.last_W
        fc_n_units = enc_fc_config['n_units']
        fc_n_hidden_layers = enc_fc_config['n_hidden_layers']
        fc_hiddens = make_multi_layers(
            [nn.Linear(fc_n_units, fc_n_units),
             nn.ReLU()],
            fc_n_hidden_layers - 1
        )
        self.fc11 = nn.Sequential(
            nn.Linear(fc_n_input, fc_n_units),
            nn.ReLU(),
            *fc_hiddens,
            nn.Linear(fc_n_units, self.latent_code_num),
        ) if fc_n_hidden_layers > 0 else nn.Linear(fc_n_input, self.latent_code_num)

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


def make_multi_layers(layers_unit, num):
    layers = []
    for i in range(num):
        layers.extend(layers_unit)
    return layers


class VQVAE(nn.Module):
    """VQ-VAE"""
    def __init__(self, config):
        super().__init__()
        self.embeddings_num = config['embeddings_num']
        self.embedding_dim = config['embedding_dim']
        self.isVQStyle = config['isVQStyle']
        self.multi_num_embeddings = config['multi_num_embeddings']
        if self.multi_num_embeddings is None:
            self.latent_embedding_1 = config['latent_embedding_1']
        else:
            self.latent_embedding_1 = len(self.multi_num_embeddings)
        self.latent_code_1 = self.latent_embedding_1 * self.embedding_dim
        self.latent_code_2 = config['latent_embedding_2'] * self.embedding_dim if \
            self.isVQStyle else config['latent_code_2']
        self.commitment_scalar = config['commitment_scalar']
        self.embedding_scalar = config['embedding_scalar']
        self.latent_code_num = self.latent_code_1 + self.latent_code_2
        self.encoder = Encoder(self.latent_code_num, config['network_config']['enc_dec'], config['network_config']['enc_fc'])
        self.vq_layer = MultiVectorQuantizer(
            self.embedding_dim,
            self.embeddings_num,
            self.commitment_scalar,
            self.embedding_scalar,
            self.multi_num_embeddings)
        self.decoder = Decoder(self.latent_code_num, config['network_config']['enc_dec'])
        self.mse_loss = nn.MSELoss(reduction='mean')
        plus_unit = config['network_config']['plus']['plus_unit']
        n_plus_layers = config['network_config']['plus']['n_hidden_layers']
        plus_hiddens = make_multi_layers(
            [nn.Linear(plus_unit, plus_unit),
            nn.ReLU()],
            n_plus_layers - 1
        )
        self.plus_net = nn.Sequential(
            nn.Linear(self.latent_code_1 * 2, plus_unit),
            nn.ReLU(),
            *plus_hiddens,
            nn.Linear(plus_unit, self.latent_code_1),
        ) if n_plus_layers > 0 else nn.Linear(self.latent_code_1 * 2, self.latent_code_1)

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

    def flat_decimal_indices(self, indices):
        base = 1
        total = torch.zeros(indices.size(0)).to(DEVICE)
        if self.multi_num_embeddings is not None:
            times = self.multi_num_embeddings
        else:
            times = [self.embeddings_num for n in range(self.latent_embedding_1)]
        for i in reversed(range(indices.size(-1))):
            total += indices[..., i] * base
            base *= times[i]
        return total.unsqueeze(-1)

    def find_indices(self, z, input_has_z_s=False, output_cat_z_s=False):
        if not input_has_z_s:
            return self.flat_decimal_indices(self.vq_layer.get_code_indices(z))
        z_c = z[..., 0:self.latent_code_1]
        z_s = z[..., self.latent_code_1:]
        z_c_idx = self.flat_decimal_indices(self.vq_layer.get_code_indices(z_c))
        if not output_cat_z_s:
            return z_c_idx
        if self.isVQStyle:
            z_s_idx = self.flat_decimal_indices(self.vq_layer.get_code_indices(z_s))
            return torch.cat((z_c_idx, z_s_idx), -1)
        else:
            return torch.cat((z_c_idx, z_s), -1)

    def save_tensor(self, tensor, path):
        """保存 tensor 对象到文件"""
        torch.save(tensor, path)

    def load_tensor(self, path):
        """从文件读取 tensor 对象"""
        return torch.load(path, map_location=torch.device(DEVICE), weights_only=False)

    def index2z(self, idx):
        if self.multi_num_embeddings is not None:
            print("error. multi_num_embeddings not implemented")
            exit(0)

    def forward(self, x):
        sizes = x[0].size()
        data_all = torch.stack(x, dim=0).reshape(3 * sizes[0], sizes[1], sizes[2], sizes[3])
        e_all, e_q_loss, z_all = self.batch_encode_to_z(data_all)
        recon = self.batch_decode_from_z(e_all)
        recon_a, recon_b, recon_c = split_into_three(recon)
        e_a, e_b, e_c = split_into_three(e_all)
        z_s = e_a[..., self.latent_code_1:]
        ea_content = e_a[..., 0:self.latent_code_1]
        eb_content = e_b[..., 0:self.latent_code_1]
        e_ab_content, e_q_loss, z_ab_content = self.plus(ea_content, eb_content)
        e_ab = torch.cat((e_ab_content, z_s), -1)
        recon_ab = self.batch_decode_from_z(e_ab)
        return recon_a, recon_b, recon_c, recon_ab, e_a, e_b, e_c, e_ab


def split_into_three(tensor):
    sizes = [3, int(tensor.size(0) / 3), *tensor.size()[1:]]
    new_tensor = tensor.reshape(*sizes)
    return new_tensor[0], new_tensor[1], new_tensor[2]

