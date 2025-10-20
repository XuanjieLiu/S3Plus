import sys
from os import path
import random

sys.path.append(path.join(path.dirname(path.abspath(__file__)), "../../"))


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch.autograd import Variable
from vector_quantize_pytorch import VectorQuantize

from model.modules.insnotes import Encoder, Decoder

from model.simple_rnn_insnotes import SymmCSAEwithPrior


def gumbel_softmax(logits, temperature=1.0, hard=False, eps=1e-9):
    """Differentiable categorical sampling."""
    g = -torch.log(-torch.log(torch.rand_like(logits) + eps) + eps)
    y_soft = F.softmax((logits + g) / temperature, dim=-1)

    if hard:
        # straight-through: forward one-hot, backward soft gradient
        index = y_soft.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
        y = y_hard + (y_soft - y_hard).detach()
    else:
        y = y_soft
    return y


class SymmCSAEwithTransition(SymmCSAEwithPrior):
    def __init__(self, config):
        super().__init__(config)

        n_atoms = config["n_atoms"]
        d_zc = config["d_zc"]

        # EBM: f_theta(z1, candidate)
        self.energy_net = nn.Sequential(
            nn.Linear(2 * d_zc, 256), nn.GELU(), nn.Linear(256, 1)
        )

    def compute_energy(self, z1, z_candidates):
        """
        z1: (B, d_zc)
        z_candidates: (n_atoms, d_zc)
        returns: (B, n_atoms) energy scores (the higher, the more compatible)
        """
        B = z1.shape[0]
        n_atoms = z_candidates.shape[0]

        z1_expand = z1.unsqueeze(1).expand(-1, n_atoms, -1)
        z_cat = torch.cat(
            [z1_expand, z_candidates.unsqueeze(0).expand(B, -1, -1)], dim=-1
        )
        energy = self.energy_net(z_cat).squeeze(-1)  # (B, n_atoms)
        return energy

    def forward_transition(self, z, temperature=1.0, hard=True):
        """
        z: (B, d_zc)
        returns:
            z_next: (B, d_zc) selected codebook vector
            probs: (B, n_atoms) softmax over energies
        """
        codebook = self.vq.codebook  # (n_atoms, d_zc)
        scores = self.compute_energy(z, codebook)  # higher = more compatible
        probs = gumbel_softmax(
            scores, temperature=temperature, hard=hard
        )  # (B, n_atoms)
        z_next = torch.matmul(probs, codebook)  # (B, d_zc)
        return z_next, probs, scores

    def transit(self, z, temperature=1.0, hard=True):
        """
        z: (B, d_zc)
        returns: z_next: (B, d_zc)
        """
        z_vq, z_indices, _ = self.quantize(z, freeze_codebook=True)
        z_next, _, _ = self.forward_transition(z_vq, temperature=temperature, hard=hard)
        return z_next
