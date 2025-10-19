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


class SymmCSAEwithTransition(SymmCSAEwithPrior):
    def __init__(self, config):
        super().__init__(config)
        self.transition_prior = nn.Parameter(
            torch.randn(config["n_atoms"], config["n_atoms"]), requires_grad=False
        ) # shape (n_atoms, n_atoms)

    def sample_transition(self, n, eps=1e-8):
        """
        transition_prior: (n_atoms, n_atoms)
        returns: (n, n_atoms,) the sampled next indices for each atom. Sample n times.
        """
        probs = F.relu(self.transition_prior)

        probs = probs / (probs.sum(dim=-1, keepdim=True) + eps)
        sampled_indices = torch.multinomial(probs, num_samples=n, replacement=True)

        return sampled_indices.reshape(n, -1)  # (n, n_atoms)

    def transit(self, z, transition):
        """
        z: input tensor of shape (B, d_zc)
        returns: output tensor of shape (B, d_zc)
        """
        z_vq, z_indices, _ = self.quantize(z, freeze_codebook=True)
        z_indices_next = transition[z_indices]  # (B,)
        z_vq_next = self.vq.codebook[z_indices_next]  # (B, d_zc)
        return z_vq_next