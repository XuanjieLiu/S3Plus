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


def gumbel_softmax(
    logits: torch.Tensor,
    tau: float = 1,
    hard: bool = False,
    gumbels: torch.Tensor = None,
    eps: float = 1e-10,
    dim: int = -1,
) -> torch.Tensor:
    r"""
    Sample from the Gumbel-Softmax distribution and optionally discretize.

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)
    """

    if gumbels is None:
        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )  # ~Gumbel(0,1)
    logits = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = logits.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).scatter_(dim, index, 1.0)
        y = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        y = y_soft
    return y, gumbels


class SymmCSAEwithTransition(SymmCSAEwithPrior):
    def __init__(self, config):
        super().__init__(config)

        n_atoms = config["n_atoms"]
        d_zc = config["d_zc"]

        # EBM: f_theta(z1, candidate)
        self.energy_net = nn.Sequential(
            nn.Linear(2 * d_zc, 256), nn.GELU(), nn.Linear(256, 1)
        )

    def compute_energies(self, z1, z_candidates):
        """
        z1: (B, N, d_zc), (B, d_zc) or (d_zc,)
        z_candidates: (n_atoms, d_zc)
        returns: (B, N, n_atoms), (B, n_atoms) or (n_atoms) energy scores (the higher, the more compatible)
        """
        if z1.dim() == 3:
            B = z1.shape[0]
            N = z1.shape[1]
            n_atoms = z_candidates.shape[0]

            z1_expand = z1.unsqueeze(2).expand(-1, -1, n_atoms, -1)
            z_cat = torch.cat(
                [
                    z1_expand,
                    z_candidates.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1),
                ],
                dim=-1,
            )
            energy = self.energy_net(z_cat).squeeze(-1)  # (B, N, n_atoms)
            return energy
        elif z1.dim() == 2:
            B = z1.shape[0]
            n_atoms = z_candidates.shape[0]

            z1_expand = z1.unsqueeze(1).expand(-1, n_atoms, -1)
            z_cat = torch.cat(
                [z1_expand, z_candidates.unsqueeze(0).expand(B, -1, -1)], dim=-1
            )
            energy = self.energy_net(z_cat).squeeze(-1)  # (B, n_atoms)
            return energy
        else:
            n_atoms = z_candidates.shape[0]

            z1_expand = z1.unsqueeze(0).expand(n_atoms, -1)
            z_cat = torch.cat([z1_expand, z_candidates], dim=-1)
            energy = self.energy_net(z_cat).squeeze(-1)  # (n_atoms,)
            return energy

    def transition_forward(self, z, tau=0.3, hard=True, gumbels=None):
        """
        Only used in inference step of the prior sampling.
        z: (B, N, d_zc)
        gumbels: (B, n_atoms, n_atoms), B sampling strategies as the transition matrix
        if gumbels is not None, use the provided gumbels for sampling
        returns:
            z_next: (B, N, d_zc) selected codebook vector
            probs: (B, N, n_atoms) softmax over energies
        """
        codebook = self.vq.codebook  # (n_atoms, d_zc)
        B = z.shape[0]
        n_atoms = codebook.shape[0]
        z, z_indices, _ = self.quantize(z, freeze_codebook=True)
        if gumbels is None:
            gumbels = (
                -torch.empty((B, n_atoms, n_atoms), device=z.device, dtype=z.dtype)
                .exponential_()
                .log()
            )  # ~Gumbel(0,1)
        scores = self.compute_energies(
            z, codebook
        )  # higher = more compatible. (B, N, n_atoms)
        chosen_gumbels = torch.gather(
            gumbels,
            dim=1,
            index=z_indices.unsqueeze(-1).expand(-1, -1, gumbels.size(2)),
        )
        probs, chosen_gumbels = gumbel_softmax(
            scores, tau=tau, hard=hard, gumbels=chosen_gumbels
        )  # (B, N, n_atoms). when hard=True, one-hot vectors
        z_next = torch.matmul(probs, codebook)  # (B, N, d_zc)
        return z_next, probs, scores, gumbels

    # def quantize_and_transit(self, z, temperature=1.0, hard=True):
    #     """
    #     Only used in inference step of the prior sampling. In training we use InfoNCE loss.
    #     z: (B, d_zc)
    #     returns: z_next: (B, d_zc)
    #     """
    #     z_vq, z_indices, _ = self.quantize(z, freeze_codebook=True)
    #     z_next, _, _ = self.transition_forward(z_vq, temperature=temperature, hard=hard, gum)
    #     return z_next
