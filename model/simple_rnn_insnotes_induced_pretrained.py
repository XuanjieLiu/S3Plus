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
from model.simple_rnn_insnotes_induced import SymmCSAEwithSecondaryPrior


class SymmCSAEwithSecondaryPriorPretrained(SymmCSAEwithSecondaryPrior):
    def __init__(self, config):
        super().__init__(config)
        self.freeze_z()
        self.freeze_secondary_prior()

    def freeze_z(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.vq.parameters():
            param.requires_grad = False

    def compute_loss(self, step, loss_config, x):
        # get the AE output
        x_hat, zc, zc_vq, indices, commit_loss, zs = self.forward(x)
        commit_loss = commit_loss * 0.0  # do not train VQ

        # get the prior output
        if isinstance(loss_config["ntf_ratio"], str):
            ntf_ratio = compute_loss_weight(loss_config["ntf_ratio"], step)
        else:
            ntf_ratio = loss_config["ntf_ratio"]
        zc_vq_hat = self.rnn_forward(zc_vq, ntf_ratio=ntf_ratio)
        prior_loss = F.mse_loss(zc_vq_hat, zc[:, 1:, :], reduction="mean")

        losses = {}
        total_loss = 0
        for k, v in loss_config["weights"].items():
            if isinstance(v, str):
                v = compute_loss_weight(v, step)
            if locals().get(k) is None:
                continue
            total_loss = total_loss + v * locals()[k]
            losses[k] = locals()[k]
        assert isinstance(total_loss, torch.Tensor)
        losses["total_loss"] = total_loss

        if torch.isnan(total_loss):
            print(losses)
            raise ValueError("Loss is NaN!")
        if torch.isinf(total_loss):
            print(losses)
            raise ValueError("Loss is Inf!")

        return losses
