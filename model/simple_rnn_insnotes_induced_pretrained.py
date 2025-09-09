import sys
from os import path

sys.path.append(path.join(path.dirname(path.abspath(__file__)), "../../"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from model.simple_rnn_insnotes_induced import SymmCSAEwithSecondaryPrior
from utils.training_utils import compute_loss_weight


class SymmCSAEwithSecondaryPriorPretrained(SymmCSAEwithSecondaryPrior):
    def __init__(self, config):
        super().__init__(config)

        assert "ae_checkpoint" in config, "Please provide pretrained checkpoint path!"
        ae_checkpoint = config["ae_checkpoint"]
        save_info = torch.load(ae_checkpoint)
        self.load_state_dict(save_info["model"])

        self.init_prior()
        self.freeze_z()
        self.freeze_secondary_prior()

    def init_prior(self):
        # re-initialize prior
        config = self.config
        if "GRU" in config.keys() and config["GRU"]:
            self.prior = nn.RNN(
                input_size=self.d_zc,
                hidden_size=self.d_zc,
                num_layers=self.n_layers_rnn,
                batch_first=True,
            )
        else:
            self.prior = nn.RNN(
                input_size=self.d_zc,
                hidden_size=self.d_zc,
                num_layers=self.n_layers_rnn,
                batch_first=True,
            )
        for name, param in self.prior.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

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
