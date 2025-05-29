import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.training_utils import compute_loss_weight, compute_loss_weights


class SymmLoss:
    def __init__(self, config, use_isymm=True):
        """
        config: a dict of loss config. Must contain key "weights".
        """
        super(SymmLoss, self).__init__()
        self.config = config
        self.use_isymm = use_isymm
        self.eps = 1e-5

    @staticmethod
    def sample_lengths(self, z):
        """
        z is a tensor of shape (B, N, D). We sample lengths on N.
        """
        N = z.shape[1]
        p1 = torch.randint(4, 6, size=())
        g1 = torch.randint(6, 8, size=())
        p2 = torch.randint(6, 8, size=())
        g2 = torch.randint(4, 6, size=())

        return p1, g1, p2, g2

    def compute_loss(self, step, model, x):
        """
        Do the forward here
        """
        # get the AE output
        x_hat, zc, zc_vq, indices, commit_loss, zs = model(x)  # vq loss and recon loss
        ae_loss = F.mse_loss(x_hat, x)

        # get the prior output
        ntf_ratio = compute_loss_weight(self.config["ntf_ratio"], step)
        zc_vq_hat = model.rnn_forward(zc_vq, ntf_ratio=ntf_ratio)
        prior_loss = F.mse_loss(zc_vq_hat, zc[:, 1:, :], reduction="mean")

        if step > self.config["start_isymm_at_n_steps"] and self.use_isymm:
            # prior symm output
            p1, g1, p2, g2 = self.sample_lengths(self, zc)

            global_prompt = zc[:, : max(p1, p2), :].clone()
            # 1 then 2
            zc_g1_tr = model.unroll(global_prompt[:, -p1:, :], g1)
            zc_p2_tr = torch.cat([global_prompt[:, -p1:, :], zc_g1_tr], dim=1)[
                :, -p2:, :
            ]
            zc_g2_tr = model.unroll(zc_p2_tr, g2)
            zc_tr = torch.cat([global_prompt, zc_g1_tr, zc_g2_tr], dim=1)

            # 2 then 1
            zc_g2_rt = model.unroll(global_prompt[:, -p2:, :], g2)
            zc_p1_rt = torch.cat([global_prompt[:, -p2:, :], zc_g2_rt], dim=1)[
                :, -p1:, :
            ]
            zc_g1_rt = model.unroll(zc_p1_rt, g1)
            zc_rt = torch.cat([global_prompt, zc_g2_rt, zc_g1_rt], dim=1)

            isymm_loss = F.mse_loss(zc_tr, zc_rt, reduction="mean")

        losses = {}
        total_loss = 0
        for k, v in self.config["weights"].items():
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
