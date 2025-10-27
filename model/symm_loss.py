import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.training_utils import compute_loss_weight, compute_loss_weights

# TODO: merge with model


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
        # if "length_sampling" in self.config.keys():
        #     if self.config["length_sampling"] == "short":
        #         p_t = torch.randint(3, 4, size=())
        #         g_t = torch.randint(5, 6, size=())
        #         p_r = torch.randint(5, 6, size=())
        #         g_r = torch.randint(3, 4, size=())
        #     elif self.config["length_sampling"] == "long":
        #         p_t = torch.randint(7, 9, size=())
        #         g_t = torch.randint(9, 11, size=())
        #         p_r = torch.randint(9, 11, size=())
        #         g_r = torch.randint(7, 9, size=())
        #     elif self.config["length_sampling"] == "mixed":
        #         p_t = torch.randint(3, 4, size=())
        #         g_t = torch.randint(9, 11, size=())
        #         p_r = torch.randint(9, 11, size=())
        #         g_r = torch.randint(3, 4, size=())
        p_t = torch.randint(2, 3, size=())
        g_t = torch.randint(1, 2, size=())
        p_r = torch.randint(7, 8, size=())
        g_r = torch.randint(7, 8, size=())

        return p_t, g_t, p_r, g_r

    def compute_loss(self, step, model, x):
        """
        Do the forward here
        """
        # get the AE output
        x_hat, zc, zc_vq, indices, commit_loss, zs = model(x)  # vq loss and recon loss
        ae_loss = F.mse_loss(x_hat, x)

        # get the prior output
        if isinstance(self.config["ntf_ratio"], str):
            ntf_ratio = compute_loss_weight(self.config["ntf_ratio"], step)
        else:
            ntf_ratio = self.config["ntf_ratio"]
        zc_vq_hat = model.rnn_forward(zc_vq, ntf_ratio=ntf_ratio)
        prior_loss = F.mse_loss(zc_vq_hat, zc[:, 1:, :], reduction="mean")

        if step > self.config["start_isymm_at_n_steps"] and self.use_isymm:
            # prior symm output
            p_t, g_t, p_r, g_r = self.sample_lengths(self, zc)

            # Old and probably wrong way to do it
            # global_prompt = zc[:, : max(p_t, p_r), :].clone()
            # # 1 then 2
            # zc_g_t_tr = model.unroll(global_prompt[:, -p_t:, :], g_t)
            # zc_p_r_tr = torch.cat([global_prompt[:, -p_t:, :], zc_g_t_tr], dim=1)[
            #     :, -p_r:, :
            # ]
            # zc_g_r_tr = model.unroll(zc_p_r_tr, g_r)
            # zc_tr = torch.cat([global_prompt, zc_g_t_tr, zc_g_r_tr], dim=1)

            # # 2 then 1
            # zc_g_r_rt = model.unroll(global_prompt[:, -p_r:, :], g_r)
            # zc_p_t_rt = torch.cat([global_prompt[:, -p_r:, :], zc_g_r_rt], dim=1)[
            #     :, -p_t:, :
            # ]
            # zc_g_t_rt = model.unroll(zc_p_t_rt, g_t)
            # zc_rt = torch.cat([global_prompt, zc_g_r_rt, zc_g_t_rt], dim=1)

            # New way to do it

            # Pick a random subsequence of zc of length p_r, with at least p_t tokens before it
            assert p_t <= zc.shape[1] - p_r + 1, (
                f"p_t ({p_t}) must be less than or equal to zc.shape[1] - p_r + 1 ({zc.shape[1] - p_r + 1})"
            )
            start_cursor = torch.randint(
                p_t, zc.shape[1] - p_r + 1, size=()
            ).item()  # TODO: use different cursor for each batch item
            zc_observed_with_p_t = zc[:, start_cursor - p_t : start_cursor + p_r, :]
            # T then R
            # T
            zc_observed_t = []
            for i in range(p_r):
                zc_observed_t.append(
                    model.unroll(zc_observed_with_p_t[:, i : p_t + i, :], g_t)[:, -1, :]
                )
            zc_observed_t = torch.stack(zc_observed_t, dim=1)  # (B, p_r, d_zc)
            # R:
            zc_observed_tr = model.unroll(zc_observed_t, g_r)

            # R then T
            # R
            zc_observed = zc_observed_with_p_t[:, -p_r:, :]
            zc_observed_r = model.unroll(zc_observed, g_r)  # (B, g_r, d_zc)
            zc_observed_r_with_p = torch.cat(
                [zc_observed_with_p_t, zc_observed_r], dim=1
            )
            zc_observed_rt = []
            for i in range(g_r):
                zc_observed_rt.append(
                    model.unroll(
                        zc_observed_r_with_p[:, -i - 1 - p_t : -i - 1, :], g_t
                    )[:, -1, :]
                )
            zc_observed_rt = torch.stack(zc_observed_rt, dim=1)

            isymm_loss = F.mse_loss(zc_observed_rt, zc_observed_tr, reduction="mean")

        losses = {}
        total_loss = 0
        for k, v in self.config["weights"].items():
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
