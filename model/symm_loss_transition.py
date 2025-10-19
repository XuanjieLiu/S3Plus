import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.training_utils import compute_loss_weight, compute_loss_weights

from model.symm_loss import SymmLoss

# TODO: merge with model


class SymmLossTransition(SymmLoss):
    def __init__(self, config, use_isymm=True):
        """
        config: a dict of loss config. Must contain key "weights".
        """
        super(SymmLossTransition, self).__init__(config, use_isymm)

    @staticmethod
    def update_transition_prior(indices, n_atoms, normalize=True):
        """
        indices: (B, T) long tensor
        n_atoms: number of discrete states
        normalize: whether to row-normalize
        returns: (n_atoms, n_atoms) averaged transition matrix
        """
        B, T = indices.shape
        transition = torch.zeros(n_atoms, n_atoms, device=indices.device)
        
        for b in range(B):
            seq = indices[b]
            src = seq[:-1]
            dst = seq[1:]
            
            # count transitions in this batch
            mat = torch.zeros(n_atoms, n_atoms, device=indices.device)
            mat.index_add_(0, src, torch.nn.functional.one_hot(dst, n_atoms).float())
            
            transition += mat

        # average across batches
        transition /= B

        if normalize:
            transition = transition / (transition.sum(dim=-1, keepdim=True) + 1e-8)

        return transition

    def compute_loss(self, step, model, x):
        """
        Do the forward here
        """
        # get the AE output
        x_hat, zc, zc_vq, indices, commit_loss, zs = model(
            x
        )  # vq loss and recon loss
        ae_loss = F.mse_loss(x_hat, x)

        # get the prior output
        if isinstance(self.config["ntf_ratio"], str):
            ntf_ratio = compute_loss_weight(self.config["ntf_ratio"], step)
        else:
            ntf_ratio = self.config["ntf_ratio"]
        zc_vq_hat = model.rnn_forward(zc_vq, ntf_ratio=ntf_ratio)
        prior_loss = F.mse_loss(zc_vq_hat, zc[:, 1:, :], reduction="mean")

        # retrain transition prior
        self.transition_prior = SymmLossTransition.update_transition_prior(
            indices, model.vq.codebook.shape[0], normalize=True
        ).detach()

        # symmetry loss
        if step > self.config["start_isymm_at_n_steps"] and self.use_isymm:
            p_t, g_t, p_r, g_r = self.sample_lengths(self, zc)
            p_t = 1
            g_t = torch.randint(1, self.config["isymm_k"] + 1, size=())

            # New way to do it
            start_cursor = torch.randint(
                0, zc.shape[1] - p_r + 1, size=()
            ).item()  # TODO: use different cursor for each batch item
            zc_observed = zc[:, start_cursor : start_cursor + p_r, :]

            # T then R
            # T
            zc_observed_t = []
            transitions = self.sample_transition(n=zc_observed.shape[0])
            for i in range(zc_observed.shape[0]): # batch size
                zc_observed_t.append(self.transit(zc_observed[i], transitions[i]))
            zc_observed_t = torch.stack(zc_observed_t, dim=0)
            # R:
            zc_observed_tr = model.unroll(zc_observed_t, g_r)

            # R then T
            # R
            zc_observed_r = model.unroll(zc_observed, g_r)  # (B, g_r, d_zc)
            # T
            zc_observed_rt = []
            for i in range(zc_observed_r.shape[0]): # batch size
                zc_observed_rt.append(self.transit(zc_observed_r[i], transitions[i]))
            zc_observed_rt = torch.stack(zc_observed_rt, dim=0)

            isymm_loss = F.mse_loss(zc_observed_rt, zc_observed_tr, reduction="mean")
        model.unfreeze_secondary_prior()

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
