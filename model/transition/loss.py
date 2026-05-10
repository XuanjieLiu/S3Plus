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

    # def update_transition_prior(indices, n_atoms, normalize=True):
    #     """
    #     indices: (B, T) long tensor
    #     n_atoms: number of discrete states
    #     normalize: whether to row-normalize
    #     returns: (n_atoms, n_atoms) averaged transition matrix
    #     """
    #     B, T = indices.shape
    #     transition = torch.zeros(n_atoms, n_atoms, device=indices.device)

    #     for b in range(B):
    #         seq = indices[b]
    #         src = seq[:-1]
    #         dst = seq[1:]

    #         # count transitions in this batch
    #         mat = torch.zeros(n_atoms, n_atoms, device=indices.device)
    #         mat.index_add_(0, src, torch.nn.functional.one_hot(dst, n_atoms).float())

    #         transition += mat

    #     # average across batches
    #     transition /= B

    #     if normalize:
    #         transition = transition / (transition.sum(dim=-1, keepdim=True) + 1e-8)

    #     return transition

    def ebm_infonce_loss(self, model, z, z_idx, temperature=1.0):
        """
        z: (B, N, D), will be requantized to get indices
        z_idx: (B, N)
        return: scalar loss
        """
        B, N, D = z.shape

        anchors = z[:, :-1, :].reshape(B * (N - 1), D)
        next_idx = z_idx[:, 1:].reshape(
            B * (N - 1),
        )
        codebook = model.vq.codebook.detach()
        logits = model.compute_energies(anchors, codebook)  # (B*(N-1), n_atoms)
        logits = logits / temperature
        loss = F.cross_entropy(logits, next_idx, reduction="mean")

        return loss

    def train_energy_net(self, model, z, n_steps=1, lr=1e-4):
        """
        z: (B, N, D)
        returns: energy loss after training the energy net
        Gradient is only on energy net.
        """
        z = z.detach()  # stop gradient
        B, N, D = z.shape

        z, z_idx, _ = model.quantize(z.reshape(B * N, D), freeze_codebook=True)
        z = z.reshape(B, N, D)
        z_idx = z_idx.reshape(B, N)

        opt = torch.optim.AdamW(model.energy_net.parameters(), lr=lr)
        for _ in range(n_steps):
            opt.zero_grad()
            loss = self.ebm_infonce_loss(model, z, z_idx)
            loss.backward()
            opt.step()
        return loss

    def compute_loss(self, step, model, x, is_train=True):
        """
        Do the forward here
        """
        # get the AE output
        x_hat, zc, zc_vq, indices, commit_loss, zs = model(
            x, freeze_codebook=not is_train
        )  # vq loss and recon loss
        ae_loss = F.mse_loss(x_hat, x)

        # zc_vq, indices, _ = model.quantize(
        #     zc, freeze_codebook=True
        # ) # ensure using the updated codebook

        # get the prior output
        if isinstance(self.config["ntf_ratio"], str):
            ntf_ratio = compute_loss_weight(self.config["ntf_ratio"], step)
        else:
            ntf_ratio = self.config["ntf_ratio"]
        zc_vq_hat = model.rnn_forward(zc_vq, ntf_ratio=ntf_ratio)
        prior_loss = F.mse_loss(zc_vq_hat, zc[:, 1:, :], reduction="mean")

        # retrain transition prior
        if is_train:
            transition_energy_loss = self.train_energy_net(model, zc_vq, n_steps=10)
        else:
            # model.energy_net.train()
            # transition_energy_loss = self.train_energy_net(model, zc_vq, n_steps=10)
            # model.energy_net.eval()
            transition_energy_loss = self.ebm_infonce_loss(model, zc_vq, indices)

        # symmetry loss
        # important: freeze the transition model
        model.freeze_transition()
        if step > self.config["start_isymm_at_n_steps"] and self.use_isymm:
            p_t, g_t, p_r, g_r = self.sample_lengths(self, zc)
            p_t = 1
            g_t = torch.randint(1, self.config["isymm_k"] + 1, size=())

            # New way to do it
            start_cursor = torch.randint(
                0, zc.shape[1] - p_r + 1, size=()
            ).item()  # TODO: use different cursor for each batch item
            zc_observed = zc[:, start_cursor : start_cursor + p_r, :]  # (B, p_r, d_zc)

            # T then R
            # T
            zc_observed_t = []
            # probs = []
            # scores = []
            # gumbels = []
            # for i in range(zc_observed.shape[0]): # batch size
            #     zc_observed_this_next, p, s, g = model.transition_forward(
            #         zc_observed[i], gumbels=None
            #     )
            #     zc_observed_t.append(zc_observed_this_next)
            #     probs.append(p)
            #     scores.append(s)
            #     gumbels.append(g)
            # zc_observed_t = torch.stack(zc_observed_t, dim=0)
            schedule_tau = max(0.3, 1.0 * (0.9995**step))
            zc_observed_t, probs, scores, gumbels = model.transition_forward(
                zc_observed, gumbels=None, tau=schedule_tau
            )
            # R:
            zc_observed_tr = model.unroll(zc_observed_t, g_r)

            # R then T
            # R
            zc_observed_r = model.unroll(zc_observed, g_r)  # (B, g_r, d_zc)
            # T
            # zc_observed_rt = []
            # for i in range(zc_observed_r.shape[0]): # batch size
            #     zc_observed_r_this_next, _, _, _ = model.transition_forward(
            #         zc_observed_r[i], gumbels=gumbels[i]
            #     )
            #     zc_observed_rt.append(zc_observed_r_this_next)
            # zc_observed_rt = torch.stack(zc_observed_rt, dim=0)
            zc_observed_rt = model.transition_forward(
                zc_observed_r, gumbels=gumbels, tau=schedule_tau
            )[0]

            isymm_loss = F.mse_loss(zc_observed_rt, zc_observed_tr, reduction="mean")
        model.unfreeze_transition()

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
        losses["transition_energy_loss"] = transition_energy_loss

        if torch.isnan(total_loss):
            print(losses)
            raise ValueError("Loss is NaN!")
        if torch.isinf(total_loss):
            print(losses)
            raise ValueError("Loss is Inf!")

        return losses
