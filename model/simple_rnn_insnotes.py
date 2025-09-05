import sys
from os import path
import random

sys.path.append(path.join(path.dirname(path.abspath(__file__)), "../../"))


import numpy as np
import torch
import torch.nn as nn

# from torch.autograd import Variable
from vector_quantize_pytorch import VectorQuantize

from model.modules.insnotes import Encoder, Decoder


class SymmCSAEwithPrior(nn.Module):
    def __init__(self, config):
        super(SymmCSAEwithPrior, self).__init__()
        self.config = config
        self.d_zc = config["d_zc"]
        self.d_zs = config["d_zs"]
        self.n_layers_rnn = config["n_layers_rnn"]
        self.n_atoms = config["n_atoms"]
        # self.base_len = config["base_len"]

        if self.d_zs <= 0:
            self.d_zs = 0

        # hard-coded parameters
        self.encoder = Encoder(n_channels=128, W=128, H=32, d_emb=self.d_zc + self.d_zs)
        self.decoder = Decoder(n_channels=128, W=128, H=32, d_emb=self.d_zc + self.d_zs)

        self.vq = VectorQuantize(
            dim=self.d_zc,
            codebook_size=config["n_atoms"],
            commitment_weight=1,
            decay=config["vq_ema_decay"] if "vq_ema_decay" in config else 0.98,
            kmeans_init=True,
            ema_update=True,
            rotation_trick=True,
            threshold_ema_dead_code=config["threshold_ema_dead_code"]
            if "threshold_ema_dead_code" in config
            else 0,
        )
        self.codebook_norm = nn.BatchNorm1d(
            self.d_zc, affine=False, track_running_stats=False
        )  # purely for normalization

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

    def get_model_size(self):
        size = sum(p.numel() for p in self.parameters())
        return "Number of trainable parameters: {:.2f} M".format(size / 1024**2)

    def encode(self, x, quantize=True):
        """
        x: input tensor of shape (B, T, c, h, w)
        """
        b, t, c, h, w = x.size()
        x = x.reshape(b * t, c, h, w)
        z = self.encoder(x)
        z = z.reshape(b, t, z.size(1))
        if self.d_zs > 0:
            zc = z[:, :, : self.d_zc]
            zs = z[:, :, self.d_zc :]
            # pool this zs so that the sequence length is 1
            zs = zs.mean(dim=1, keepdim=True)
        else:
            zc = z
            zs = None
        if quantize:
            zc_vq, indices, commit_loss = self.quantize(zc)
            return zc_vq, indices, commit_loss, zs
        else:
            return zc, zs

    def quantize(self, x, freeze_codebook=False):
        if not freeze_codebook and self.training:
            self.vq.codebook = self.codebook_norm(self.vq.codebook)
        quantized, indices, commit_loss = self.vq(x, freeze_codebook=freeze_codebook)

        return quantized, indices, commit_loss

    def decode(self, zc, zs):
        if zs is not None:
            # expand zs to match the sequence length of zc
            zs = zs.expand(-1, zc.size(1), -1)
            z = torch.cat((zc, zs), dim=-1)
        else:
            z = zc
        out = self.decoder(z)
        return out

    def forward(self, x):
        """
        Basic forward without going through the prior
        """
        zc, zs = self.encode(x, quantize=False)
        zc_vq, indices, commit_loss = self.quantize(zc)
        out = self.decode(zc_vq, zs)
        return out, zc, zc_vq, indices, commit_loss, zs

    def rnn_forward(self, z, ntf_ratio=1.0):
        """
        z: input tensor of shape (B, T, d_zc), full sequence
        ntf_ratio: non teacher forcing ratio
        """
        B, T, D = z.size()
        h = torch.zeros(self.n_layers_rnn, B, self.d_zc, device=z.device)

        preds = []
        input_t = z[:, 0, :].unsqueeze(1)  # initial input: z_0

        for t in range(1, T):
            output, h = self.prior(input_t, h)  # output: (B, 1, D)
            # output = self.quantize(output, freeze_codebook=True)[0]
            pred_t = output[:, 0, :]  # (B, D)
            preds.append(pred_t)

            # decide next input: ground truth or prediction
            use_model = torch.rand(B, device=z.device) < ntf_ratio
            use_model = use_model.float().unsqueeze(1)  # (B, 1)

            gt_next = z[:, t, :]  # (B, D)
            next_input = use_model * pred_t + (1 - use_model) * gt_next
            input_t = next_input.unsqueeze(1)  # (B, 1, D)

        preds = torch.stack(preds, dim=1)  # (B, T-1, D)
        return preds

    def unroll(self, z, n_steps, z_future_gt=None):
        """
        Unroll the RNN for n_steps autoregressively.
        Return only the future predictions.

        The z should be the quantized z sequence (prompt) of shape (B, T, d)
        z_future_gt is the ground truth future z sequence of shape (B, n_steps, d) for teacher forcing.
        If z_future_gt is None, the model will predict the future z sequence.
        """
        h_0 = torch.zeros(self.n_layers_rnn, z.size(0), self.d_zc, device=z.device)
        output, h_n = self.prior(z, h_0)
        last_output = self.quantize(output[:, -1, :], freeze_codebook=True)[0]
        predictions = [last_output]
        for i in range(n_steps - 1):
            last_output = last_output.unsqueeze(1)
            next_output, h_n = self.prior(last_output, h_n)
            next_output = self.quantize(next_output, freeze_codebook=True)[0]
            last_output = next_output.squeeze(1)
            predictions.append(last_output)
        predictions = torch.stack(predictions, dim=1)

        return predictions

    """all following are unwanted"""

    # def batch_decode_from_z(self, z):
    #     out3 = self.fc3(z).view(z.size(0), CHANNELS[-1], LAST_H, LAST_W)
    #     frames = self.decoder(out3)
    #     return frames

    # def batch_encode_to_z(self, x):
    #     out = self.encoder(x)
    #     mu = self.fc11(out.view(out.size(0), -1))
    #     logvar = self.fc12(out.view(out.size(0), -1))
    #     z1 = self.reparameterize(mu, logvar)
    #     return z1, mu, logvar

    # def batch_seq_encode_to_z(self, x):
    #     img_in = x.contiguous().view(
    #         x.size(0) * x.size(1), x.size(2), x.size(3), x.size(4)
    #     )
    #     z1, mu, logvar = self.batch_encode_to_z(img_in)
    #     return [
    #         z1.view(x.size(0), x.size(1), z1.size(-1)),
    #         mu.view(x.size(0), x.size(1), z1.size(-1)),
    #         logvar.view(x.size(0), x.size(1), z1.size(-1)),
    #     ]

    # def batch_seq_decode_from_z(self, z):
    #     z_in = z.reshape(z.size(0) * z.size(1), z.size(2))
    #     recon = self.batch_decode_from_z(z_in)
    #     return recon.reshape(
    #         z.size(0), z.size(1), recon.size(-3), recon.size(-2), recon.size(-1)
    #     )

    # def do_rnn(self, z, hidden):
    #     out_r, hidden_rz = self.prior(z.unsqueeze(1), hidden)
    #     z2 = self.fc2(out_r.squeeze(1))
    #     return z2, hidden_rz

    # def predict_with_symmetry(self, z_gt, sample_points, symm_func, all_input_steps):
    #     z_SR_seq_batch = []
    #     hidden_r = torch.zeros(
    #         self.n_layers_rnn, z_gt.size(0), self.d_hidden_rnn, device=DEVICE
    #     )
    #     for i in range(all_input_steps):
    #         """Schedule sample"""
    #         if i in sample_points:
    #             z = z_gt[:, i]
    #             z_S = symm_func(z)
    #         else:
    #             z_S = z_SR_seq_batch[-1]
    #         z_SR, hidden_r = self.do_rnn(z_S, hidden_r)
    #         z_SR_seq_batch.append(z_SR)
    #     z_x0ESR = (
    #         torch.stack(z_SR_seq_batch, dim=0).permute(1, 0, 2).contiguous()[:, :-1, :]
    #     )
    #     return z_x0ESR

    # """Z Repeat"""

    # def recon_via_rnn(self, z):
    #     sample_points = list(range(z.size(1)))[: self.base_len]
    #     z_s = z[..., 0:1]
    #     z_c = z[..., 1:]
    #     z_cr = repeat_one_dim(z_c, sample_range=self.base_len)
    #     z_s1 = self.predict_with_symmetry(z_s, sample_points, lambda x: x, z.size(1))
    #     z_time_combine = torch.cat((z_s[:, 0:1, ...], z_s1), dim=1)
    #     z_code_combine = torch.cat((z_time_combine, z_cr), -1)
    #     return self.batch_seq_decode_from_z(z_code_combine), z_code_combine

    # def recon_via_rnn(self, z):
    #     sample_points = list(range(z.size(1)))[self.base_len:]
    #     z_1 = self.predict_with_symmetry(z, sample_points, lambda x: x)
    #     z_time_combine = torch.cat((z[:, 0:1, ...], z_1), dim=1)
    #     return self.batch_seq_decode_from_z(z_time_combine), z_time_combine


if __name__ == "__main__":
    # Test the SymmAE class
    config = {
        "d_zc": 128,
        "d_zs": 128,
        "n_layers_rnn": 2,
        "n_atoms": 256,
        # "base_len": 16,
        "vq_ema_decay": 0.98,
        "threshold_ema_dead_code": 0.1,
    }
    model = SymmCSAEwithPrior(config)
    print(model.get_model_size())

    # Create a random input tensor
    x = torch.randn(3, 10, 1, 128, 64)  # Batch size of 1, sequence length of 10
    # Test the forward pass
    x_hat, zc, zc_vq, indices, commit_loss, zs = model(x)
    print("Output shape:", x_hat.shape)
    print("Zc shape:", zc.shape)
    print("Zc VQ shape:", zc_vq.shape)
    print("Zs shape:", zs.shape)
    print("Indices shape:", indices.shape)
    print("Commit loss:", commit_loss)
    # Test the unroll method
    n_steps = 13
    # z_future_gt = torch.randn(1, n_steps, config["d_hidden_rnn"])  # Random future z
    zc_predictions = model.unroll(zc, n_steps, z_future_gt=None)
    print("Z_Predictions shape:", zc_predictions.shape)
    predictions = model.decode(zc_predictions, zs)
    print("Predictions shape:", predictions.shape)


# TODO: stochasticity
