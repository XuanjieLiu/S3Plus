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

from model.simple_rnn_insnotes import SymmCSAEwithPrior


class SymmCSAEwithSecondaryPrior(SymmCSAEwithPrior):
    def __init__(self, config):
        super().__init__(config)
        self.n_layers_secondary_rnn = 1  # secondary RNN has only 1 layer
        self.secondary_prior = nn.RNN(
            input_size=self.d_zc,
            hidden_size=self.d_zc,
            num_layers=1,
            batch_first=True,
        )

    def secondary_rnn_forward(self, z, ntf_ratio=1.0):
        """
        z: input tensor of shape (B, T, d_zc), full sequence
        ntf_ratio: non teacher forcing ratio
        """
        B, T, D = z.size()
        h = torch.zeros(self.n_layers_secondary_rnn, B, self.d_zc, device=z.device)

        preds = []
        input_t = z[:, 0, :].unsqueeze(1)  # initial input: z_0

        for t in range(1, T):
            output, h = self.secondary_prior(input_t, h)  # output: (B, 1, D)
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

    def secondary_unroll(self, z, n_steps, z_future_gt=None):
        """
        Unroll the RNN for n_steps autoregressively.
        Return only the future predictions.

        The z should be the quantized z sequence (prompt) of shape (B, T, d)
        z_future_gt is the ground truth future z sequence of shape (B, n_steps, d) for teacher forcing.
        If z_future_gt is None, the model will predict the future z sequence.
        """
        h_0 = torch.zeros(
            self.n_layers_secondary_rnn, z.size(0), self.d_zc, device=z.device
        )
        output, h_n = self.secondary_prior(z, h_0)
        last_output = self.quantize(output[:, -1, :], freeze_codebook=True)[0]
        predictions = [last_output]
        for i in range(n_steps - 1):
            last_output = last_output.unsqueeze(1)
            next_output, h_n = self.secondary_prior(last_output, h_n)
            next_output = self.quantize(next_output, freeze_codebook=True)[0]
            last_output = next_output.squeeze(1)
            predictions.append(last_output)
        predictions = torch.stack(predictions, dim=1)

        return predictions

    def freeze_secondary_prior(self):
        for param in self.secondary_prior.parameters():
            param.requires_grad = False

    def unfreeze_secondary_prior(self):
        for param in self.secondary_prior.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    # Test the SymmAE class
    config = {
        "d_zc": 128,
        "d_zs": 0,
        "n_layers_rnn": 2,
        "n_atoms": 256,
        # "base_len": 16,
        "vq_ema_decay": 0.98,
        "threshold_ema_dead_code": 0.1,
    }
    model = SymmCSAEwithSecondaryPrior(config)
    print(model.get_model_size())

    # Create a random input tensor
    x = torch.randn(3, 10, 1, 128, 32)  # Batch size of 1, sequence length of 10
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

    # Test the secondary RNN
    zc_secondary_predictions = model.secondary_unroll(zc, n_steps)
    print("Secondary Predictions shape:", zc_secondary_predictions.shape)
    secondary_predictions = model.decode(zc_secondary_predictions, zs)
    print("Secondary Predictions shape:", secondary_predictions.shape)
