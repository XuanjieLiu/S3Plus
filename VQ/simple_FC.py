import torch
import torch.nn as nn
from VQVAE import make_multi_layers



class SimpleFC(nn.Module):
    def __init__(self, fc_net_config, input_dim, output_dim):
        super().__init__()
        n_fc_layers = fc_net_config['n_hidden_layers']
        plus_unit = fc_net_config['plus_unit']
        plus_hiddens = make_multi_layers(
            [nn.Linear(plus_unit, plus_unit),
             nn.ReLU()],
            n_fc_layers - 1
        )
        self.classify_fc_net = nn.Sequential(
            nn.Linear(input_dim, plus_unit),
            nn.ReLU(),
            *plus_hiddens,
            nn.Linear(plus_unit, output_dim),
        ) if n_fc_layers > 0 else nn.Linear(input_dim, output_dim)

        self.recon_fc_net = nn.Sequential(
            nn.Linear(input_dim, plus_unit),
            nn.ReLU(),
            *plus_hiddens,
            nn.Linear(plus_unit, int(input_dim/2)),
        ) if n_fc_layers > 0 else nn.Linear(input_dim, int(input_dim/2))

    def classify_composition(self, z_a, z_b):
        comb = torch.cat((z_a, z_b), dim=-1)
        z_comp = self.classify_fc_net(comb)
        return z_comp

    def recon_composition(self, z_a, z_b):
        comb = torch.cat((z_a, z_b), dim=-1)
        z_comp = self.recon_fc_net(comb)
        return z_comp