import torch
import torch.nn as nn

from VQ.VQVAE import MultiVectorQuantizer


def comb_q_z(ea, eb, q):
    if q.dim() == 1:
        q = q.unsqueeze(0).expand(ea.size(0), -1)
    return torch.cat([ea, eb, q], dim=-1)


class OperNet(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            n_hidden_layers,
            unit,
            vq_layer: MultiVectorQuantizer,
            train_vq: bool = False,
            condition_mode: str = 'concat',
            pair_dim: int = None,
            query_dim: int = None,
            film_init_identity: bool = True):
        super().__init__()
        self.condition_mode = condition_mode.lower()
        self.pair_dim = pair_dim
        self.query_dim = query_dim
        self.n_hidden_layers = n_hidden_layers
        if self.condition_mode not in {'concat', 'film'}:
            raise ValueError(f"Unsupported operator condition_mode: {condition_mode}")
        if n_hidden_layers < 1:
            raise ValueError("OperNet expects at least one hidden layer")

        if self.condition_mode == 'concat':
            layers = [nn.Linear(in_dim, unit), nn.ReLU()]
            for _ in range(n_hidden_layers - 1):
                layers.extend([nn.Linear(unit, unit), nn.ReLU()])
            layers.append(nn.Linear(unit, out_dim))
            self.net = nn.Sequential(*layers)
        else:
            if pair_dim is None or query_dim is None:
                raise ValueError("FiLM OperNet requires pair_dim and query_dim")
            self.input_layer = nn.Linear(pair_dim, unit)
            self.hidden_layers = nn.ModuleList([
                nn.Linear(unit, unit) for _ in range(n_hidden_layers - 1)
            ])
            self.film_layers = nn.ModuleList([
                nn.Linear(query_dim, unit * 2) for _ in range(n_hidden_layers)
            ])
            self.output_layer = nn.Linear(unit, out_dim)
            if film_init_identity:
                for layer in self.film_layers:
                    nn.init.zeros_(layer.weight)
                    nn.init.zeros_(layer.bias)

        self.vq_layer = vq_layer
        self.train_vq = train_vq
        self._set_vq_trainable(train_vq)

    def _set_vq_trainable(self, train_vq: bool):
        self.train_vq = train_vq
        for p in self.vq_layer.parameters():
            p.requires_grad = train_vq

    def _film(self, h, q, layer_idx):
        gamma, beta = self.film_layers[layer_idx](q).chunk(2, dim=-1)
        return h * (1.0 + gamma) + beta

    def _film_forward(self, x):
        z_pair = x[..., :self.pair_dim]
        q = x[..., self.pair_dim:]
        h = self.input_layer(z_pair)
        h = torch.relu(self._film(h, q, 0))
        for idx, layer in enumerate(self.hidden_layers, start=1):
            h = layer(h)
            h = torch.relu(self._film(h, q, idx))
        return self.output_layer(h)

    def forward(self, x):
        if self.condition_mode == 'concat':
            z_oper = self.net(x)
        else:
            z_oper = self._film_forward(x)
        e_oper, e_q_loss = self.vq_layer(z_oper)
        return e_oper, e_q_loss, z_oper

    def train(self, mode: bool = True):
        super().train(mode)
        if self.train_vq:
            self.vq_layer.train(mode)
        else:
            self.vq_layer.eval()
        return self
