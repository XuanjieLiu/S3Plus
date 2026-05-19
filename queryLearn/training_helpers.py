import torch

from shared import DEVICE
from VQ.VQVAE import split_into_three


def init_queries(query_config, query_dim):
    init_values = query_config.get('init_queries', None)
    if init_values is None:
        return torch.randn(2, query_dim, device=DEVICE)
    queries = torch.tensor(init_values, dtype=torch.float32, device=DEVICE)
    if queries.shape != (2, query_dim):
        raise ValueError(
            f"init_queries shape {tuple(queries.shape)} does not match expected {(2, query_dim)}"
        )
    return queries


def regul_sample(z_all_content):
    idx_1 = torch.randperm(z_all_content.size(0))
    z_perm = z_all_content[idx_1, ...]
    z_a, z_b, z_c = split_into_three(z_perm)
    return z_a, z_b, z_c


def sanity_check_oper_loss(per_loss_q1, per_loss_q2, label_a, label_b, label_c):
    is_add = torch.tensor(
        [a + b == c for a, b, c in zip(label_a, label_b, label_c)],
        device=per_loss_q1.device,
        dtype=torch.bool,
    )
    per_loss = torch.where(is_add, per_loss_q1, per_loss_q2)
    return per_loss.mean()
