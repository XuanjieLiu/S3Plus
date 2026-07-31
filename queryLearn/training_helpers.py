import math

import torch

from shared import DEVICE
from VQ.VQVAE import split_into_three


OPERATION_LOSS_HARD_MIN = 'hard_min'
OPERATION_LOSS_GAUSSIAN_MIXTURE_NLL = 'gaussian_mixture_nll'


def resolve_operation_loss_config(config, latent_dim):
    if config is None:
        config = {}
    if isinstance(config, str):
        config = {'type': config}
    if not isinstance(config, dict):
        raise ValueError("operation_loss must be a dict or loss type string")
    if latent_dim <= 0:
        raise ValueError("operation_loss latent_dim must be positive")

    loss_type = str(config.get('type', OPERATION_LOSS_HARD_MIN)).lower()
    if loss_type not in {
            OPERATION_LOSS_HARD_MIN,
            OPERATION_LOSS_GAUSSIAN_MIXTURE_NLL}:
        raise ValueError(f"Unsupported operation loss type: {loss_type}")

    weight = float(config.get('weight', 1.0))
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("operation_loss weight must be finite and non-negative")

    resolved = {
        'type': loss_type,
        'weight': weight,
    }
    if loss_type == OPERATION_LOSS_HARD_MIN:
        return resolved

    variance = float(config.get('variance', 1.0))
    if not math.isfinite(variance) or variance <= 0:
        raise ValueError("operation_loss variance must be finite and positive")

    mixture_weights = config.get('mixture_weights', [0.5, 0.5])
    if not isinstance(mixture_weights, (list, tuple)) or len(mixture_weights) != 2:
        raise ValueError("operation_loss mixture_weights must contain exactly two values")
    mixture_weights = [float(value) for value in mixture_weights]
    if any(not math.isfinite(value) or value <= 0 for value in mixture_weights):
        raise ValueError("operation_loss mixture_weights must be finite and positive")
    if not math.isclose(sum(mixture_weights), 1.0, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError("operation_loss mixture_weights must sum to 1")

    resolved.update({
        'variance': variance,
        'mixture_weights': mixture_weights,
        'temperature': 2.0 * variance / latent_dim,
    })
    return resolved


def operation_loss(per_loss_q1, per_loss_q2, config):
    if per_loss_q1.shape != per_loss_q2.shape:
        raise ValueError("q1 and q2 per-sample losses must have the same shape")

    hard_min_loss = torch.minimum(per_loss_q1, per_loss_q2).mean()
    loss_type = config['type']
    if loss_type == OPERATION_LOSS_HARD_MIN:
        return hard_min_loss * config['weight'], hard_min_loss

    temperature = config['temperature']
    log_weights = per_loss_q1.new_tensor(config['mixture_weights']).log()
    component_logits = torch.stack(
        [
            log_weights[0] - per_loss_q1 / temperature,
            log_weights[1] - per_loss_q2 / temperature,
        ],
        dim=-1,
    )
    scaled_nll = -temperature * torch.logsumexp(component_logits, dim=-1)
    return scaled_nll.mean() * config['weight'], hard_min_loss


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
