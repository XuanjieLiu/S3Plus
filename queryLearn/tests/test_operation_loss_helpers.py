import sys
import unittest
from pathlib import Path

import torch


S3PLUS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(S3PLUS_ROOT))

from queryLearn.training_helpers import (  # noqa: E402
    operation_loss,
    resolve_operation_loss_config,
)


class OperationLossHelpersTest(unittest.TestCase):
    def test_default_config_preserves_hard_min_loss(self):
        config = resolve_operation_loss_config(None, latent_dim=8)
        q1_loss = torch.tensor([0.1, 0.4])
        q2_loss = torch.tensor([0.3, 0.2])

        loss, hard_min_loss = operation_loss(q1_loss, q2_loss, config)

        self.assertAlmostEqual(loss.item(), 0.15, places=6)
        self.assertAlmostEqual(hard_min_loss.item(), 0.15, places=6)

    def test_equal_component_errors_preserve_mse_scale(self):
        config = resolve_operation_loss_config(
            {
                'type': 'gaussian_mixture_nll',
                'variance': 0.04,
                'mixture_weights': [0.5, 0.5],
                'weight': 1.0,
            },
            latent_dim=8,
        )
        q1_loss = torch.tensor([0.2, 0.4])
        q2_loss = q1_loss.clone()

        loss, _ = operation_loss(q1_loss, q2_loss, config)

        self.assertAlmostEqual(loss.item(), 0.3, places=6)
        self.assertAlmostEqual(config['temperature'], 0.01, places=8)

    def test_small_variance_approaches_hard_min(self):
        config = resolve_operation_loss_config(
            {
                'type': 'gaussian_mixture_nll',
                'variance': 4e-6,
                'mixture_weights': [0.5, 0.5],
            },
            latent_dim=8,
        )
        q1_loss = torch.tensor([0.1, 0.4])
        q2_loss = torch.tensor([0.3, 0.2])

        loss, hard_min_loss = operation_loss(q1_loss, q2_loss, config)

        expected_offset = config['temperature'] * torch.log(torch.tensor(2.0))
        self.assertAlmostEqual(
            loss.item(),
            (hard_min_loss + expected_offset).item(),
            places=6,
        )

    def test_gmm_loss_gives_both_components_gradient(self):
        config = resolve_operation_loss_config(
            {
                'type': 'gaussian_mixture_nll',
                'variance': 0.4,
                'mixture_weights': [0.5, 0.5],
            },
            latent_dim=8,
        )
        q1_loss = torch.tensor([0.1], requires_grad=True)
        q2_loss = torch.tensor([0.2], requires_grad=True)

        loss, _ = operation_loss(q1_loss, q2_loss, config)
        loss.backward()

        self.assertGreater(q1_loss.grad.item(), 0)
        self.assertGreater(q2_loss.grad.item(), 0)
        self.assertAlmostEqual(q1_loss.grad.item() + q2_loss.grad.item(), 1.0, places=6)

    def test_invalid_variance_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "variance"):
            resolve_operation_loss_config(
                {'type': 'gaussian_mixture_nll', 'variance': 0},
                latent_dim=8,
            )

    def test_invalid_mixture_weights_are_rejected(self):
        invalid_weights = [
            [1.0],
            [0.0, 1.0],
            [0.4, 0.4],
        ]
        for weights in invalid_weights:
            with self.subTest(weights=weights):
                with self.assertRaisesRegex(ValueError, "mixture_weights"):
                    resolve_operation_loss_config(
                        {
                            'type': 'gaussian_mixture_nll',
                            'variance': 0.04,
                            'mixture_weights': weights,
                        },
                        latent_dim=8,
                    )


if __name__ == "__main__":
    unittest.main()
