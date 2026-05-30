import sys
import unittest
from pathlib import Path

import torch


S3PLUS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(S3PLUS_ROOT))

from queryLearn.label_codebook import (  # noqa: E402
    build_label_codebook,
    load_label_codebook,
)
from queryLearn.opernet import OperNet  # noqa: E402
from shared import DEVICE  # noqa: E402


class FakeOriginalVQLayer:
    def __init__(self, code_indices):
        self.code_indices = torch.tensor(code_indices, dtype=torch.long, device=DEVICE)
        self.commitment_cost = 0.25
        self.embedding_cost = 1.0

    def get_code_indices(self, x):
        return self.code_indices[:x.size(0)]


class FakeSpsModel:
    def __init__(self, z, original_vq_layer):
        self.latent_code_1 = z.size(1)
        self.model = self
        self.vq_layer = original_vq_layer
        self.z = z.to(DEVICE)

    def batch_encode_to_z(self, x):
        return self.z[:x.size(0)], None, None


def prototype_tensor(num_codes=21, dim=4):
    values = torch.arange(num_codes * dim, dtype=torch.float32, device=DEVICE)
    return values.reshape(num_codes, dim)


class LabelCodebookHelpersTest(unittest.TestCase):
    def test_builds_21_code_quantizer_from_unique_labels(self):
        z = prototype_tensor()
        labels = list(range(21))
        codebook = build_label_codebook(
            z,
            labels,
            FakeOriginalVQLayer([[i] for i in range(21)]),
            codes=list(range(21)),
        )

        self.assertEqual(codebook.vq_layer.num_embeddings, 21)
        self.assertEqual(codebook.num_labels, list(range(21)))
        self.assertEqual(codebook.label_to_index[20], 20)
        self.assertTrue(torch.equal(codebook.num_z_c, z))

    def test_missing_label_raises(self):
        z = prototype_tensor(20)
        labels = [label for label in range(21) if label != 17]
        with self.assertRaisesRegex(ValueError, "missing labels"):
            build_label_codebook(
                z,
                labels,
                FakeOriginalVQLayer([[i] for i in range(20)]),
                codes=list(range(21)),
            )

    def test_duplicate_label_raises(self):
        z = prototype_tensor(21)
        labels = list(range(20)) + [3]
        with self.assertRaisesRegex(ValueError, "duplicate labels"):
            build_label_codebook(
                z,
                labels,
                FakeOriginalVQLayer([[i] for i in range(21)]),
                codes=list(range(21)),
            )

    def test_original_code_collision_raises(self):
        z = prototype_tensor()
        labels = list(range(21))
        code_indices = [[i] for i in range(20)] + [[0]]
        with self.assertRaisesRegex(ValueError, "collide"):
            build_label_codebook(
                z,
                labels,
                FakeOriginalVQLayer(code_indices),
                codes=list(range(21)),
            )

    def test_load_label_codebook_smoke_with_fake_loader_and_sps(self):
        z = prototype_tensor()
        labels = [f"{i}-circle-blue.png" for i in range(21)]
        fake_loader = [(torch.zeros(21, 3, 4, 4, device=DEVICE), labels)]
        sps_model = FakeSpsModel(z, FakeOriginalVQLayer([[i] for i in range(21)]))

        codebook = load_label_codebook(
            fake_loader,
            sps_model,
            codes=list(range(21)),
        )

        self.assertEqual(codebook.vq_layer.num_embeddings, 21)

    def test_label_codebook_quantizer_can_be_attached_to_opernet(self):
        z = prototype_tensor()
        codebook = build_label_codebook(
            z,
            list(range(21)),
            FakeOriginalVQLayer([[i] for i in range(21)]),
            codes=list(range(21)),
        )

        oper_net = OperNet(
            in_dim=10,
            out_dim=4,
            n_hidden_layers=1,
            unit=8,
            vq_layer=codebook.vq_layer,
            train_vq=False,
        ).to(DEVICE)

        self.assertEqual(oper_net.vq_layer.num_embeddings, 21)
        self.assertTrue(all(not param.requires_grad for param in oper_net.vq_layer.parameters()))


if __name__ == "__main__":
    unittest.main()
