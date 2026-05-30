from collections import Counter
from dataclasses import dataclass

import torch

from dataloader import load_enc_eval_data
from shared import DEVICE
from VQ.VQVAE import MultiVectorQuantizer


DEFAULT_LABEL_CODEBOOK_CODES = list(range(21))


@dataclass
class LabelCodebook:
    codes: list
    num_z_c: torch.Tensor
    num_labels: list
    label_to_index: dict
    original_code_indices: list
    vq_layer: MultiVectorQuantizer


def _code_key(indices):
    if isinstance(indices, torch.Tensor):
        values = indices.detach().cpu().reshape(-1).tolist()
    else:
        values = list(indices)
    values = [int(value) for value in values]
    return values[0] if len(values) == 1 else tuple(values)


def validate_label_prototypes(num_z_c, num_labels, original_vq_layer, codes, fail_fast=True):
    codes = [int(code) for code in codes]
    label_counts = Counter(int(label) for label in num_labels)
    expected = set(codes)
    actual = set(label_counts)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    duplicates = sorted(label for label, count in label_counts.items() if count > 1)
    errors = []
    if missing:
        errors.append(f"missing labels: {missing}")
    if duplicates:
        errors.append(f"duplicate labels: {duplicates}")
    if errors:
        msg = "Invalid label codebook prototypes: " + "; ".join(errors)
        raise ValueError(msg)
    if extra:
        msg = f"Invalid label codebook prototypes: unexpected labels: {extra}"
        if fail_fast:
            raise ValueError(msg)
        print(f"WARNING: {msg}; ignored for label codebook initialization")

    label_to_source_index = {int(label): i for i, label in enumerate(num_labels)}
    ordered_indices = [label_to_source_index[int(code)] for code in codes]
    ordered_num_z_c = num_z_c[ordered_indices].detach()
    original_indices = original_vq_layer.get_code_indices(ordered_num_z_c)
    original_code_keys = [_code_key(row) for row in original_indices]
    code_counts = Counter(original_code_keys)
    collisions = sorted(code for code, count in code_counts.items() if count > 1)
    if collisions:
        collision_labels = {
            str(code): [codes[i] for i, key in enumerate(original_code_keys) if key == code]
            for code in collisions
        }
        msg = f"Label codebook prototypes collide in original SPS codebook: {collision_labels}"
        if fail_fast:
            raise ValueError(msg)
        print(f"WARNING: {msg}")

    return ordered_num_z_c, original_code_keys


def build_label_codebook(num_z_c, num_labels, original_vq_layer, codes=None, fail_fast=True):
    codes = DEFAULT_LABEL_CODEBOOK_CODES if codes is None else [int(code) for code in codes]
    if num_z_c.device != DEVICE:
        num_z_c = num_z_c.to(DEVICE)
    ordered_num_z_c, original_code_keys = validate_label_prototypes(
        num_z_c,
        num_labels,
        original_vq_layer,
        codes,
        fail_fast=fail_fast,
    )
    vq_layer = MultiVectorQuantizer(
        num_embeddings=len(codes),
        embedding_dim=ordered_num_z_c.shape[1],
        commitment_cost=original_vq_layer.commitment_cost,
        embedding_cost=original_vq_layer.embedding_cost,
        init_embs=ordered_num_z_c.detach().cpu(),
    ).to(DEVICE)
    for parameter in vq_layer.parameters():
        parameter.requires_grad = False
    vq_layer.eval()
    return LabelCodebook(
        codes=list(codes),
        num_z_c=ordered_num_z_c.to(DEVICE),
        num_labels=list(codes),
        label_to_index={code: i for i, code in enumerate(codes)},
        original_code_indices=original_code_keys,
        vq_layer=vq_layer,
    )


def load_label_codebook(single_img_eval_loader, sps_model, codes=None, fail_fast=True):
    if single_img_eval_loader is None:
        raise ValueError("operator.use_label_codebook=True requires single_img_eval_set_path")
    with torch.no_grad():
        num_z, num_labels = load_enc_eval_data(
            single_img_eval_loader,
            lambda x: sps_model.model.batch_encode_to_z(x)[0],
        )
    num_z_c = num_z[:, :sps_model.latent_code_1].detach()
    return build_label_codebook(
        num_z_c,
        num_labels,
        sps_model.model.vq_layer,
        codes=codes,
        fail_fast=fail_fast,
    )
