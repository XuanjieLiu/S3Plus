import os
import sys

import torch
from torch import optim
import torch.nn as nn

sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from VQ.VQVAE import VQVAE, MultiVectorQuantizer, split_into_three
from VQ.common_func import load_config_from_exp_name, parse_label
from VQ.eval_common import CommonEvaler
from dataloader import load_enc_eval_data
from loss_counter import LossCounter
from shared import DEVICE
from utils import init_dataloaders

try:
    from opVQLearn.op_vis import save_op_assignment_tables
except ImportError:
    from op_vis import save_op_assignment_tables


VQSPS_EXP_ROOT = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../VQ/exp/')
STAGE_TRAIN = 'train'
STAGE_VAL = 'val'

EVAL_TERMS = [
    'pred_loss',
    'op_vq_loss',
    'sps_vq_loss',
    'balance_loss',
    'symm_loss',
    'total_loss',
    'add_acc',
    'mm21_acc',
    'add_total',
    'mm21_total',
    'code1_rate',
    'code2_rate',
    'add_code1_rate',
    'add_code2_rate',
    'mm21_code1_rate',
    'mm21_code2_rate',
    'code_entropy',
]


def load_VQSPS_loader(config):
    vqsps_exp_name = config['VQSPS']['EXP_NAME']
    vqsps_config = load_config_from_exp_name(vqsps_exp_name)
    model_path = os.path.join(VQSPS_EXP_ROOT, vqsps_exp_name, config['VQSPS']['CHECK_POINT_NAME'])
    return CommonEvaler(vqsps_config, model_path), vqsps_config


def comb_op_z(ea, eb, q):
    return torch.cat([ea, eb, q], dim=-1)


def make_mlp(in_dim, out_dim, n_hidden_layers, unit):
    if n_hidden_layers < 1:
        raise ValueError("MLP expects at least one hidden layer")
    layers = [nn.Linear(in_dim, unit), nn.ReLU()]
    for _ in range(n_hidden_layers - 1):
        layers.extend([nn.Linear(unit, unit), nn.ReLU()])
    layers.append(nn.Linear(unit, out_dim))
    return nn.Sequential(*layers)


class OpEncoder(nn.Module):
    def __init__(self, in_dim, op_code_dim, n_hidden_layers, unit):
        super().__init__()
        self.net = make_mlp(in_dim, op_code_dim, n_hidden_layers, unit)

    def forward(self, ea, eb, ec):
        return self.net(torch.cat([ea, eb, ec], dim=-1))


class OpVQCodebook(nn.Module):
    def __init__(
            self,
            op_code_dim,
            num_codes,
            commitment_cost=0.25,
            embedding_cost=1.0,
            softmax_tau=0.5,
            init_codes=None):
        super().__init__()
        if num_codes != 2:
            raise ValueError("OpVQ v1 expects exactly two operation codes")
        self.op_code_dim = op_code_dim
        self.num_codes = num_codes
        self.commitment_cost = commitment_cost
        self.embedding_cost = embedding_cost
        self.softmax_tau = softmax_tau
        self.codes = nn.Embedding(num_codes, op_code_dim)
        if init_codes is not None:
            init_tensor = torch.tensor(init_codes, dtype=torch.float32)
            if tuple(init_tensor.shape) != (num_codes, op_code_dim):
                raise ValueError(
                    f"init_codes shape {tuple(init_tensor.shape)} does not match {(num_codes, op_code_dim)}"
                )
            self.codes.weight.data.copy_(init_tensor)
        self.mse_loss = nn.MSELoss()

    def distances(self, z_op):
        return (
            torch.sum(z_op ** 2, dim=1, keepdim=True) +
            torch.sum(self.codes.weight ** 2, dim=1) -
            2.0 * torch.matmul(z_op, self.codes.weight.t())
        )

    def forward(self, z_op):
        distances = self.distances(z_op)
        indices = torch.argmin(distances, dim=1)
        quantized = self.codes(indices)
        embedding_loss = self.mse_loss(quantized, z_op.detach()) * self.embedding_cost
        commitment_loss = self.mse_loss(z_op, quantized.detach()) * self.commitment_cost
        vq_loss = embedding_loss + commitment_loss
        quantized_st = z_op + (quantized - z_op).detach()
        probs = torch.softmax(-distances / self.softmax_tau, dim=1)
        return quantized_st, vq_loss, indices, probs


class OperDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, n_hidden_layers, unit, vq_layer: MultiVectorQuantizer):
        super().__init__()
        self.net = make_mlp(in_dim, out_dim, n_hidden_layers, unit)
        self.vq_layer = vq_layer
        for p in self.vq_layer.parameters():
            p.requires_grad = False
        self.vq_layer.eval()

    def forward(self, x):
        z_pred = self.net(x)
        e_pred, sps_vq_loss = self.vq_layer(z_pred)
        return e_pred, z_pred, sps_vq_loss

    def train(self, mode: bool = True):
        super().train(mode)
        self.vq_layer.eval()
        return self


class OpVQLearn:
    def __init__(self, config, model_path=None, loaded_model: VQVAE = None):
        self.config = config
        self._exp_dir = config.get('_exp_dir', None)
        self.sps_model, sps_config = load_VQSPS_loader(config)
        self._set_sps_untrainable()
        self.train_loader, self.eval_loader, self.single_img_eval_loader = init_dataloaders(config)

        op_code_config = config['op_code']
        self.op_code_dim = op_code_config['dim']
        self.num_op_codes = op_code_config.get('num_codes', 2)
        self.op_vq_loss_scalar = op_code_config.get('loss_scalar', 1.0)
        self.sps_vq_loss_scalar = config.get('sps_vq_loss_scalar', config.get('eqLoss_scalar', 0.05))

        self.op_encoder = OpEncoder(
            in_dim=self.sps_model.latent_code_1 * 3,
            op_code_dim=self.op_code_dim,
            n_hidden_layers=config['op_encoder']['n_hidden_layers'],
            unit=config['op_encoder']['unit'],
        ).to(DEVICE)
        self.op_vq = OpVQCodebook(
            op_code_dim=self.op_code_dim,
            num_codes=self.num_op_codes,
            commitment_cost=op_code_config.get('commitment_cost', 0.25),
            embedding_cost=op_code_config.get('embedding_cost', 1.0),
            softmax_tau=op_code_config.get('softmax_tau', 0.5),
            init_codes=op_code_config.get('init_codes', None),
        ).to(DEVICE)
        self.oper_decoder = OperDecoder(
            in_dim=self.sps_model.latent_code_1 * 2 + self.op_code_dim,
            out_dim=self.sps_model.latent_code_1,
            n_hidden_layers=config['decoder']['n_hidden_layers'],
            unit=config['decoder']['unit'],
            vq_layer=self.sps_model.model.vq_layer,
        ).to(DEVICE)

        balance_config = config.get('balance', {})
        self.use_balance_loss = balance_config.get('use_balance_loss', False)
        self.balance_loss_scalar = balance_config.get('loss_scalar', 0.01)
        symm_config = config.get('symm', {})
        self.use_symm_loss = symm_config.get('use_symm_loss', False)
        self.symm_loss_scalar = symm_config.get('loss_scalar', 0.05)
        self.symm_include_sps_vq_loss = symm_config.get('include_sps_vq_loss', True)
        self.train_result_path = config['train_result_path']
        self.eval_result_path = config['eval_result_path']
        self.train_record_path = config['train_record_path']
        self.eval_record_path = config['eval_record_path']
        self.log_interval = config['log_interval']
        self.eval_interval = config['eval_interval']
        self.checkpoint_interval = config['checkpoint_interval']
        self.checkpoint_after = config.get('checkpoint_after', 0)
        self.max_iter_num = config['max_iter_num']
        self.model_path = model_path or config['model_path']
        self.op_vis_format = config.get('op_vis_format', 'png').lower().lstrip('.')
        self.grad_clip_norm = config.get('grad_clip_norm', None)
        self.num_z_c = None
        self.num_labels = None
        self._num_label_to_index = None
        self.mean_mse = nn.MSELoss(reduction='mean')
        print(
            f"OpVQ: op_code_dim={self.op_code_dim}, num_codes={self.num_op_codes}, "
            f"balance={self.use_balance_loss}, symm={self.use_symm_loss}"
        )

    def _set_sps_untrainable(self):
        for p in self.sps_model.model.parameters():
            p.requires_grad = False
        self.sps_model.model.eval()

    def _ensure_num_z_c(self):
        if self.num_z_c is not None and self.num_labels is not None and self._num_label_to_index is not None:
            return
        num_z, num_labels = load_enc_eval_data(
            self.single_img_eval_loader,
            lambda x: self.sps_model.model.batch_encode_to_z(x)[0]
        )
        num_z_c = num_z[:, :self.sps_model.latent_code_1].detach()
        if num_z_c.device != DEVICE:
            num_z_c = num_z_c.to(DEVICE)
        self.num_z_c = num_z_c
        self.num_labels = num_labels
        label_to_index = {}
        for i, lab in enumerate(num_labels):
            if lab not in label_to_index:
                label_to_index[lab] = i
        self._num_label_to_index = label_to_index

    def _balance_loss(self, probs):
        mean_probs = probs.mean(dim=0)
        target = torch.full_like(mean_probs, 1.0 / self.num_op_codes)
        return (mean_probs - target).pow(2).sum()

    def _decode_with_q(self, ea, eb, q):
        return self.oper_decoder(comb_op_z(ea, eb, q))

    def _symm_loss_for_q(self, z_a, z_b, z_c, q):
        e_ab, _, sps_ab = self._decode_with_q(z_a, z_b, q)
        e_abc_1, _, sps_abc_1 = self._decode_with_q(e_ab, z_c, q)
        e_ac, _, sps_ac = self._decode_with_q(z_a, z_c, q)
        e_acb_1, _, sps_acb_1 = self._decode_with_q(e_ac, z_b, q)
        e_bac_2, _, sps_bac_2 = self._decode_with_q(z_b, e_ac, q)
        e_bc, _, sps_bc = self._decode_with_q(z_b, z_c, q)
        e_abc_2, _, sps_abc_2 = self._decode_with_q(z_a, e_bc, q)

        symm_raw = self.mean_mse(e_abc_1, e_acb_1) + self.mean_mse(e_abc_2, e_bac_2)
        if not self.symm_include_sps_vq_loss:
            return symm_raw

        sps_vq_loss = sps_ab + sps_abc_1 + sps_ac + sps_acb_1 + sps_bac_2 + sps_bc + sps_abc_2
        return symm_raw + sps_vq_loss * self.sps_vq_loss_scalar

    def _symm_loss(self, ea, eb, ec):
        if not self.use_symm_loss:
            return (ea.sum() + eb.sum() + ec.sum()) * 0.0

        content = torch.cat([ea, eb, ec], dim=0)
        if content.size(0) < 3:
            return content.sum() * 0.0

        perm = torch.randperm(content.size(0), device=content.device)
        third = content.size(0) // 3
        if third == 0:
            return content.sum() * 0.0
        z_a = content[perm[:third]]
        z_b = content[perm[third:2 * third]]
        z_c = content[perm[2 * third:3 * third]]

        losses = []
        for code_ndx in range(self.num_op_codes):
            q = self.op_vq.codes.weight[code_ndx].view(1, -1).expand(z_a.size(0), -1)
            losses.append(self._symm_loss_for_q(z_a, z_b, z_c, q))
        return sum(losses) * self.symm_loss_scalar

    @staticmethod
    def _code_entropy(probs):
        mean_probs = probs.mean(dim=0)
        return float((-(mean_probs * torch.log(mean_probs + 1e-12)).sum()).detach().cpu())

    def _forward_batch(self, ea, eb, ec):
        z_op = self.op_encoder(ea, eb, ec)
        q, op_vq_loss_raw, code_indices, code_probs = self.op_vq(z_op)
        e_pred, z_pred, sps_vq_loss_raw = self.oper_decoder(comb_op_z(ea, eb, q))
        pred_loss = self.mean_mse(e_pred, ec)
        op_vq_loss = op_vq_loss_raw * self.op_vq_loss_scalar
        sps_vq_loss = sps_vq_loss_raw * self.sps_vq_loss_scalar
        balance_raw = self._balance_loss(code_probs)
        balance_loss = balance_raw * self.balance_loss_scalar if self.use_balance_loss else balance_raw.detach() * 0.0
        symm_loss = self._symm_loss(ea, eb, ec)
        total_loss = pred_loss + op_vq_loss + sps_vq_loss + balance_loss + symm_loss
        return {
            'pred_loss': pred_loss,
            'op_vq_loss': op_vq_loss,
            'sps_vq_loss': sps_vq_loss,
            'balance_loss': balance_loss,
            'symm_loss': symm_loss,
            'total_loss': total_loss,
            'z_pred': z_pred,
            'code_indices': code_indices,
            'code_probs': code_probs,
        }

    def one_epoch(
            self,
            epoch,
            data_loader,
            optimizer=None,
            stage=STAGE_TRAIN,
            loss_counter: LossCounter = None,
            save_op_vis=False,
            op_vis_dir=None):
        self.sps_model.model.eval()
        epoch_losses = {
            key: [] for key in ('pred_loss', 'op_vq_loss', 'sps_vq_loss', 'balance_loss', 'symm_loss', 'total_loss')
        }
        epoch_label_a = []
        epoch_label_b = []
        epoch_label_c = []
        epoch_z_pred = []
        epoch_code_indices = []
        epoch_code_probs = []

        for batch_ndx, sample in enumerate(data_loader):
            if optimizer is not None:
                optimizer.zero_grad()
            data, labels = sample
            sizes = data[0].size()
            data_all = torch.stack(data, dim=0).reshape(3 * sizes[0], sizes[1], sizes[2], sizes[3])
            data_all = data_all.to(DEVICE, non_blocking=True)
            e_all, e_q_loss, z_all = self.sps_model.model.batch_encode_to_z(data_all)
            e_content = e_all[..., 0:self.sps_model.latent_code_1]
            label_a = [parse_label(x) for x in labels[0]]
            label_b = [parse_label(x) for x in labels[1]]
            label_c = [parse_label(x) for x in labels[2]]
            ea, eb, ec = split_into_three(e_content)

            out = self._forward_batch(ea, eb, ec)

            if loss_counter is not None or save_op_vis:
                for key in epoch_losses:
                    epoch_losses[key].append(out[key].item())
                epoch_label_a.extend(label_a)
                epoch_label_b.extend(label_b)
                epoch_label_c.extend(label_c)
                epoch_z_pred.append(out['z_pred'].detach().cpu())
                epoch_code_indices.append(out['code_indices'].detach().cpu())
                epoch_code_probs.append(out['code_probs'].detach().cpu())

            if optimizer is not None:
                out['total_loss'].backward()
                if self.grad_clip_norm is not None:
                    params = self._trainable_params()
                    nn.utils.clip_grad_norm_(params, self.grad_clip_norm)
                optimizer.step()

        if (loss_counter is not None or save_op_vis) and epoch_losses['total_loss']:
            z_pred_epoch = torch.cat(epoch_z_pred, dim=0).to(DEVICE)
            code_indices_epoch = torch.cat(epoch_code_indices, dim=0)
            code_probs_epoch = torch.cat(epoch_code_probs, dim=0)
            pred_correct = self._target_correct(epoch_label_c, z_pred_epoch)
            metrics = self._batch_metrics(
                epoch_label_a,
                epoch_label_b,
                epoch_label_c,
                code_indices_epoch,
                code_probs_epoch,
                pred_correct,
            )
            if loss_counter is not None:
                loss_counter.add_values([
                    self._mean(epoch_losses['pred_loss']),
                    self._mean(epoch_losses['op_vq_loss']),
                    self._mean(epoch_losses['sps_vq_loss']),
                    self._mean(epoch_losses['balance_loss']),
                    self._mean(epoch_losses['symm_loss']),
                    self._mean(epoch_losses['total_loss']),
                    metrics['add_acc'],
                    metrics['mm21_acc'],
                    metrics['add_total'],
                    metrics['mm21_total'],
                    metrics['code1_rate'],
                    metrics['code2_rate'],
                    metrics['add_code1_rate'],
                    metrics['add_code2_rate'],
                    metrics['mm21_code1_rate'],
                    metrics['mm21_code2_rate'],
                    metrics['code_entropy'],
                ])
            if save_op_vis:
                if op_vis_dir is None:
                    op_vis_dir = self.eval_result_path if stage == STAGE_VAL else self.train_result_path
                stage_name = 'eval' if stage == STAGE_VAL else stage
                save_op_assignment_tables(
                    op_vis_dir,
                    stage_name,
                    epoch,
                    epoch_label_a,
                    epoch_label_b,
                    epoch_label_c,
                    code_indices_epoch,
                    pred_correct.detach().cpu(),
                    self.op_vis_format,
                )

    @staticmethod
    def _mean(values):
        return sum(values) / len(values) if values else 0.0

    def _trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def parameters(self):
        for module in (self.op_encoder, self.op_vq, self.oper_decoder):
            yield from module.parameters()

    def _resume_model(self):
        if os.path.exists(self.model_path):
            ckpt = torch.load(self.model_path, map_location=DEVICE)
            self.op_encoder.load_state_dict(ckpt['op_encoder_state_dict'])
            self.op_vq.load_state_dict(ckpt['op_vq_state_dict'])
            self.oper_decoder.load_state_dict(ckpt['oper_decoder_state_dict'])
            print(f"Model is loaded from {self.model_path}")
        else:
            print("No checkpoint found, training from scratch")

    def _save_model(self, path, epoch):
        ckpt = {
            'op_encoder_state_dict': self.op_encoder.state_dict(),
            'op_vq_state_dict': self.op_vq.state_dict(),
            'oper_decoder_state_dict': self.oper_decoder.state_dict(),
            'epoch': epoch,
        }
        torch.save(ckpt, path)

    def train(self):
        os.makedirs(self.train_result_path, exist_ok=True)
        os.makedirs(self.eval_result_path, exist_ok=True)
        self.op_encoder.train()
        self.op_vq.train()
        self.oper_decoder.train()
        train_loss_counter = LossCounter(EVAL_TERMS, record_path=self.train_record_path)
        eval_loss_counter = LossCounter(EVAL_TERMS, record_path=self.eval_record_path)
        optim_params = self._trainable_params()
        weight_decay = self.config.get('weight_decay', 0.0)
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        if optimizer_name == 'adam':
            optimizer = optim.Adam(optim_params, lr=self.config['learning_rate'], weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(optim_params, lr=self.config['learning_rate'], weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        start_epoch = train_loss_counter.load_iter_num(self.train_record_path)
        self._resume_model()
        for epoch in range(start_epoch, self.max_iter_num):
            print(f"Epoch {epoch}")
            is_log_epoch = epoch % self.log_interval == 0
            self.one_epoch(
                epoch,
                self.train_loader,
                optimizer,
                stage=STAGE_TRAIN,
                loss_counter=train_loss_counter,
                save_op_vis=epoch % self.eval_interval == 0,
                op_vis_dir=self.train_result_path,
            )

            if is_log_epoch:
                train_loss_counter.record_and_clear(num=epoch)

            if epoch % self.eval_interval == 0:
                self.op_encoder.eval()
                self.op_vq.eval()
                self.oper_decoder.eval()
                with torch.no_grad():
                    self.one_epoch(
                        epoch,
                        self.eval_loader,
                        optimizer=None,
                        stage=STAGE_VAL,
                        loss_counter=eval_loss_counter,
                        save_op_vis=is_log_epoch,
                        op_vis_dir=self.eval_result_path,
                    )
                eval_loss_counter.record_and_clear(num=epoch)
                self.op_encoder.train()
                self.op_vq.train()
                self.oper_decoder.train()

            if epoch % self.checkpoint_interval == 0 and epoch >= self.checkpoint_after:
                ckpt_dir = self._exp_dir
                ckpt_name = os.path.join(ckpt_dir, f'checkpoint_{epoch}.pt')
                self._save_model(ckpt_name, epoch)
                self._save_model(self.model_path, epoch)

    def _target_correct(self, target_labels, pred_z):
        self._ensure_num_z_c()
        target_idx = [self._num_label_to_index.get(label, -1) for label in target_labels]
        target_idx = torch.tensor(target_idx, device=DEVICE)
        valid_mask = target_idx >= 0
        correct = torch.zeros(len(target_labels), device=DEVICE, dtype=torch.bool)
        if not valid_mask.any():
            return correct

        pred_valid = pred_z[valid_mask]
        idx_valid = target_idx[valid_mask]
        target_z = self.num_z_c[idx_valid]
        dist_target = (pred_valid - target_z).pow(2).sum(dim=-1)
        dist_all = torch.cdist(pred_valid, self.num_z_c, p=2).pow(2)
        dist_all[torch.arange(dist_all.size(0), device=DEVICE), idx_valid] = float('inf')
        min_other = dist_all.min(dim=1).values
        correct[valid_mask] = dist_target < min_other
        return correct

    def _batch_metrics(self, label_a, label_b, label_c, code_indices, code_probs, pred_correct):
        valid_op = []
        add_mask = []
        mm21_mask = []
        for a, b, c in zip(label_a, label_b, label_c):
            is_add = c == a + b
            is_mm21 = c == (a * b) % 21
            is_special = is_add and is_mm21
            valid_op.append(is_add or is_mm21)
            add_mask.append(is_add and not is_special)
            mm21_mask.append(is_mm21 and not is_special)

        def rate(mask, code=None, correct=False):
            total = sum(1 for value in mask if value)
            if total == 0:
                return 0.0
            count = 0
            for i, value in enumerate(mask):
                if not value:
                    continue
                if code is not None and int(code_indices[i].item()) != code:
                    continue
                if correct and not bool(pred_correct[i].item()):
                    continue
                count += 1
            return count / total

        def correct_rate(mask):
            return rate(mask, correct=True)

        add_total = sum(1 for value in add_mask if value)
        mm21_total = sum(1 for value in mm21_mask if value)
        code_entropy = self._code_entropy(code_probs.to(DEVICE))
        return {
            'add_acc': correct_rate(add_mask),
            'mm21_acc': correct_rate(mm21_mask),
            'add_total': add_total,
            'mm21_total': mm21_total,
            'code1_rate': rate(valid_op, code=0),
            'code2_rate': rate(valid_op, code=1),
            'add_code1_rate': rate(add_mask, code=0),
            'add_code2_rate': rate(add_mask, code=1),
            'mm21_code1_rate': rate(mm21_mask, code=0),
            'mm21_code2_rate': rate(mm21_mask, code=1),
            'code_entropy': code_entropy,
        }
