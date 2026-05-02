import torch
from torch import optim
import torch.nn as nn
import os
import sys
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from VQ.VQVAE import VQVAE, MultiVectorQuantizer, split_into_three
from VQ.common_func import load_config_from_exp_name
from VQ.eval_common import CommonEvaler
from shared import DEVICE
from utils import init_dataloaders
from loss_counter import LossCounter
from VQ.common_func import parse_label
from dataloader import load_enc_eval_data
try:
    from queryLearn.query_vis import record_dir, save_query_operation_tables
except ImportError:
    from query_vis import record_dir, save_query_operation_tables


VQSPS_EXP_ROOT = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../VQ/exp/')
STAGE_TRAIN = 'train'
STAGE_VAL = 'val'
EVAL_TERMS = [
    'oper_loss',
    'symm_loss',
    'total_loss',
    'add_acc_q0',
    'add_acc_q1',
    'mul_acc_q0',
    'mul_acc_q1',
    'add_acc',
    'mul_acc',
    'add_total',
    'mul_total',
    'q_l2_dist',
]


def load_VQSPS_loader(config):
    vqsps_exp_name = config['VQSPS']['EXP_NAME']
    vqsps_config = load_config_from_exp_name(vqsps_exp_name)
    model_path = os.path.join(VQSPS_EXP_ROOT, vqsps_exp_name, config['VQSPS']['CHECK_POINT_NAME'])
    return CommonEvaler(vqsps_config, model_path), vqsps_config


def comb_q_z(ea, eb, q):
    if q.dim() == 1:
        q = q.unsqueeze(0).expand(ea.size(0), -1)
    return torch.cat([ea, eb, q], dim=-1)


def regul_sample(z_all_content):
    idx_1 = torch.randperm(z_all_content.size(0))
    z_perm = z_all_content[idx_1, ...]
    z_a, z_b, z_c = split_into_three(z_perm)
    return z_a, z_b, z_c


def sanity_check_oper_loss(per_loss_1, per_loss_2, label_a, label_b, label_c):
    is_add = torch.tensor(
        [a + b == c for a, b, c in zip(label_a, label_b, label_c)],
        device=per_loss_1.device,
        dtype=torch.bool,
    )
    per_loss = torch.where(is_add, per_loss_1, per_loss_2)
    return per_loss.mean()


class OperNet(nn.Module):
    def __init__(self, in_dim, out_dim, n_hidden_layers, unit, vq_layer: MultiVectorQuantizer, train_vq: bool = False):
        super().__init__()
        layers = [nn.Linear(in_dim, unit), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers.extend([nn.Linear(unit, unit), nn.ReLU()])
        layers.append(nn.Linear(unit, out_dim))
        self.net = nn.Sequential(*layers)
        self.vq_layer = vq_layer
        self.train_vq = train_vq
        self._set_vq_trainable(train_vq)

    def _set_vq_trainable(self, train_vq: bool):
        self.train_vq = train_vq
        for p in self.vq_layer.parameters():
            p.requires_grad = train_vq

    def forward(self, x):
        z_oper = self.net(x)
        e_oper, e_q_loss = self.vq_layer(z_oper)
        return e_oper, e_q_loss, z_oper

    def train(self, mode: bool = True):
        super().train(mode)
        if self.train_vq:
            self.vq_layer.train(mode)
        else:
            self.vq_layer.eval()
        return self



class QueryLearn:
    def __init__(self, config, model_path=None, loaded_model: VQVAE = None):
        self.config = config
        self._exp_dir = config.get('_exp_dir', None)
        self.sps_model, sps_config = load_VQSPS_loader(config)
        self._set_sps_trainable()
        self.train_loader, self.eval_loader, self.single_img_eval_loader = init_dataloaders(config)
        self.query_dim = config.get('query_dim', config.get('query_learner', {}).get('in_dim', 8))
        self.queries = nn.Parameter(torch.randn(2, self.query_dim, device=DEVICE))
        self.oper_net = OperNet(
            in_dim=self.sps_model.latent_code_1 * 2 + self.query_dim,
            out_dim=self.sps_model.latent_code_1,
            n_hidden_layers=config['operator']['n_hidden_layers'],
            unit=config['operator']['unit'],
            vq_layer=self.sps_model.model.vq_layer,
            train_vq=False,
        ).to(DEVICE)
        self.train_result_path = config['train_result_path']
        self.eval_result_path = config['eval_result_path']
        self.train_record_path = config['train_record_path']
        self.eval_record_path = config['eval_record_path']
        self.log_interval = config['log_interval']
        self.eval_interval = config['eval_interval']
        self.checkpoint_interval = config['checkpoint_interval']
        self.checkpoint_after = config.get('checkpoint_after', 0)
        self.max_iter_num = config['max_iter_num']
        self.model_path = config['model_path']
        self.is_symm = config.get('is_symm', False)
        self.is_assoc = config.get('is_assoc', False)
        self.eqLoss_scalar = config.get('eqLoss_scalar', 0.05)
        self.symm_loss_scalar = config.get('symm_loss_scalar', 0.01)
        self.num_z_c = None
        self.num_labels = None
        self._num_label_to_index = None
        self.mean_mse = nn.MSELoss(reduction='mean')
        self.sanity_check = config.get('sanity_check', False)
        self.query_vis_format = config.get('query_vis_format', 'png').lower().lstrip('.')
        print('Sanity check mode...')

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

    def _set_sps_trainable(self):
        for p in self.sps_model.model.parameters():
            p.requires_grad = False
        self.sps_model.model.eval()

    def symm_loss(self, z_a, z_b, z_c, q):
        e_ab, e_q_loss_ab, z_ab = self.oper_net(comb_q_z(z_a, z_b, q))
        e_abc_1, e_q_loss_abc_1, z_abc_1 = self.oper_net(comb_q_z(e_ab, z_c, q))
        # symm_assoc
        e_ac, e_q_loss_ac, z_ac = self.oper_net(comb_q_z(z_a, z_c, q))
        e_acb_1, e_q_loss_acb_1, z_acb_1 = self.oper_net(comb_q_z(e_ac, z_b, q))
        e_bac_2, e_q_loss_bac_2, z_bac_2 = self.oper_net(comb_q_z(z_b, e_ac, q))
        # pure_assoc
        e_bc, e_q_loss_bc, z_bc = self.oper_net(comb_q_z(z_b, z_c, q))
        e_abc_2, e_q_loss_abc_2, z_abc_2 = self.oper_net(comb_q_z(z_a, e_bc, q))
        # choose loss
        assoc_plus_loss = torch.zeros(1)[0].to(DEVICE)
        if self.is_symm:
            assoc_plus_loss += self.mean_mse(e_abc_1, e_acb_1) * self.symm_loss_scalar
            assoc_plus_loss += self.mean_mse(e_abc_2, e_bac_2) * self.symm_loss_scalar
        if self.is_assoc:
            assoc_plus_loss += self.mean_mse(e_abc_1, e_abc_2) * self.symm_loss_scalar
        e_q_loss = e_q_loss_ab + e_q_loss_abc_1
        if self.is_symm:
            e_q_loss += e_q_loss_acb_1
            e_q_loss += e_q_loss_bac_2
        if self.is_assoc:
            e_q_loss += e_q_loss_abc_2
        return assoc_plus_loss + self.eqLoss_scalar * e_q_loss

    def one_epoch(
            self,
            epoch,
            data_loader,
            optimizer=None,
            stage=STAGE_TRAIN,
            loss_counter: LossCounter=None,
            save_query_vis=False,
            query_vis_dir=None):
        self.sps_model.model.eval()
        epoch_oper_losses = []
        epoch_symm_losses = []
        epoch_total_losses = []
        epoch_label_a = []
        epoch_label_b = []
        epoch_label_c = []
        epoch_q_out_1 = []
        epoch_q_out_2 = []
        epoch_ec = []
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
            q0 = self.queries[0]
            q1 = self.queries[1]
            q_in_1 = comb_q_z(ea, eb, q0)
            q_in_2 = comb_q_z(ea, eb, q1)
            e_q_out_1, eq_loss_1, z_q_out_1 = self.oper_net(q_in_1)
            e_q_out_2, eq_loss_2, z_q_out_2 = self.oper_net(q_in_2)

            # Per-sample MSE to ec, then choose the smaller one as loss
            per_loss_1 = (e_q_out_1 - ec).pow(2).mean(dim=-1) + eq_loss_1 * self.eqLoss_scalar
            per_loss_2 = (e_q_out_2 - ec).pow(2).mean(dim=-1) + eq_loss_2 * self.eqLoss_scalar

            if self.sanity_check:
                oper_loss = sanity_check_oper_loss(per_loss_1, per_loss_2, label_a, label_b, label_c)
            else:
                per_loss = torch.minimum(per_loss_1, per_loss_2)
                oper_loss = per_loss.mean()

            # symm loss
            q1_symm_loss = self.symm_loss(*regul_sample(e_content), q0)
            q2_symm_loss = self.symm_loss(*regul_sample(e_content), q1)
            symm_loss = q1_symm_loss + q2_symm_loss

            total_loss = oper_loss + symm_loss

            if loss_counter is not None or save_query_vis:
                epoch_oper_losses.append(oper_loss.item())
                epoch_symm_losses.append(symm_loss.item())
                epoch_total_losses.append(total_loss.item())
                epoch_label_a.extend(label_a)
                epoch_label_b.extend(label_b)
                epoch_label_c.extend(label_c)
                epoch_q_out_1.append(z_q_out_1.detach().cpu())
                epoch_q_out_2.append(z_q_out_2.detach().cpu())
                epoch_ec.append(ec.detach().cpu())

            if optimizer is not None:
                total_loss.backward()
                optimizer.step()

        if (loss_counter is not None or save_query_vis) and epoch_oper_losses:
            q_out_1_epoch = torch.cat(epoch_q_out_1, dim=0).to(DEVICE)
            q_out_2_epoch = torch.cat(epoch_q_out_2, dim=0).to(DEVICE)
            ec_epoch = torch.cat(epoch_ec, dim=0).to(DEVICE)
            accu = self._batch_query_accu(
                epoch_label_a,
                epoch_label_b,
                epoch_label_c,
                q_out_1_epoch,
                q_out_2_epoch,
                ec_epoch,
            )
            if loss_counter is not None:
                q_l2_dist = torch.norm(self.queries[0] - self.queries[1], p=2).item()
                loss_counter.add_values([
                    sum(epoch_oper_losses) / len(epoch_oper_losses),
                    sum(epoch_symm_losses) / len(epoch_symm_losses),
                    sum(epoch_total_losses) / len(epoch_total_losses),
                    accu['add_acc_q0'],
                    accu['add_acc_q1'],
                    accu['mul_acc_q0'],
                    accu['mul_acc_q1'],
                    accu['add_acc'],
                    accu['mul_acc'],
                    accu['add_total'],
                    accu['mul_total'],
                    q_l2_dist,
                ])
            if save_query_vis:
                if query_vis_dir is None:
                    query_vis_dir = self.eval_result_path if stage == STAGE_VAL else self.train_result_path
                stage_name = 'eval' if stage == STAGE_VAL else stage
                q1_correct = self._query_target_correct(epoch_label_c, q_out_1_epoch)
                q2_correct = self._query_target_correct(epoch_label_c, q_out_2_epoch)
                query_specs = [
                    ('q1', q1_correct, accu['add_acc_q0'], accu['mul_acc_q0']),
                    ('q2', q2_correct, accu['add_acc_q1'], accu['mul_acc_q1']),
                ]
                save_query_operation_tables(
                    query_vis_dir,
                    stage_name,
                    epoch,
                    epoch_label_a,
                    epoch_label_b,
                    epoch_label_c,
                    query_specs,
                    self.query_vis_format,
                )

    def _resume_model(self):
        if os.path.exists(self.model_path):
            ckpt = torch.load(self.model_path, map_location=DEVICE)
            if isinstance(ckpt, dict) and 'oper_net_state_dict' in ckpt:
                self.oper_net.load_state_dict(ckpt['oper_net_state_dict'])
                if 'queries' in ckpt:
                    ckpt_queries = ckpt['queries'].to(DEVICE)
                    if ckpt_queries.shape != self.queries.shape:
                        raise ValueError(
                            f"Checkpoint query shape {tuple(ckpt_queries.shape)} does not match "
                            f"current query shape {tuple(self.queries.shape)}"
                        )
                    self.queries.data.copy_(ckpt_queries)
            else:
                self.oper_net.load_state_dict(ckpt)
            print(f"Model is loaded from {self.model_path}")
        else:
            print("No checkpoint found, training from scratch")

    def _save_model(self, path, epoch):
        ckpt = {
            'oper_net_state_dict': self.oper_net.state_dict(),
            'queries': self.queries.detach().cpu(),
            'epoch': epoch,
        }
        torch.save(ckpt, path)

    def train(self):
        os.makedirs(self.train_result_path, exist_ok=True)
        os.makedirs(self.eval_result_path, exist_ok=True)
        self.oper_net.train()
        train_loss_counter = LossCounter(EVAL_TERMS, record_path=self.train_record_path)
        eval_loss_counter = LossCounter(EVAL_TERMS, record_path=self.eval_record_path)
        optim_params = list(self.oper_net.net.parameters()) + [self.queries]
        optimizer = optim.Adam(optim_params, lr=self.config['learning_rate'])
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
                save_query_vis=epoch % self.eval_interval == 0,
                query_vis_dir=self.train_result_path,
            )

            if is_log_epoch:
                train_loss_counter.record_and_clear(num=epoch)

            if epoch % self.eval_interval == 0:
                self.oper_net.eval()
                with torch.no_grad():
                    self.one_epoch(
                        epoch,
                        self.eval_loader,
                        optimizer=None,
                        stage=STAGE_VAL,
                        loss_counter=eval_loss_counter,
                        save_query_vis=is_log_epoch,
                        query_vis_dir=self.eval_result_path,
                    )
                eval_loss_counter.record_and_clear(num=epoch)
                self.oper_net.train()

            if epoch % self.checkpoint_interval == 0 and epoch >= self.checkpoint_after:
                ckpt_dir = self._exp_dir
                ckpt_name = os.path.join(ckpt_dir, f'checkpoint_{epoch}.pt')
                self._save_model(ckpt_name, epoch)
                self._save_model(self.model_path, epoch)

    def _query_target_correct(self, target_labels, q_out):
        self._ensure_num_z_c()
        target_idx = [self._num_label_to_index.get(label, -1) for label in target_labels]
        target_idx = torch.tensor(target_idx, device=DEVICE)
        valid_mask = target_idx >= 0
        correct = torch.zeros(len(target_labels), device=DEVICE, dtype=torch.bool)
        if not valid_mask.any():
            return correct

        q_valid = q_out[valid_mask]
        idx_valid = target_idx[valid_mask]
        target_z = self.num_z_c[idx_valid]
        dist_target = (q_valid - target_z).pow(2).sum(dim=-1)
        dist_all = torch.cdist(q_valid, self.num_z_c, p=2).pow(2)
        dist_all[torch.arange(dist_all.size(0), device=DEVICE), idx_valid] = float('inf')
        min_other = dist_all.min(dim=1).values
        correct[valid_mask] = dist_target < min_other
        return correct

    def _batch_query_accu(self, label_a, label_b, label_c, q_out_1, q_out_2, ec):
        self._ensure_num_z_c()
        label_c_idx = []
        valid_mask = []
        for a, b, c in zip(label_a, label_b, label_c):
            is_add = (c == a + b)
            is_mul = (c == (a * b) % 21)
            if is_add and is_mul:
                valid_mask.append(0)
                label_c_idx.append(-1)
                continue
            valid_mask.append(1)
            label_c_idx.append(self._num_label_to_index.get(c, -1))

        label_c_idx = torch.tensor(label_c_idx, device=DEVICE)
        valid_mask = torch.tensor(valid_mask, device=DEVICE, dtype=torch.bool)
        valid_mask = valid_mask & (label_c_idx >= 0)

        add_total = 0
        add_correct_q0 = 0
        add_correct_q1 = 0
        add_correct = 0
        mul_total = 0
        mul_correct_q0 = 0
        mul_correct_q1 = 0
        mul_correct = 0

        if not valid_mask.any():
            return {
                'add_acc_q0': 0.0,
                'add_acc_q1': 0.0,
                'mul_acc_q0': 0.0,
                'mul_acc_q1': 0.0,
                'add_acc': 0.0,
                'mul_acc': 0.0,
                'add_total': 0,
                'mul_total': 0,
            }

        ec_valid = ec[valid_mask]
        q1_valid = q_out_1[valid_mask]
        q2_valid = q_out_2[valid_mask]
        idx_valid = label_c_idx[valid_mask]
        valid_positions = valid_mask.nonzero(as_tuple=False).flatten().tolist()
        pos_to_valid = {pos: j for j, pos in enumerate(valid_positions)}

        dist_ec_q1 = (q1_valid - ec_valid).pow(2).sum(dim=-1)
        dist_ec_q2 = (q2_valid - ec_valid).pow(2).sum(dim=-1)

        dist_all_q1 = torch.cdist(q1_valid, self.num_z_c, p=2).pow(2)
        dist_all_q2 = torch.cdist(q2_valid, self.num_z_c, p=2).pow(2)

        inf_mask = torch.zeros_like(dist_all_q1, dtype=torch.bool)
        inf_mask[torch.arange(dist_all_q1.size(0), device=DEVICE), idx_valid] = True
        dist_all_q1 = dist_all_q1.masked_fill(inf_mask, float('inf'))
        dist_all_q2 = dist_all_q2.masked_fill(inf_mask, float('inf'))

        min_other_q1 = dist_all_q1.min(dim=1).values
        min_other_q2 = dist_all_q2.min(dim=1).values

        correct_q1 = dist_ec_q1 < min_other_q1
        correct_q2 = dist_ec_q2 < min_other_q2

        for i, (a, b, c) in enumerate(zip(label_a, label_b, label_c)):
            is_add = (c == a + b)
            is_mul = (c == (a * b) % 21)
            if is_add and is_mul:
                continue
            if not is_add and not is_mul:
                continue
            if i not in pos_to_valid:
                continue
            j = pos_to_valid[i]
            if is_add:
                add_total += 1
                if correct_q1[j].item():
                    add_correct_q0 += 1
                if correct_q2[j].item():
                    add_correct_q1 += 1
                if correct_q1[j].item() or correct_q2[j].item():
                    add_correct += 1
            elif is_mul:
                mul_total += 1
                if correct_q1[j].item():
                    mul_correct_q0 += 1
                if correct_q2[j].item():
                    mul_correct_q1 += 1
                if correct_q1[j].item() or correct_q2[j].item():
                    mul_correct += 1

        add_acc_q0 = add_correct_q0 / add_total if add_total > 0 else 0.0
        add_acc_q1 = add_correct_q1 / add_total if add_total > 0 else 0.0
        mul_acc_q0 = mul_correct_q0 / mul_total if mul_total > 0 else 0.0
        mul_acc_q1 = mul_correct_q1 / mul_total if mul_total > 0 else 0.0
        add_acc = add_correct / add_total if add_total > 0 else 0.0
        mul_acc = mul_correct / mul_total if mul_total > 0 else 0.0
        return {
            'add_acc_q0': add_acc_q0,
            'add_acc_q1': add_acc_q1,
            'mul_acc_q0': mul_acc_q0,
            'mul_acc_q1': mul_acc_q1,
            'add_acc': add_acc,
            'mul_acc': mul_acc,
            'add_total': add_total,
            'mul_total': mul_total,
        }
