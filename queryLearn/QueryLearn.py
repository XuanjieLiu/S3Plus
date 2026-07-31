import torch
from torch import optim
import torch.nn as nn
import os
import sys
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from VQ.VQVAE import VQVAE, split_into_three
from VQ.common_func import load_config_from_exp_name
from VQ.eval_common import CommonEvaler
from shared import DEVICE
from utils import init_dataloaders
from loss_counter import LossCounter
from VQ.common_func import parse_label
from dataloader import load_enc_eval_data
from queryLearn.opernet import OperNet, comb_q_z
from queryLearn.training_helpers import (
    init_queries,
    operation_loss,
    regul_sample,
    resolve_operation_loss_config,
    sanity_check_oper_loss,
)
from queryLearn.pair_diagnostics import PairDiagnosticsRecorder
from queryLearn.label_codebook import (
    DEFAULT_LABEL_CODEBOOK_CODES,
    load_label_codebook,
)
try:
    from queryLearn.query_vis import record_dir, save_query_operation_tables
except ImportError:
    from query_vis import record_dir, save_query_operation_tables
try:
    from queryLearn.record_visualizer import RecordVisualizer, best_record_metric
except ImportError:
    from record_visualizer import RecordVisualizer, best_record_metric


VQSPS_EXP_ROOT = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../VQ/exp/')
STAGE_TRAIN = 'train'
STAGE_VAL = 'val'
EVAL_TERMS = [
    'oper_loss',
    'symm_loss',
    'total_loss',
    'add_acc_q1',
    'add_acc_q2',
    'mm21_acc_q1',
    'mm21_acc_q2',
    'add_acc',
    'mm21_acc',
    'add_total',
    'mm21_total',
    'q_l2_dist',
]


def load_VQSPS_loader(config):
    vqsps_exp_name = config['VQSPS']['EXP_NAME']
    vqsps_config = load_config_from_exp_name(vqsps_exp_name)
    model_path = os.path.join(VQSPS_EXP_ROOT, vqsps_exp_name, config['VQSPS']['CHECK_POINT_NAME'])
    return CommonEvaler(vqsps_config, model_path), vqsps_config


class QueryLearn:
    def __init__(self, config, model_path=None, loaded_model: VQVAE = None):
        self.config = config
        self._exp_dir = config.get('_exp_dir', None)
        self.sps_model, sps_config = load_VQSPS_loader(config)
        self._set_sps_untrainable()
        self.train_loader, self.eval_loader, self.single_img_eval_loader = init_dataloaders(config)
        self.use_eval_set = self.eval_loader is not None
        query_config = config.get('query_learner', {})
        self.query_dim = query_config.get('query_dim', query_config.get('in_dim', 8))
        self.train_queries = query_config.get('train_queries', False)
        queries = init_queries(query_config, self.query_dim)
        self.queries = nn.Parameter(queries) if self.train_queries else queries
        operator_config = config['operator']
        operator_condition_mode = operator_config.get('condition_mode', 'concat')
        self.use_label_codebook = operator_config.get('use_label_codebook', False)
        self.label_codebook = None
        vq_layer = self.sps_model.model.vq_layer
        if self.use_label_codebook:
            label_codes = operator_config.get('label_codebook_codes', DEFAULT_LABEL_CODEBOOK_CODES)
            fail_fast = operator_config.get('label_codebook_fail_fast', True)
            self.label_codebook = load_label_codebook(
                self.single_img_eval_loader,
                self.sps_model,
                codes=label_codes,
                fail_fast=fail_fast,
            )
            vq_layer = self.label_codebook.vq_layer
        self.oper_net = OperNet(
            in_dim=self.sps_model.latent_code_1 * 2 + self.query_dim,
            out_dim=self.sps_model.latent_code_1,
            n_hidden_layers=operator_config['n_hidden_layers'],
            unit=operator_config['unit'],
            vq_layer=vq_layer,
            train_vq=False,
            condition_mode=operator_condition_mode,
            pair_dim=self.sps_model.latent_code_1 * 2,
            query_dim=self.query_dim,
            film_init_identity=operator_config.get('film_init_identity', True),
        ).to(DEVICE)
        self.train_result_path = config['train_result_path']
        self.eval_result_path = config['eval_result_path']
        self.train_record_path = config['train_record_path']
        self.eval_record_path = config['eval_record_path']
        self.critical_pair_record_path = config.get('critical_pair_record_path', 'CriticalPairStats_record.csv')
        self.pair_risk_record_path = config.get('pair_risk_record_path', 'PairRiskStats_record.csv')
        self.log_interval = config['log_interval']
        self.eval_interval = config['eval_interval']
        self.checkpoint_interval = config.get('checkpoint_interval', self.log_interval)
        self.checkpoint_after = config.get('checkpoint_after', 0)
        self.max_iter_num = config['max_iter_num']
        self.model_path = config['model_path']
        self.best_model_path = config.get('best_model_path', 'best_model.pt')
        self.best_checkpoint_metric = config.get('best_checkpoint_metric', 'total_loss')
        self.best_checkpoint_value = float('inf')
        self.is_symm = config.get('is_symm', False)
        self.is_assoc = config.get('is_assoc', False)
        self.eqLoss_scalar = config.get('eqLoss_scalar', 0.05)
        self.symm_loss_scalar = config.get('symm_loss_scalar', 0.01)
        self.num_z_c = None
        self.num_labels = None
        self._num_label_to_index = None
        self.mean_mse = nn.MSELoss(reduction='mean')
        self.sanity_check = config.get('sanity_check', False)
        self.operation_loss_config = resolve_operation_loss_config(
            config.get('operation_loss', None),
            self.sps_model.latent_code_1,
        )
        self.eval_terms = list(EVAL_TERMS)
        self.record_hard_min_loss = (
            self.operation_loss_config['type'] == 'gaussian_mixture_nll'
        )
        if self.record_hard_min_loss:
            self.eval_terms.insert(1, 'hard_min_loss')
        self.query_vis_format = config.get('query_vis_format', 'png').lower().lstrip('.')
        self.grad_clip_norm = config.get('grad_clip_norm', None)
        self.record_visualizer = RecordVisualizer(
            config,
            self.train_record_path,
            self.eval_record_path,
        )
        self.pair_diagnostics = PairDiagnosticsRecorder(
            self.train_loader.dataset,
            self.critical_pair_record_path,
            self.pair_risk_record_path,
        )
        print(f"Query dim: {self.query_dim}, OperNet condition mode: {operator_condition_mode}")
        if self.use_label_codebook:
            print(
                f"Label codebook enabled: codes={self.label_codebook.codes}, "
                f"num_codes={len(self.label_codebook.codes)}"
            )
            mapping = {
                label: code_idx
                for label, code_idx in zip(
                    self.label_codebook.codes,
                    self.label_codebook.original_code_indices,
                )
            }
            print(f"Label -> original SPS code index: {mapping}")
        if not self.use_eval_set:
            print("Pair evaluation is disabled or eval set is empty")
        print(self.pair_diagnostics.summary_text())
        if self.sanity_check:
            print('Sanity check mode...')
        print(f"Operation loss: {self.operation_loss_config}")
        print(
            f"Checkpoint policy: latest={self.model_path}, "
            f"best={self.best_model_path} by train/{self.best_checkpoint_metric}"
        )

    def _ensure_num_z_c(self):
        if self.num_z_c is not None and self.num_labels is not None and self._num_label_to_index is not None:
            return
        if self.label_codebook is not None:
            self.num_z_c = self.label_codebook.num_z_c
            self.num_labels = self.label_codebook.num_labels
            self._num_label_to_index = self.label_codebook.label_to_index
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

    def _set_sps_untrainable(self):
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
        e_q_loss = torch.zeros(1)[0].to(DEVICE)
        if self.is_symm:
            assoc_plus_loss += self.mean_mse(e_abc_1, e_acb_1) * self.symm_loss_scalar
            assoc_plus_loss += self.mean_mse(e_abc_2, e_bac_2) * self.symm_loss_scalar
            e_q_loss += e_q_loss_acb_1
            e_q_loss += e_q_loss_bac_2
        if self.is_assoc:
            assoc_plus_loss += self.mean_mse(e_abc_1, e_abc_2) * self.symm_loss_scalar
            e_q_loss += e_q_loss_abc_2
        if self.is_symm or self.is_assoc:
            e_q_loss += e_q_loss_ab + e_q_loss_abc_1
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
        epoch_hard_min_losses = []
        epoch_symm_losses = []
        epoch_total_losses = []
        epoch_label_a = []
        epoch_label_b = []
        epoch_label_c = []
        epoch_q1_out = []
        epoch_q2_out = []
        epoch_ec = []
        pair_diagnostic_epoch_stats = (
            self.pair_diagnostics.new_epoch_stats()
            if stage == STAGE_TRAIN
            else None
        )
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
            q1 = self.queries[0]
            q2 = self.queries[1]
            q1_in = comb_q_z(ea, eb, q1)
            q2_in = comb_q_z(ea, eb, q2)
            e_q1_out, eq_loss_q1, z_q1_out = self.oper_net(q1_in)
            e_q2_out, eq_loss_q2, z_q2_out = self.oper_net(q2_in)

            # Per-sample MSE decides query assignment; VQ loss is batch-level regularization.
            per_loss_q1 = (e_q1_out - ec).pow(2).mean(dim=-1)
            per_loss_q2 = (e_q2_out - ec).pow(2).mean(dim=-1)
            query_eq_loss = (eq_loss_q1 + eq_loss_q2) * self.eqLoss_scalar
            self.pair_diagnostics.update_epoch_stats(
                pair_diagnostic_epoch_stats,
                per_loss_q1,
                per_loss_q2,
                label_a,
                label_b,
                label_c,
            )

            if self.sanity_check:
                oper_loss = sanity_check_oper_loss(per_loss_q1, per_loss_q2, label_a, label_b, label_c)
                hard_min_loss = torch.minimum(per_loss_q1, per_loss_q2).mean()
            else:
                oper_loss, hard_min_loss = operation_loss(
                    per_loss_q1,
                    per_loss_q2,
                    self.operation_loss_config,
                )

            # symm loss
            q1_symm_loss = self.symm_loss(*regul_sample(e_content), q1)
            q2_symm_loss = self.symm_loss(*regul_sample(e_content), q2)
            symm_loss = q1_symm_loss + q2_symm_loss

            total_loss = oper_loss + query_eq_loss + symm_loss

            if loss_counter is not None or save_query_vis:
                epoch_oper_losses.append(oper_loss.item())
                epoch_hard_min_losses.append(hard_min_loss.item())
                epoch_symm_losses.append(symm_loss.item())
                epoch_total_losses.append(total_loss.item())
                epoch_label_a.extend(label_a)
                epoch_label_b.extend(label_b)
                epoch_label_c.extend(label_c)
                epoch_q1_out.append(z_q1_out.detach().cpu())
                epoch_q2_out.append(z_q2_out.detach().cpu())
                epoch_ec.append(ec.detach().cpu())

            if optimizer is not None:
                total_loss.backward()
                if self.grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(self.oper_net.parameters(), self.grad_clip_norm)
                optimizer.step()

        self.pair_diagnostics.accumulate_epoch_stats(pair_diagnostic_epoch_stats)

        if (loss_counter is not None or save_query_vis) and epoch_oper_losses:
            q1_out_epoch = torch.cat(epoch_q1_out, dim=0).to(DEVICE)
            q2_out_epoch = torch.cat(epoch_q2_out, dim=0).to(DEVICE)
            ec_epoch = torch.cat(epoch_ec, dim=0).to(DEVICE)
            accu = self._batch_query_accu(
                epoch_label_a,
                epoch_label_b,
                epoch_label_c,
                q1_out_epoch,
                q2_out_epoch,
                ec_epoch,
            )
            if loss_counter is not None:
                q_l2_dist = torch.norm(self.queries[0] - self.queries[1], p=2).item()
                loss_values = [
                    sum(epoch_oper_losses) / len(epoch_oper_losses),
                ]
                if self.record_hard_min_loss:
                    loss_values.append(
                        sum(epoch_hard_min_losses) / len(epoch_hard_min_losses)
                    )
                loss_values.extend([
                    sum(epoch_symm_losses) / len(epoch_symm_losses),
                    sum(epoch_total_losses) / len(epoch_total_losses),
                    accu['add_acc_q1'],
                    accu['add_acc_q2'],
                    accu['mm21_acc_q1'],
                    accu['mm21_acc_q2'],
                    accu['add_acc'],
                    accu['mm21_acc'],
                    accu['add_total'],
                    accu['mm21_total'],
                    q_l2_dist,
                ])
                loss_counter.add_values(loss_values)
            if save_query_vis:
                if query_vis_dir is None:
                    query_vis_dir = self.eval_result_path if stage == STAGE_VAL else self.train_result_path
                stage_name = 'eval' if stage == STAGE_VAL else stage
                q1_correct = self._query_target_correct(epoch_label_c, q1_out_epoch)
                q2_correct = self._query_target_correct(epoch_label_c, q2_out_epoch)
                q1_target_dist = (q1_out_epoch - ec_epoch).pow(2).sum(dim=-1)
                q2_target_dist = (q2_out_epoch - ec_epoch).pow(2).sum(dim=-1)
                save_query_operation_tables(
                    query_vis_dir,
                    stage_name,
                    epoch,
                    epoch_label_a,
                    epoch_label_b,
                    epoch_label_c,
                    q1_correct,
                    q2_correct,
                    q1_target_dist,
                    q2_target_dist,
                    self.query_vis_format,
                )

    def _resume_model(self):
        if os.path.exists(self.model_path):
            ckpt = torch.load(self.model_path, map_location=DEVICE)
            if isinstance(ckpt, dict) and 'oper_net_state_dict' in ckpt:
                self._load_oper_net_state_dict(ckpt['oper_net_state_dict'])
                if 'queries' in ckpt:
                    ckpt_queries = ckpt['queries'].to(DEVICE)
                    if ckpt_queries.shape != self.queries.shape:
                        raise ValueError(
                            f"Checkpoint query shape {tuple(ckpt_queries.shape)} does not match "
                            f"current query shape {tuple(self.queries.shape)}"
                        )
                    with torch.no_grad():
                        self.queries.copy_(ckpt_queries)
            else:
                self._load_oper_net_state_dict(ckpt)
            print(f"Model is loaded from {self.model_path}")
        else:
            print("No checkpoint found, training from scratch")

    def _load_oper_net_state_dict(self, state_dict):
        if self.label_codebook is None:
            self.oper_net.load_state_dict(state_dict)
            return

        filtered_state_dict = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith('vq_layer.')
        }
        missing, unexpected = self.oper_net.load_state_dict(filtered_state_dict, strict=False)
        unexpected = [key for key in unexpected if not key.startswith('vq_layer.')]
        missing = [key for key in missing if not key.startswith('vq_layer.')]
        if unexpected or missing:
            raise ValueError(
                "Checkpoint is incompatible with current label-codebook OperNet. "
                f"missing={missing}, unexpected={unexpected}"
            )
        print("Skipped checkpoint vq_layer weights because label codebook is rebuilt from labels")

    def _save_model(self, path, epoch, extra=None):
        ckpt = {
            'oper_net_state_dict': self.oper_net.state_dict(),
            'queries': self.queries.detach().cpu(),
            'epoch': epoch,
        }
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)

    def _init_best_checkpoint_value(self):
        if not os.path.exists(self.best_model_path):
            self.best_checkpoint_value = float('inf')
            return
        record_best = best_record_metric(self.train_record_path, self.best_checkpoint_metric)
        self.best_checkpoint_value = record_best if record_best is not None else float('inf')

    def _maybe_save_logged_checkpoints(self, epoch, train_metric_values):
        metric_value = train_metric_values.get(self.best_checkpoint_metric, None)
        self._save_model(
            self.model_path,
            epoch,
            {
                'checkpoint_kind': 'latest',
                'best_checkpoint_metric': self.best_checkpoint_metric,
            },
        )
        if metric_value is None:
            return
        if metric_value < self.best_checkpoint_value:
            self.best_checkpoint_value = metric_value
            self._save_model(
                self.best_model_path,
                epoch,
                {
                    'checkpoint_kind': 'best',
                    'best_checkpoint_metric': self.best_checkpoint_metric,
                    'best_metric_value': metric_value,
                },
            )
            print(
                f"New best checkpoint: {self.best_model_path} "
                f"epoch={epoch} {self.best_checkpoint_metric}={metric_value:.6g}"
            )

    def train(self):
        os.makedirs(self.train_result_path, exist_ok=True)
        os.makedirs(self.eval_result_path, exist_ok=True)
        self.oper_net.train()
        train_loss_counter = LossCounter(self.eval_terms, record_path=self.train_record_path)
        eval_loss_counter = LossCounter(self.eval_terms, record_path=self.eval_record_path)
        optim_params = [p for p in self.oper_net.parameters() if p.requires_grad]
        if self.train_queries:
            optim_params.append(self.queries)
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
        self._init_best_checkpoint_value()
        for epoch in range(start_epoch, self.max_iter_num):
            print(f"Epoch {epoch}")
            is_log_epoch = epoch % self.log_interval == 0
            did_write_record = False
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
                self.pair_diagnostics.record_interval(epoch)
                train_values = train_loss_counter.calc_values_mean()
                train_metric_values = dict(zip(self.eval_terms, train_values))
                train_loss_counter.record_and_clear(num=epoch)
                self._maybe_save_logged_checkpoints(epoch, train_metric_values)
                did_write_record = True

            if self.use_eval_set and epoch % self.eval_interval == 0:
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
                did_write_record = True

            if did_write_record:
                self.record_visualizer.refresh()

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

    def _batch_query_accu(self, label_a, label_b, label_c, q1_out, q2_out, ec):
        self._ensure_num_z_c()
        label_c_idx = []
        valid_mask = []
        for a, b, c in zip(label_a, label_b, label_c):
            is_add = (c == a + b)
            is_mm21 = (c == (a * b) % 21)
            if is_add and is_mm21:
                valid_mask.append(0)
                label_c_idx.append(-1)
                continue
            valid_mask.append(1)
            label_c_idx.append(self._num_label_to_index.get(c, -1))

        label_c_idx = torch.tensor(label_c_idx, device=DEVICE)
        valid_mask = torch.tensor(valid_mask, device=DEVICE, dtype=torch.bool)
        valid_mask = valid_mask & (label_c_idx >= 0)

        add_total = 0
        add_correct_q1 = 0
        add_correct_q2 = 0
        add_correct = 0
        mm21_total = 0
        mm21_correct_q1 = 0
        mm21_correct_q2 = 0
        mm21_correct = 0

        if not valid_mask.any():
            return {
                'add_acc_q1': 0.0,
                'add_acc_q2': 0.0,
                'mm21_acc_q1': 0.0,
                'mm21_acc_q2': 0.0,
                'add_acc': 0.0,
                'mm21_acc': 0.0,
                'add_total': 0,
                'mm21_total': 0,
            }

        ec_valid = ec[valid_mask]
        q1_valid = q1_out[valid_mask]
        q2_valid = q2_out[valid_mask]
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
            is_mm21 = (c == (a * b) % 21)
            if is_add and is_mm21:
                continue
            if not is_add and not is_mm21:
                continue
            if i not in pos_to_valid:
                continue
            j = pos_to_valid[i]
            if is_add:
                add_total += 1
                if correct_q1[j].item():
                    add_correct_q1 += 1
                if correct_q2[j].item():
                    add_correct_q2 += 1
                if correct_q1[j].item() or correct_q2[j].item():
                    add_correct += 1
            elif is_mm21:
                mm21_total += 1
                if correct_q1[j].item():
                    mm21_correct_q1 += 1
                if correct_q2[j].item():
                    mm21_correct_q2 += 1
                if correct_q1[j].item() or correct_q2[j].item():
                    mm21_correct += 1

        add_acc_q1 = add_correct_q1 / add_total if add_total > 0 else 0.0
        add_acc_q2 = add_correct_q2 / add_total if add_total > 0 else 0.0
        mm21_acc_q1 = mm21_correct_q1 / mm21_total if mm21_total > 0 else 0.0
        mm21_acc_q2 = mm21_correct_q2 / mm21_total if mm21_total > 0 else 0.0
        add_acc = add_correct / add_total if add_total > 0 else 0.0
        mm21_acc = mm21_correct / mm21_total if mm21_total > 0 else 0.0
        return {
            'add_acc_q1': add_acc_q1,
            'add_acc_q2': add_acc_q2,
            'mm21_acc_q1': mm21_acc_q1,
            'mm21_acc_q2': mm21_acc_q2,
            'add_acc': add_acc,
            'mm21_acc': mm21_acc,
            'add_total': add_total,
            'mm21_total': mm21_total,
        }
