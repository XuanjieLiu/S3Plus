import os
import random
import sys
from typing import Any, Callable, Dict, List

from VQ.plot_multistyle_zc import MultiStyleZcEvaler
from eval_plus_nd import VQvaePlusEval, plot_plusZ_against_label, calc_ks_enc_plus_z, calc_multi_emb_plus_accu, \
    calc_one2one_plus_accu, calc_plus_z_self_cycle_consistency, calc_plus_z_mode_emb_label_cycle_consistency, \
    calc_emb_select_plus_accu

sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.utils import save_image
import torch.optim as optim
from dataloader_plus import MultiImgDataset
from dataloader import SingleImgDataset, load_enc_eval_data, load_enc_eval_data_with_style
from loss_counter import LossCounter, RECORD_PATH_DEFAULT
from VQVAE import VQVAE
from shared import *
from two_dim_num_vis import plot_z_against_label, MumEval
from visual_imgs import VisImgs
from eval_common import EvalHelper
from common_func import make_dataset_trans, solve_label_emb_one2one_matching
from eval.dec_vis_eval_2digit import plot_dec_img
import torch.nn.functional as F


def make_translation_batch(batch_size, dim=np.array([1, 0, 1]), is_std_normal=False, t_range=(-3, 3)):
    scale = t_range[1] - t_range[0]
    if is_std_normal:
        T_mat = torch.randn(batch_size, len(dim))
    else:
        T_mat = torch.rand(batch_size, len(dim)) * scale + t_range[0]
    T = T_mat.mul(torch.from_numpy(dim)).to(DEVICE)
    T_R = -T
    return T, T_R


def is_need_train(train_config):
    loss_counter = LossCounter([])
    iter_num = loss_counter.load_iter_num(train_config['train_record_path'])
    if train_config['max_iter_num'] > iter_num:
        print("Continue training")
        return True
    else:
        print("No more training is needed")
        return False


def iter_add(a, b):
    return [a[i] + b[i] for i in range(0, len(a))]


def split_into_three(tensor, not_divisible_ok=False):
    group_size = int(tensor.size(0) / 3)
    sizes = [3, group_size, *tensor.size()[1:]]
    if not_divisible_ok:
        new_tensor = tensor[0:group_size*3].reshape(*sizes)
    else:
        new_tensor = tensor.reshape(*sizes)
    return new_tensor[0], new_tensor[1], new_tensor[2]


def switch_digital(a_con: torch.Tensor, b_con: torch.Tensor, emb_dim: int):
    dig_num = int(a_con.size(1) / emb_dim)
    batch_size = a_con.size(0)
    sub_batch = int(batch_size / (dig_num + 1))
    a_list = []
    b_list = []
    for i in range(0, dig_num):
        a_mask_list = []
        for j in range(0, dig_num):
            if j == i:
                a_mask_list.append(torch.ones(emb_dim).to(DEVICE))
            else:
                a_mask_list.append(torch.zeros(emb_dim).to(DEVICE))
        a_mask = torch.concat(a_mask_list, dim=0)
        b_mask = torch.ones(a_con.size(1)).to(DEVICE) - a_mask
        a_slice = a_con[i*sub_batch:(i+1)*sub_batch, ...]
        b_slice = b_con[i * sub_batch:(i + 1) * sub_batch, ...]
        a_list.append(a_slice.mul(a_mask) + b_slice.mul(b_mask))
        b_list.append(a_slice.mul(b_mask) + b_slice.mul(a_mask))
    a_list.append(a_con[dig_num * sub_batch:, ...])
    b_list.append(b_con[dig_num * sub_batch:, ...])
    a_switch = torch.concat(a_list)
    b_switch = torch.concat(b_list)
    return a_switch, b_switch


def init_dataloaders(config):
    aug_t = config.get('augment_times', 1)
    blur_cfg = config.get('blur_config', None)
    is_blur = config.get('is_blur', False)
    trans = make_dataset_trans(is_blur, blur_cfg) if is_blur else None
    n_workers = config.get('num_workers', 0)
    loader_config = {
        'batch_size': config['batch_size'],
        'shuffle': True,
        'num_workers': n_workers,
        'persistent_workers': True if n_workers > 0 else False,
    }
    # 新增：从 config 读取可选的随机种子
    split_seed = config.get('random_split_seed', None)
    split_generator = None
    if split_seed is not None:
        # 只用于 random_split，本身不影响全局 RNG
        split_generator = torch.Generator()
        # 建议显式转 int，避免从 JSON 读取成 str 报错
        split_generator.manual_seed(int(split_seed))
    if config.get('is_random_split_data', False):
        print("Using random split data")
        train_ratio = config['train_data_ratio']
        dataset_all = make_dataset(config['train_data_path'], MultiImgDataset, transform=trans, augment_times=aug_t)
        train_size = int(len(dataset_all) * train_ratio)
        eval_size = len(dataset_all) - train_size
        # 仅当提供了 split_seed 时，才把 generator 传进去；否则保持原行为
        if split_generator is None:
            train_dataset, eval_dataset = torch.utils.data.random_split(dataset_all, [train_size, eval_size])
        else:
            train_dataset, eval_dataset = torch.utils.data.random_split(
                dataset_all, [train_size, eval_size], generator=split_generator
            )
        plus_train_loader = DataLoader(train_dataset, **loader_config)
        plus_eval_loader = DataLoader(eval_dataset, **loader_config)
    else:
        print("Using predefined datasets")
        plus_train_set = make_dataset(config['train_data_path'], MultiImgDataset, transform=trans, augment_times=aug_t)
        plus_eval_set = make_dataset(config['plus_eval_set_path'], MultiImgDataset, transform=trans, augment_times=aug_t)
        plus_train_loader = DataLoader(plus_train_set, **loader_config)
        plus_eval_loader = DataLoader(plus_eval_set, **loader_config)
    if config.get('single_img_eval_set_path', None) is not None:
        single_img_eval_set = make_dataset(config['single_img_eval_set_path'], SingleImgDataset, transform=trans, augment_times=aug_t)
        single_img_eval_loader = DataLoader(single_img_eval_set, **loader_config)
    else:
        single_img_eval_loader = None
    return plus_train_loader, plus_eval_loader, single_img_eval_loader


def make_dataset(data_path: str | List[str], dataset_cls: Callable[..., Any],
                     transform: Any = None, augment_times: int = 1) -> ConcatDataset:
    print(f"data_path is: {data_path}")
    path_list = data_path if isinstance(data_path, list) else [data_path]
    print(f"Making dataset from paths: {path_list}")
    # 如果有多个 set_path, 创建多个 dataset 并合并
    datasets = ConcatDataset([dataset_cls(path, augment_times=augment_times, transform=transform)
                              for path in path_list])
    return datasets


class PlusTrainer:
    def __init__(self, config, is_train=True):
        self.config = config
        self.batch_size = config['batch_size']
        self.train_loader, self.plus_eval_loader, self.single_img_eval_loader = init_dataloaders(config)
        self.min_loss_scalar = config['min_loss_scalar']
        self.model = VQVAE(config).to(DEVICE)
        self.train_result_path = config['train_result_path']
        self.eval_result_path = config['eval_result_path']
        self.train_record_path = config['train_record_path']
        self.eval_record_path = config['eval_record_path']
        self.plus_accu_record_path = config['plus_accu_record_path']
        self.model_path = config['model_path']
        self.learning_rate = config['learning_rate']
        self.scheduler_base_num = config['scheduler_base_num']
        self.is_save_img = config['is_save_img']
        self.log_interval = config['log_interval']
        self.checkpoint_interval = config['checkpoint_interval']
        self.max_iter_num = config['max_iter_num']
        self.embedding_dim = config['embedding_dim']
        self.multi_num_embeddings = config['multi_num_embeddings']
        if self.multi_num_embeddings is None:
            self.latent_embedding_1 = config['latent_embedding_1']
        else:
            self.latent_embedding_1 = len(self.multi_num_embeddings)
        self.latent_code_1 = self.latent_embedding_1 * self.embedding_dim
        self.sub_batch = int(self.batch_size / 3)
        self.sum_mse = nn.MSELoss(reduction='sum')
        self.mean_mse = nn.MSELoss(reduction='mean')
        self.z_plus_loss_scalar = config['z_plus_loss_scalar']
        self.commutative_z_loss_scalar = config['commutative_z_loss_scalar']
        self.associative_z_loss_scalar = config['associative_z_loss_scalar']
        self.K = config['K']
        self.assoc_aug_range = config['assoc_aug_range']
        self.eval_help = EvalHelper(config)
        self.VQPlus_eqLoss_scalar = config['VQPlus_eqLoss_scalar']
        self.plus_recon_loss_scalar = config['plus_recon_loss_scalar']
        self.plus_by_embedding = config['plus_by_embedding']
        self.plus_by_zcode = config['plus_by_zcode']
        self.isVQStyle = config['isVQStyle']
        self.is_commutative_train = config['is_commutative_train']
        self.embeddings_num = config['embeddings_num']
        self.plus_mse_scalar = config['plus_mse_scalar']
        self.is_switch_digital = config['is_switch_digital']
        self.is_assoc_within_batch = config['is_assoc_within_batch']
        self.is_plot_zc_value = config['is_plot_zc_value']
        self.is_plot_vis_num = config['is_plot_vis_num']
        self.is_symm_assoc = config['is_symm_assoc']
        self.is_pure_assoc = config['is_pure_assoc']
        self.is_commutative_all = config['is_commutative_all']
        self.is_full_symm = config['is_full_symm']
        self.is_twice_oper = config['is_twice_oper']
        self.plus_l1_weight = config.get('plus_l1_weight', 0.0)
        self.plus_l1_include_bias = config.get('plus_l1_include_bias', False)
        self.use_v3_loss = config.get('use_v3_loss', False)
        self.disable_rand_style_sample = config.get('disable_rand_style_sample', False)
        self.checkpoint_after = config.get('checkpoint_after', 0)
        self.is_plot_num_line = config.get('is_plot_num_line', True)

    def resume(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(self.model.load_tensor(self.model_path))
            print(f"Model is loaded")
        else:
            print("New model is initialized")

    def scheduler_func(self, curr_iter):
        return self.scheduler_base_num ** curr_iter

    def one_epoch(self, epoch_num, loss_counter: LossCounter, data_loader,
                  is_log, vis_imgs: VisImgs, optimizer: torch.optim.Optimizer = None):
        print(f'Epoch: {epoch_num}')
        for batch_ndx, sample in enumerate(data_loader):
            if optimizer is not None:
                optimizer.zero_grad()
            data, labels = sample
            sizes = data[0].size()
            data_all = torch.stack(data, dim=0).reshape(3 * sizes[0], sizes[1], sizes[2], sizes[3])
            data_all = data_all.to(DEVICE, non_blocking=True)
            data = split_into_three(data_all)
            e_all, e_q_loss, z_all = self.model.batch_encode_to_z(data_all)
            e_content = e_all[..., 0:self.latent_code_1]
            z_content = z_all[..., 0:self.latent_code_1]

            recon = self.model.batch_decode_from_z(e_all)
            recon_a, recon_b, recon_c = split_into_three(recon)
            vis_imgs.gt_a, vis_imgs.gt_b, vis_imgs.gt_c = data[0][0], data[1][0], data[2][0]
            vis_imgs.recon_a, vis_imgs.recon_b, vis_imgs.recon_c = recon_a[0], recon_b[0], recon_c[0]
            vae_loss = self.vae_loss(data_all, recon)

            plus_loss = torch.zeros(2).to(DEVICE)
            if self.plus_recon_loss_scalar > self.min_loss_scalar:
                if self.plus_by_embedding:
                    plus_loss = iter_add(plus_loss, self.bi_plus_loss(e_all, data[2], vis_imgs))
                if self.plus_by_zcode:
                    plus_loss = iter_add(plus_loss, self.bi_plus_loss(z_all, data[2], vis_imgs))

            operations_loss = torch.zeros(1)[0].to(DEVICE)
            if self.plus_by_embedding:
                operations_loss += self.operation_loss_z(e_content)
            if self.plus_by_zcode:
                operations_loss += self.operation_loss_z(z_content)
            
            # V3 loss calculation
            e_style = e_all[..., self.latent_code_1:]
            v3_loss = cal_v3_loss(e_content, z_content, e_style) if self.use_v3_loss else {
                "content_loss": torch.zeros(1)[0].to(DEVICE),
                "style_loss": torch.zeros(1)[0].to(DEVICE),
                "sample_loss": torch.zeros(1)[0].to(DEVICE),
                "fragment_loss": torch.zeros(1)[0].to(DEVICE),
            }

            loss = self.loss_func(vae_loss, e_q_loss, plus_loss, operations_loss, v3_loss, loss_counter)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

        if is_log:
            print(loss_counter.make_record(epoch_num))
            loss_counter.record_and_clear(RECORD_PATH_DEFAULT, epoch_num)
            if optimizer is not None:
                self.model.save_tensor(self.model.state_dict(), self.model_path)
            if self.is_save_img:
                vis_imgs.save_img(f'{epoch_num}.png')

    def train(self, is_resume=True, optimizer: torch.optim.Optimizer = None, eval_func: callable = None):
        os.makedirs(self.train_result_path, exist_ok=True)
        os.makedirs(self.eval_result_path, exist_ok=True)
        self.model.train()
        if is_resume:
            self.resume()
        loss_counter_keys = ['loss_ED', 'VQ_C', 'plus_recon', 'plus_z', 'loss_oper', "content_loss", "style_loss", "sample_loss", "fragment_loss"]
        train_loss_counter = LossCounter(loss_counter_keys, self.train_record_path)
        eval_loss_counter = LossCounter(loss_counter_keys, self.eval_record_path)
        eval_keys = ['emb_select_accu', 'emb_select_accu_cycle', 'one2n_accu', 'one2n_accu_cycle', 'one2one_accu', 'one2one_accu_cycle',
                'emb_self_consistency', 'emb_label_consistency', 'z_c_recognition_rate', 'z_c_cycle_recognition_rate']
        eval_loss_counter_keys = ([f'train_{key}' for key in eval_keys] + [f'eval_{key}' for key in eval_keys] +
                                  ['one2n_match_rate', 'one2one_matching_rate', 'nna_score'])
        special_loss_counter = LossCounter(eval_loss_counter_keys, self.plus_accu_record_path)
        start_epoch = train_loss_counter.load_iter_num(self.train_record_path)
        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: self.scheduler_func(start_epoch))
        for epoch_num in range(start_epoch, self.max_iter_num):
            if eval_func is not None:
                eval_func(epoch_num, self.model)
            is_log = (epoch_num % self.log_interval == 0)
            train_vis_img = VisImgs(self.train_result_path)
            eval_vis_img = VisImgs(self.eval_result_path)

            # Evaluation phase
            if is_log:
                self.model.eval()
                print('Eval phase')
                with torch.no_grad():
                    self.one_epoch(epoch_num, eval_loss_counter, self.plus_eval_loader, True, eval_vis_img, None)
                self.model.train()
                if self.is_save_img:
                    with torch.no_grad():
                        # self.plot_enc_z(epoch_num, self.single_img_eval_loader)
                        single_img_results = self.single_img_eval(epoch_num)
                        train_plus_results = self.overall_plus_eval(epoch_num, self.train_loader, self.train_result_path)
                        eval_plus_results = self.overall_plus_eval(epoch_num, self.plus_eval_loader, self.eval_result_path)
                        loss_values = train_plus_results + eval_plus_results + single_img_results
                        special_loss_counter.add_values(loss_values)
                        special_loss_counter.record_and_clear(RECORD_PATH_DEFAULT, epoch_num)
                        if self.is_plot_zc_value:
                            self.plot_plus_z(epoch_num, self.plus_eval_loader, self.eval_result_path, 'plus_z')

            # Training phase
            self.one_epoch(epoch_num, train_loss_counter, self.train_loader, is_log, train_vis_img, optimizer)

            if epoch_num % self.checkpoint_interval == 0 and epoch_num != 0 and epoch_num >= self.checkpoint_after:
                self.model.save_tensor(self.model.state_dict(), f'checkpoint_{epoch_num}.pt')
        print("train ends")

    def plot_enc_z(self, epoch_num, data_loader):
        num_z, num_labels = load_enc_eval_data(
                                data_loader,
                                lambda x: self.model.find_indices(
                                    self.model.batch_encode_to_z(x)[0],
                                    True, True
                                )
        )
        z_eval_path = os.path.join(self.train_result_path, f'{epoch_num}_z.png')
        plot_z_against_label(num_z, num_labels, z_eval_path, self.eval_help)

        if self.latent_embedding_1 == 2 and self.is_plot_vis_num:
            vis_eval_path = os.path.join(self.eval_result_path, f'{epoch_num}_numVis.png')
            enc_flat_z = [int(t.item()) for t in num_z]
            plot_dec_img(
                loaded_model=self.model,
                dict_size=self.embeddings_num,
                digit_num=2,
                save_path=vis_eval_path,
                enc_flat_z=enc_flat_z,
                enc_labels=num_labels,
                dict_sizes=self.multi_num_embeddings
            )

    def calc_plus_accu(self, data_loader):
        plus_eval = VQvaePlusEval(self.config, loaded_model=self.model)
        all_enc_z, all_plus_z = plus_eval.load_plusZ_eval_data(data_loader)
        ks, accu = calc_ks_enc_plus_z(all_enc_z, all_plus_z)
        return ks, accu

    def overall_plus_eval(self, epoch_num, data_loader, result_path):
        plus_eval = VQvaePlusEval(self.config, loaded_model=self.model)
        all_enc_z, all_plus_z = plus_eval.load_plusZ_eval_data(data_loader)
        emb_select_accu, emb_select_accu_cycle, _, _ = calc_emb_select_plus_accu(all_enc_z, all_plus_z)
        one2n_accu, one2n_accu_cycle = calc_multi_emb_plus_accu(all_enc_z, all_plus_z)
        one2one_accu, one2one_accu_cycle = calc_one2one_plus_accu(all_enc_z, all_plus_z)
        emb_self_consistency = calc_plus_z_self_cycle_consistency(all_plus_z)
        emb_label_consistency, z_c_recognition_rate, z_c_cycle_recognition_rate = (
            calc_plus_z_mode_emb_label_cycle_consistency(all_enc_z, all_plus_z))
        eval_path = os.path.join(result_path, f'{epoch_num}_accu_{one2n_accu_cycle}')
        plot_plusZ_against_label(all_enc_z, all_plus_z, eval_path,
                                 is_scatter_lines=True, y_label="Emb idx")
        return [emb_select_accu, emb_select_accu_cycle, one2n_accu, one2n_accu_cycle, one2one_accu, one2one_accu_cycle,
                emb_self_consistency, emb_label_consistency, z_c_recognition_rate, z_c_cycle_recognition_rate]

    def single_img_eval(self, epoch_num):
        mr_evaler = MultiStyleZcEvaler(self.config, loaded_model=self.model)
        num_z, num_labels, colors, shapes = load_enc_eval_data_with_style(
            self.single_img_eval_loader,
            lambda x: mr_evaler.model.find_indices(
                mr_evaler.model.batch_encode_to_z(x)[0],
                True, False
            )
        )
        num_emb_idx = [x[0] for x in num_z.detach().cpu().numpy()]
        one2n_match_rate = mr_evaler.calc_emb_matching_score(num_emb_idx, num_labels)
        one2one_matching_rate = solve_label_emb_one2one_matching(num_emb_idx, num_labels)[1]

        orderliness_evaler = MumEval(self.config, loaded_model=self.model)
        save_path = os.path.join(self.train_result_path, f'{epoch_num}')
        nna_score = orderliness_evaler.num_eval_two_dim(self.single_img_eval_loader, save_path, is_draw_graph=self.is_plot_num_line)
        return [one2n_match_rate, one2one_matching_rate, nna_score]

    def plot_plus_z(self, epoch_num, data_loader, result_path, result_name="plus_z"):
        plus_eval = VQvaePlusEval(self.config, loaded_model=self.model)
        all_enc_z, all_plus_z = plus_eval.load_plusZ_eval_data(data_loader)
        n_row, n_col = self.eval_help.calc_subfigures_row_col()
        eval_path = os.path.join(result_path, f'{epoch_num}_{result_name}')
        plot_plusZ_against_label(all_enc_z, all_plus_z, eval_path, n_row, n_col,
                                 is_gird=True, sub_title='Emb dim', y_label="Dim value")

    def bi_plus_loss(self, z_all, imgs_c, vis_imgs: VisImgs = None):
        za, zb, zc = split_into_three(z_all)
        plus_loss1 = self.plus_loss(za, zb, zc, imgs_c, vis_imgs)
        plus_loss2 = self.plus_loss(zb, za, zc, imgs_c) if self.is_commutative_train else torch.zeros(2)
        plus_loss = iter_add(plus_loss1, plus_loss2)
        return plus_loss

    def plus_mse(self, a, b):
        if self.plus_mse_scalar < 0:
            return self.mean_mse(a, b)
        else:
            return self.mean_mse(a, b.detach()) * self.plus_mse_scalar + self.mean_mse(a.detach(), b)

    def plus_loss(self, za, zb, zc, imgs_c, vis_imgs: VisImgs = None):
        if self.disable_rand_style_sample:
            z_s = zc[..., self.latent_code_1:]
        else:
            z_s = za[..., self.latent_code_1:] if random.random() > 0.5 else zb[..., self.latent_code_1:]
        if self.isVQStyle:
            z_s = self.model.vq_layer(z_s)[0]
        za_content = za[..., 0:self.latent_code_1]
        zb_content = zb[..., 0:self.latent_code_1]
        if self.is_switch_digital:
            za_content, zb_content = switch_digital(za_content, zb_content, self.embedding_dim)
        e_ab_content, e_q_loss, z_ab_content = self.model.plus(za_content, zb_content)
        e_ab = torch.cat((e_ab_content, z_s), -1)
        z_ab = torch.cat((z_ab_content, z_s), -1)
        recon_c = self.model.batch_decode_from_z(e_ab)
        recon_loss = nn.MSELoss()(recon_c, imgs_c) * self.plus_recon_loss_scalar if bool(self.VQPlus_eqLoss_scalar) else 0
        if vis_imgs is not None:
            vis_imgs.plus_c = recon_c[0]
        if self.z_plus_loss_scalar > self.min_loss_scalar:
            if self.plus_by_embedding:
                z_loss = self.plus_mse(e_ab, zc) * self.z_plus_loss_scalar
            else:
                z_loss = self.plus_mse(z_ab, zc) * self.z_plus_loss_scalar
        else:
            z_loss = torch.zeros(1)[0]
        return recon_loss, z_loss + e_q_loss * self.VQPlus_eqLoss_scalar

    def commutative_z_loss(self, z_1, z_2):
        if self.is_commutative_all or self.is_twice_oper:
            z_all = torch.concat([z_1, z_2], dim=0)
            idx_1 = torch.randperm(z_all.size(0))
            z_perm = z_all[idx_1, ...]
            z_a = z_perm[:z_1.size(0), ...]
            z_b = z_perm[z_1.size(0):, ...]
        else:
            z_a, z_b = z_1, z_2
        e_ab, e_q_loss_ab, z_ab = self.model.plus(z_a, z_b)
        e_ba, e_q_loss_ba, z_ba = self.model.plus(z_b, z_a)
        if self.plus_by_embedding:
            loss = self.mean_mse(e_ab, e_ba) * self.commutative_z_loss_scalar
        else:
            loss = self.mean_mse(z_ab, z_ba) * self.commutative_z_loss_scalar
        # loss += self.VQPlus_eqLoss_scalar * (e_q_loss_ab + e_q_loss_ba)
        return loss

    def rand_associative_z(self, z_all_content):
        max_z = torch.max(z_all_content).item()
        min_z = torch.min(z_all_content).item()
        std_z = torch.std(z_all_content).item()
        assoc_aug_range = (min_z - std_z, max_z + std_z)
        trans_dim = np.ones(self.latent_code_1, dtype=int)
        z_a, _ = make_translation_batch(self.K, dim=trans_dim, t_range=assoc_aug_range)
        z_b, _ = make_translation_batch(self.K, dim=trans_dim, t_range=assoc_aug_range)
        z_c, _ = make_translation_batch(self.K, dim=trans_dim, t_range=assoc_aug_range)
        return z_a.to(DEVICE), z_b.to(DEVICE), z_c.to(DEVICE)

    def zc_based_associative_z(self, z_all_content):
        if self.is_assoc_within_batch and not self.is_twice_oper:
            za, zb, zc = split_into_three(z_all_content)
            z_all = torch.concat([za, zb, za, zb], dim=0)
        else:
            z_all = z_all_content
        idx_1 = torch.randperm(z_all.size(0))
        z_perm = z_all[idx_1, ...]
        z_slice = z_perm[:z_all_content.size(0), ...]
        z_a, z_b, z_c = split_into_three(z_slice)
        return z_a, z_b, z_c

    def associative_loss(self, z_a, z_b, z_c):
        e_ab, e_q_loss_ab, z_ab = self.model.plus(z_a, z_b)
        e_abc_1, e_q_loss_abc_1, z_abc_1 = self.model.plus(e_ab, z_c)
        # symm_assoc
        e_ac, e_q_loss_ac, z_ac = self.model.plus(z_a, z_c)
        e_acb_1, e_q_loss_acb_1, z_acb_1 = self.model.plus(e_ac, z_b)
        e_bac_2, e_q_loss_bac_2, z_bac_2 = self.model.plus(z_b, e_ac)
        # pure_assoc
        e_bc, e_q_loss_bc, z_bc = self.model.plus(z_b, z_c)
        e_abc_2, e_q_loss_abc_2, z_abc_2 = self.model.plus(z_a, e_bc)
        # choose loss
        assoc_plus_loss = torch.zeros(1)[0].to(DEVICE)
        if self.config['is_assoc_on_e']:
            if self.is_symm_assoc:
                assoc_plus_loss += self.mean_mse(e_abc_1, e_acb_1) * self.associative_z_loss_scalar
                if self.is_full_symm:
                    assoc_plus_loss += self.mean_mse(e_abc_2, e_bac_2) * self.associative_z_loss_scalar
            if self.is_pure_assoc:
                assoc_plus_loss += self.mean_mse(e_abc_1, e_abc_2) * self.associative_z_loss_scalar
        if self.config['is_assoc_on_z']:
            if self.is_symm_assoc:
                assoc_plus_loss += self.mean_mse(z_abc_1, z_acb_1) * self.associative_z_loss_scalar
                if self.is_full_symm:
                    assoc_plus_loss += self.mean_mse(z_abc_2, z_bac_2) * self.associative_z_loss_scalar
            if self.is_pure_assoc:
                assoc_plus_loss += self.mean_mse(z_abc_1, z_abc_2) * self.associative_z_loss_scalar
        e_q_loss = e_q_loss_ab + e_q_loss_abc_1
        if self.is_symm_assoc:
            e_q_loss += e_q_loss_acb_1
            if self.is_full_symm:
                e_q_loss += e_q_loss_bac_2
        if self.is_pure_assoc:
            e_q_loss += e_q_loss_abc_2
        return assoc_plus_loss + self.VQPlus_eqLoss_scalar * e_q_loss

    def operation_loss_z(self, z_all_content):
        if self.is_twice_oper:
            random_c = self.random_plus(z_all_content, z_all_content)
            z_contents = torch.cat([z_all_content, random_c], dim=0)
        else:
            z_contents = z_all_content
        za, zb, zc = split_into_three(z_contents)
        loss = torch.zeros(1)[0].to(DEVICE)
        if self.commutative_z_loss_scalar > self.min_loss_scalar and self.is_commutative_train:
            loss += self.commutative_z_loss(za, zb)
        if self.associative_z_loss_scalar > self.min_loss_scalar:
            if self.config['is_zc_based_assoc']:
                loss += self.associative_loss(*self.zc_based_associative_z(z_contents))
            if self.config['is_rand_z_assoc']:
                loss += self.associative_loss(*self.rand_associative_z(z_contents))
        return loss

    def random_plus(self, za, zb):
        assert za.size(0) == zb.size(0), f"za size {za.size(0)} != zb size {zb.size(0)}"
        z_all = torch.cat([za, zb], dim=0)
        idx_1 = torch.randperm(z_all.size(0))
        z_perm = z_all[idx_1, ...]
        z_a = z_perm[:za.size(0), ...]
        z_b = z_perm[za.size(0):, ...]
        e_c, e_q_loss_c, z_c = self.model.plus(z_a, z_b)
        if self.plus_by_embedding:
            return e_c
        else:
            return z_c

    def vae_loss(self, data, recon):
        recon_loss = nn.MSELoss(reduction='mean')(recon, data)
        return recon_loss

    def l1_on_plusnet_params(self):
        if self.plus_l1_weight <= 0:
            return torch.zeros(1, device=DEVICE)[0]
        l1 = torch.zeros(1, device=DEVICE)[0]
        for name, p in self.model.plus_net.named_parameters():
            if p is None or not p.requires_grad:
                continue
            if (not self.plus_l1_include_bias) and ('bias' in name):
                continue
            l1 = l1 + p.abs().sum()
        return self.plus_l1_weight * l1


    def loss_func(self, vae_loss, e_q_loss, plus_loss, operations_loss, v3_loss, loss_counter):
        xloss_ED = vae_loss
        plus_recon_loss, plus_z_loss = plus_loss
        loss = torch.zeros(1)[0].to(DEVICE)
        loss += xloss_ED + e_q_loss
        loss += plus_recon_loss + plus_z_loss
        loss += operations_loss
        loss += self.l1_on_plusnet_params()
        loss += sum(v3_loss.values())
        loss_counter.add_values([
            xloss_ED.item(),
            e_q_loss.item(),
            plus_recon_loss.item(),
            plus_z_loss.item(),
            operations_loss.item(),
            v3_loss["content_loss"].item(),
            v3_loss["style_loss"].item(),
            v3_loss["sample_loss"].item(),
            v3_loss["fragment_loss"].item(),
        ])
        return loss


def cal_v3_loss(z_content, e_content, z_style):
    z_a, z_b, z_c = split_into_three(z_content)
    e_a, e_b, e_c = split_into_three(e_content)
    z_s_a, z_s_b, z_s_c = split_into_three(z_style)
    v3_z_c = torch.stack([z_a, z_b, z_c], dim=1)
    v3_z_c_vq = torch.stack([e_a, e_b, e_c], dim=1)
    v3_z_s = torch.stack([z_s_a, z_s_b, z_s_c], dim=1)
    v3_losses = _compute_loss_pure(
        v3_z_c,
        v3_z_c_vq,
        v3_z_s,
        relativity=10,
        eps=1e-5
    )
    return v3_losses

def _compute_loss_pure(
        z_c,
        z_c_vq,
        z_s,
        relativity=10,
        eps=1e-5
    ):
        """
        Standard form of V3 loss

        Output: a dict of losses.
        z_c: (batch_size, n_fragments, d_emb_c)
        z_s: (batch_size, n_fragments, d_emb_s)
        vq_indices: (batch_size, n_fragments)
        vq_commit_loss: already computed in VQ
        y: (batch_size, n_fragments, W, H), of course it can be other shapes
        """

        # compute statistics
        content_frag_var = torch.mean(mpd(z_c) / 2 + mpd(z_c_vq) / 2, dim=0)
        content_samp_var = (
            mpd(torch.mean(z_c, dim=1)) / 2 + mpd(torch.mean(z_c_vq, dim=1)) / 2
        )
        style_frag_var = torch.mean(mpd(z_s), dim=0)
        style_samp_var = mpd(torch.mean(z_s, dim=1))

        # compute the losses using the relative variability difference
        r = relativity

        content_loss = F.relu(r - content_frag_var / (content_samp_var + eps)) / r
        style_loss = F.relu(r - style_samp_var / (style_frag_var + eps)) / r
        sample_loss = F.relu(r - style_samp_var / (content_samp_var + eps)) / r
        fragment_loss = F.relu(r - content_frag_var / (style_frag_var + eps)) / r

        losses = {
            "content_loss": content_loss,
            "style_loss": style_loss,
            "sample_loss": sample_loss,
            "fragment_loss": fragment_loss,
        }
        return losses


def mpd(x):
    """
    Mean pairwise distance
    x is a tensor of shape (b, n, d) or (n, d)
    """
    if len(x.shape) == 2:
        n, d = x.shape
        x1 = x.unsqueeze(0).expand(n, n, d)
        x2 = x.unsqueeze(1).expand(n, n, d)
        stack = torch.stack([x1, x2], dim=0)
        pairwise_d = torch.norm(stack[0] - stack[1], dim=-1)
        mpd = torch.sum(torch.sum(pairwise_d, dim=0), dim=0) / (n * (n - 1))
        return mpd  # scalar
    elif len(x.shape) == 3:
        b, n, d = x.shape
        x1 = x.unsqueeze(1).expand(b, n, n, d)
        x2 = x.unsqueeze(2).expand(b, n, n, d)
        stack = torch.stack([x1, x2], dim=1)
        pairwise_d = torch.norm(stack[:, 0] - stack[:, 1], dim=-1)
        mpd = torch.sum(torch.sum(pairwise_d, dim=1), dim=1) / (n * (n - 1))
        return mpd  # (b)
