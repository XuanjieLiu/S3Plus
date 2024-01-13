import os
import random
import sys
from eval_plus_nd import VQvaePlusEval, plot_plusZ_against_label, calc_ks_enc_plus_z

sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.optim as optim
from dataloader_plus import Dataset
from dataloader import SingleImgDataset, load_enc_eval_data
from loss_counter import LossCounter, RECORD_PATH_DEFAULT
from VQVAE import VQVAE
from shared import *
from num_eval import plot_z_against_label
from visual_imgs import VisImgs
from eval_common import EvalHelper
from eval.dec_vis_eval_2digit import plot_dec_img


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

def split_into_three(tensor):
    sizes = [3, int(tensor.size(0) / 3), *tensor.size()[1:]]
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


class PlusTrainer:
    def __init__(self, config, is_train=True):
        self.config = config
        dataset = Dataset(config['train_data_path'])
        single_img_eval_set = SingleImgDataset(config['single_img_eval_set_path'])
        plus_eval_set = Dataset(config['plus_eval_set_path'])
        self.batch_size = config['batch_size']
        self.min_loss_scalar = config['min_loss_scalar']
        self.loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.single_img_eval_loader = DataLoader(single_img_eval_set, batch_size=self.batch_size)
        self.plus_eval_loader = DataLoader(plus_eval_set, batch_size=self.batch_size)
        self.plus_eval_loader_2 = self.init_plus_eval_loader_2(config, 'plus_eval_set_path_2')
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

    def init_plus_eval_loader_2(self, config, key):
        if config[key] is None:
            print(f"Key {key} is None")
            return None
        else:
            return DataLoader(Dataset(config[key]), batch_size=self.batch_size)

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
            e_all, e_q_loss, z_all = self.model.batch_encode_to_z(data_all)
            e_content = e_all[..., 0:self.latent_code_1]

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
                z_content = z_all[..., 0:self.latent_code_1]
                operations_loss += self.operation_loss_z(z_content)

            loss = self.loss_func(vae_loss, e_q_loss, plus_loss, operations_loss, loss_counter)
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

    def train(self):
        os.makedirs(self.train_result_path, exist_ok=True)
        os.makedirs(self.eval_result_path, exist_ok=True)
        self.model.train()
        self.resume()
        loss_counter_keys = ['loss_ED', 'VQ_C', 'plus_recon', 'plus_z', 'loss_oper']
        train_loss_counter = LossCounter(loss_counter_keys, self.train_record_path)
        eval_loss_counter = LossCounter(loss_counter_keys, self.eval_record_path)
        special_loss_counter_keys = ['train_ks', 'train_accu', 'eval_ks', 'eval_accu']
        if self.plus_eval_loader_2 is not None:
            special_loss_counter_keys.extend(['eval_ks_2', 'eval_accu_2'])
        special_loss_counter = LossCounter(special_loss_counter_keys, self.plus_accu_record_path)
        start_epoch = train_loss_counter.load_iter_num(self.train_record_path)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: self.scheduler_func(start_epoch))
        for epoch_num in range(start_epoch, self.max_iter_num):
            is_log = (epoch_num % self.log_interval == 0 and epoch_num != 0)
            train_vis_img = VisImgs(self.train_result_path)
            eval_vis_img = VisImgs(self.eval_result_path)
            self.one_epoch(epoch_num, train_loss_counter, self.loader, is_log, train_vis_img, optimizer)
            # scheduler.step()
            if is_log:
                self.model.eval()
                self.one_epoch(epoch_num, eval_loss_counter, self.plus_eval_loader, True, eval_vis_img, None)
                self.model.train()
                if self.is_save_img:
                    self.plot_enc_z(epoch_num, self.single_img_eval_loader)
                    train_ks, train_accu = self.plot_plus_idx(epoch_num, self.loader, self.train_result_path)
                    eval_ks, eval_accu = self.plot_plus_idx(epoch_num, self.plus_eval_loader, self.eval_result_path, "plus_idx")
                    loss_values = [train_ks, train_accu, eval_ks, eval_accu]
                    if self.plus_eval_loader_2 is not None:
                        eval_ks_2, eval_accu_2 = self.plot_plus_idx(epoch_num, self.plus_eval_loader_2, self.eval_result_path, "plus_idx_2")
                        loss_values.extend([eval_ks_2, eval_accu_2])
                    special_loss_counter.add_values(loss_values)
                    special_loss_counter.record_and_clear(RECORD_PATH_DEFAULT, epoch_num)
                    if self.is_plot_zc_value:
                        self.plot_plus_z(epoch_num, self.plus_eval_loader, self.eval_result_path, 'plus_z')
                        if self.plus_eval_loader_2 is not None:
                            self.plot_plus_z(epoch_num, self.plus_eval_loader_2, self.eval_result_path, 'plus_z_2')

            if epoch_num % self.checkpoint_interval == 0 and epoch_num != 0:
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

    def plot_plus_idx(self, epoch_num, data_loader, result_path, result_name="plus_idx"):
        plus_eval = VQvaePlusEval(self.config, loaded_model=self.model)
        all_enc_z, all_plus_z = plus_eval.load_plusZ_eval_data(data_loader, True)
        ks, accu = calc_ks_enc_plus_z(all_enc_z, all_plus_z)
        eval_path = os.path.join(result_path, f'{epoch_num}_{result_name}_ks_{ks}')
        plot_plusZ_against_label(all_enc_z, all_plus_z, eval_path,
                                 is_scatter_lines=True, y_label="Emb idx")
        return ks, accu

    def plot_plus_z(self, epoch_num, data_loader, result_path, result_name="plus_z"):
        plus_eval = VQvaePlusEval(self.config, loaded_model=self.model)
        all_enc_z, all_plus_z = plus_eval.load_plusZ_eval_data(data_loader, False)
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

    def commutative_z_loss(self, z_a, z_b):
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
        if self.is_assoc_within_batch:
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
        e_abc1, e_q_loss_abc1, z_abc1 = self.model.plus(e_ab, z_c)
        if self.is_symm_assoc:
            e_ac, e_q_loss_ac, z_ac = self.model.plus(z_a, z_c)
            e_abc2, e_q_loss_abc2, z_abc2 = self.model.plus(e_ac, z_b)
            e_q_loss_12 = e_q_loss_ac
        else:
            e_bc, e_q_loss_bc, z_bc = self.model.plus(z_b, z_c)
            e_abc2, e_q_loss_abc2, z_abc2 = self.model.plus(z_a, e_bc)
            e_q_loss_12 = e_q_loss_bc
        assoc_plus_loss = torch.zeros(1)[0].to(DEVICE)
        if self.config['is_assoc_on_e']:
            assoc_plus_loss += self.mean_mse(e_abc1, e_abc2) * self.associative_z_loss_scalar
        if self.config['is_assoc_on_z']:
            assoc_plus_loss += self.mean_mse(z_abc1, z_abc2) * self.associative_z_loss_scalar
        e_q_loss = self.VQPlus_eqLoss_scalar * (e_q_loss_ab + e_q_loss_abc1 + e_q_loss_12 + e_q_loss_abc2)
        return assoc_plus_loss + e_q_loss

    def symm_assoc_loss(self, z_a, z_b, z_c):
        e_ab, e_q_loss_ab, z_ab = self.model.plus(z_a, z_b)
        e_abc1, e_q_loss_abc1, z_abc1 = self.model.plus(e_ab, z_c)
        e_ac, e_q_loss_ac, z_ac = self.model.plus(z_a, z_c)
        e_abc2, e_q_loss_abc2, z_abc2 = self.model.plus(e_ac, z_b)



    def operation_loss_z(self, z_all_content):
        za, zb, zc = split_into_three(z_all_content)
        loss = torch.zeros(1)[0].to(DEVICE)
        if self.commutative_z_loss_scalar > self.min_loss_scalar and self.is_commutative_train:
            loss += self.commutative_z_loss(za, zb)
        if self.associative_z_loss_scalar > self.min_loss_scalar:
            if self.config['is_zc_based_assoc']:
                loss += self.associative_loss(*self.zc_based_associative_z(z_all_content))
            if self.config['is_rand_z_assoc']:
                loss += self.associative_loss(*self.rand_associative_z(z_all_content))
        return loss

    def vae_loss(self, data, recon):
        recon_loss = nn.MSELoss(reduction='mean')(recon, data)
        return recon_loss

    def loss_func(self, vae_loss, e_q_loss, plus_loss, operations_loss, loss_counter):
        xloss_ED = vae_loss
        plus_recon_loss, plus_z_loss = plus_loss
        loss = torch.zeros(1)[0].to(DEVICE)
        loss += xloss_ED + e_q_loss
        loss += plus_recon_loss + plus_z_loss
        loss += operations_loss
        loss_counter.add_values([xloss_ED.item(),
                                 e_q_loss.item(),
                                 plus_recon_loss.item(),
                                 plus_z_loss.item(),
                                 operations_loss.item()
                                 ])
        return loss


