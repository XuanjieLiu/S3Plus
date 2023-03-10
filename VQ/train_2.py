import os
import random
import sys
from eval_plus_nd import VQvaePlusEval, plot_plusZ_against_label
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.optim as optim
from dataloader_plus import Dataset
from dataloader import SingleImgDataset
from loss_counter import LossCounter, RECORD_PATH_DEFAULT
from VQVAE import VQVAE
from shared import *
from num_eval import plot_z_against_label, load_enc_eval_data
from visual_imgs import VisImgs
from eval_common import EvalHelper


def make_translation_batch(batch_size, dim=np.array([1, 0, 1]), is_std_normal=False, t_range=(-3, 3)):
    scale = t_range[1] - t_range[0]
    if is_std_normal:
        T_mat = torch.randn(batch_size, len(dim))
    else:
        T_mat = torch.rand(batch_size, len(dim)) * scale + t_range[0]
    T = T_mat.mul(torch.from_numpy(dim)).cuda()
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


def split_into_three(tensor):
    sizes = [3, int(tensor.size(0) / 3), *tensor.size()[1:]]
    new_tensor = tensor.reshape(*sizes)
    return new_tensor[0], new_tensor[1], new_tensor[2]


class PlusTrainer:
    def __init__(self, config, is_train=True):
        self.config = config
        dataset = Dataset(config['train_data_path'])
        eval_set_1 = SingleImgDataset(config['eval_path_1'])
        eval_set_2 = Dataset(config['eval_path_2'])
        self.batch_size = config['batch_size']
        self.min_loss_scalar = config['min_loss_scalar']
        self.loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.eval_loader_1 = DataLoader(eval_set_1, batch_size=self.batch_size)
        self.eval_loader_2 = DataLoader(eval_set_2, batch_size=self.batch_size)
        self.model = VQVAE(config).to(DEVICE)
        self.train_result_path = config['train_result_path']
        self.eval_result_path = config['eval_result_path']
        self.train_record_path = config['train_record_path']
        self.eval_record_path = config['eval_record_path']
        self.model_path = config['model_path']
        self.learning_rate = config['learning_rate']
        self.scheduler_base_num = config['scheduler_base_num']
        self.is_save_img = config['is_save_img']
        self.log_interval = config['log_interval']
        self.checkpoint_interval = config['checkpoint_interval']
        self.max_iter_num = config['max_iter_num']
        self.embedding_dim = config['embedding_dim']
        self.latent_code_1 = config['latent_embedding_1'] * self.embedding_dim
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
        self.isVQStyle = config['isVQStyle']
        self.is_commutative_train = config['is_commutative_train']

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

            if self.plus_recon_loss_scalar < self.min_loss_scalar:
                plus_loss = torch.zeros(2)
            else:
                if self.plus_by_embedding:
                    plus_loss = self.bi_plus_loss(e_all, data[2], vis_imgs)
                else:
                    plus_loss = self.bi_plus_loss(z_all, data[2], vis_imgs)

            if self.plus_by_embedding:
                operations_loss = self.operation_loss_z(e_content)
            else:
                z_content = z_all[..., 0:self.latent_code_1]
                operations_loss = self.operation_loss_z(z_content)

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
        train_loss_counter = LossCounter(['loss_ED',
                                          'VQ_C',
                                          'plus_recon',
                                          'plus_z',
                                          'loss_oper'], self.train_record_path)
        eval_loss_counter = LossCounter(['loss_ED',
                                          'VQ_C',
                                          'plus_recon',
                                          'plus_z',
                                          'loss_oper'], self.eval_record_path)
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
                self.one_epoch(epoch_num, eval_loss_counter, self.eval_loader_2, True, eval_vis_img, None)
                self.model.train()
                if self.is_save_img:
                    self.plot_enc_z(epoch_num, self.eval_loader_1)
                    self.plot_plus_z(epoch_num, self.loader, self.train_result_path)
                    self.plot_plus_z(epoch_num, self.eval_loader_2, self.eval_result_path)
            if epoch_num % self.checkpoint_interval == 0 and epoch_num != 0:
                self.model.save_tensor(self.model.state_dict(), f'checkpoint_{epoch_num}.pt')

    print("train ends")

    def plot_enc_z(self, epoch_num, data_loader):
        num_z, num_labels = load_enc_eval_data(
                                data_loader,
                                lambda x: self.model.find_indices(
                                    self.model.batch_encode_to_z(x)[0],
                                    True
                                )
        )
        eval_path = os.path.join(self.train_result_path, f'{epoch_num}_z.png')
        plot_z_against_label(num_z, num_labels, eval_path, self.eval_help)

    def plot_plus_z(self, epoch_num, data_loader, result_path):
        plus_eval = VQvaePlusEval(self.config, data_loader, loaded_model=self.model)
        all_enc_z, all_plus_z = plus_eval.load_plusZ_eval_data()
        eval_path = os.path.join(result_path, f'{epoch_num}_plus_z')
        plot_plusZ_against_label(all_enc_z, all_plus_z, eval_path, self.eval_help)

    def bi_plus_loss(self, z_all, imgs_c, vis_imgs: VisImgs = None):
        za, zb, zc = split_into_three(z_all)
        plus_loss1 = self.plus_loss(za, zb, zc, imgs_c, vis_imgs)
        plus_loss2 = self.plus_loss(zb, za, zc, imgs_c) if self.is_commutative_train else torch.zeros(2)
        plus_loss = (
            plus_loss1[0] + plus_loss2[0],
            plus_loss1[1] + plus_loss2[1]
        )
        return plus_loss

    def plus_loss(self, za, zb, zc, imgs_c, vis_imgs: VisImgs = None):
        z_s = za[..., self.latent_code_1:] if random.random() > 0.5 else zb[..., self.latent_code_1:]
        if self.isVQStyle:
            z_s = self.model.vq_layer(z_s)[0]
        e_ab_content, e_q_loss, z_ab_content = self.model.plus(za[..., 0:self.latent_code_1], zb[..., 0:self.latent_code_1])
        e_ab = torch.cat((e_ab_content, z_s), -1)
        z_ab = torch.cat((z_ab_content, z_s), -1)
        recon_c = self.model.batch_decode_from_z(e_ab)
        recon_loss = nn.MSELoss()(recon_c, imgs_c) * self.plus_recon_loss_scalar
        if vis_imgs is not None:
            vis_imgs.plus_c = recon_c[0]
        if self.z_plus_loss_scalar > self.min_loss_scalar:
            if self.plus_by_embedding:
                z_loss = self.mean_mse(e_ab, zc) * self.z_plus_loss_scalar
            else:
                z_loss = self.mean_mse(z_ab, zc) * self.z_plus_loss_scalar
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

    # def rand_associative_z_loss(self):
    #     trans_dim = np.ones(self.latent_code_1, dtype=int)
    #     z_a, _ = make_translation_batch(self.K, dim=trans_dim, t_range=self.assoc_aug_range)
    #     z_b, _ = make_translation_batch(self.K, dim=trans_dim, t_range=self.assoc_aug_range)
    #     z_c, _ = make_translation_batch(self.K, dim=trans_dim, t_range=self.assoc_aug_range)
    #     e_ab, e_q_loss_ab, z_ab = self.model.plus(z_a, z_b)
    #     e_abc1, e_q_loss_abc1, z_abc1 = self.model.plus(e_ab, z_c)
    #     e_bc, e_q_loss_bc, z_bc = self.model.plus(z_b, z_c)
    #     e_abc2, e_q_loss_abc2, z_abc2 = self.model.plus(z_a, e_bc)
    #     accoc_plus_loss = self.mean_mse(e_abc1, e_abc2) * self.associative_z_loss_scalar
    #     e_q_loss = self.VQPlus_eqLoss_scalar * (e_q_loss_ab + e_q_loss_abc1 + e_q_loss_bc + e_q_loss_abc2)
    #     return accoc_plus_loss + e_q_loss

    def associative_z_loss(self, z_all_content):
        idx_1 = torch.randperm(z_all_content.size(0))
        z_perm = z_all_content[idx_1, ...]
        z_a, z_b, z_c = split_into_three(z_perm)
        e_ab, e_q_loss_ab, z_ab = self.model.plus(z_a, z_b)
        e_abc1, e_q_loss_abc1, z_abc1 = self.model.plus(e_ab, z_c)
        e_bc, e_q_loss_bc, z_bc = self.model.plus(z_b, z_c)
        e_abc2, e_q_loss_abc2, z_abc2 = self.model.plus(z_a, e_bc)
        accoc_plus_loss = self.mean_mse(e_abc1, e_abc2) * self.associative_z_loss_scalar
        e_q_loss = self.VQPlus_eqLoss_scalar * (e_q_loss_ab + e_q_loss_abc1 + e_q_loss_bc + e_q_loss_abc2)
        return accoc_plus_loss + e_q_loss

    def operation_loss_z(self, z_all_content):
        za, zb, zc = split_into_three(z_all_content)
        loss = torch.zeros(1)[0].to(DEVICE)
        if self.commutative_z_loss_scalar > self.min_loss_scalar and self.is_commutative_train:
            loss += self.commutative_z_loss(za, zb)
        if self.associative_z_loss_scalar > self.min_loss_scalar:
            loss += self.associative_z_loss(z_all_content)
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


