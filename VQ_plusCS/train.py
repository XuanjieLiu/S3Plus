import os
import random
import sys

from VQ_plusCS.eval_plus_nd import load_plusZ_eval_data, plot_plusZ_against_label

sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.optim as optim
from dataloader_plus import MultiImgDataset
from dataloader import SingleImgDataset
from loss_counter import LossCounter
from VQVAE import VQVAE
from shared import *
from train_config import CONFIG
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
        dataset = MultiImgDataset(config['train_data_path'])
        eval_set_1 = SingleImgDataset(config['single_img_eval_set_path'])
        self.batch_size = config['batch_size']
        self.min_loss_scalar = config['min_loss_scalar']
        self.loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.eval_loader_1 = DataLoader(eval_set_1, batch_size=self.batch_size)
        self.model = VQVAE(config).to(DEVICE)
        self.train_result_path = config['train_result_path']
        self.train_record_path = config['train_record_path']
        self.model_path = config['model_path']
        self.learning_rate = config['learning_rate']
        self.scheduler_base_num = config['scheduler_base_num']
        self.is_save_img = config['is_save_img']
        self.log_interval = config['log_interval']
        self.checkpoint_interval = config['checkpoint_interval']
        self.max_iter_num = config['max_iter_num']
        self.latent_code_1 = config['latent_code_1']
        self.z_std_loss_scalar = config['z_std_loss_scalar']
        self.sub_batch = int(self.batch_size / 3)
        self.sum_mse = nn.MSELoss(reduction='sum')
        self.mean_mse = nn.MSELoss(reduction='mean')
        self.z_minus_loss_scalar = config['z_minus_loss_scalar']
        self.z_plus_loss_scalar = config['z_plus_loss_scalar']
        self.commutative_z_loss_scalar = config['commutative_z_loss_scalar']
        self.associative_z_loss_scalar = config['associative_z_loss_scalar']
        self.K = config['K']
        self.assoc_aug_range = config['assoc_aug_range']
        self.eval_help = EvalHelper(config)
        self.VQPlus_eqLoss_scalar = config['VQPlus_eqLoss_scalar']
        self.plus_recon_loss_scalar = config['plus_recon_loss_scalar']

    def resume(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(self.model.load_tensor(self.model_path))
            print(f"Model is loaded")
        else:
            print("New model is initialized")

    def scheduler_func(self, curr_iter):
        return self.scheduler_base_num ** curr_iter

    def train(self):
        os.makedirs(self.train_result_path, exist_ok=True)
        self.model.train()
        self.resume()
        train_loss_counter = LossCounter(['loss_ED',
                                          'VQ_C',
                                          'plus_recon',
                                          'plus_z',
                                          'loss_oper'])
        start_epoch = train_loss_counter.load_iter_num(self.train_record_path)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: self.scheduler_func(start_epoch))
        for epoch_num in range(start_epoch, self.max_iter_num):
            print(f'Epoch: {epoch_num}')
            is_log = (epoch_num % self.log_interval == 0 and epoch_num != 0)
            for batch_ndx, sample in enumerate(self.loader):
                vis_imgs = VisImgs()
                optimizer.zero_grad()
                data, labels = sample
                sizes = data[0].size()
                data_all = torch.stack(data, dim=0).reshape(3 * sizes[0], sizes[1], sizes[2], sizes[3])
                z_all, e_q_loss = self.model.batch_encode_to_z(data_all)
                z_content = z_all[..., 0:self.latent_code_1]
                za, zb, zc = split_into_three(z_all)
                recon = self.model.batch_decode_from_z(z_all)
                recon_a, recon_b, recon_c = split_into_three(recon)
                vis_imgs.gt_a, vis_imgs.gt_b, vis_imgs.gt_c = data[0][0], data[1][0], data[2][0]
                vis_imgs.recon_a, vis_imgs.recon_b, vis_imgs.recon_c = recon_a[0], recon_b[0], recon_c[0]
                vae_loss = self.vae_loss(data_all, recon)

                plus_loss = self.bi_plus_loss(za, zb, zc, data[2], vis_imgs)

                operations_loss = self.operation_loss_z(z_content)

                loss = self.loss_func(vae_loss, e_q_loss, plus_loss, operations_loss, train_loss_counter)
                loss.backward()
                optimizer.step()
                if self.is_save_img and batch_ndx == 0 and is_log:
                    # save_image(recon[0], os.path.join(self.train_result_path, f'{epoch_num}.png'))
                    vis_imgs.save_img(os.path.join(self.train_result_path, f'{epoch_num}.png'))

            # scheduler.step()

            if is_log:
                self.model.save_tensor(self.model.state_dict(), self.model_path)
                print(train_loss_counter.make_record(epoch_num))
                train_loss_counter.record_and_clear(self.train_record_path, epoch_num)
                if self.is_save_img:
                    self.plot_enc_z(epoch_num)
                    self.plot_plus_z(epoch_num)

            if epoch_num % self.checkpoint_interval == 0 and epoch_num != 0:
                self.model.save_tensor(self.model.state_dict(), f'checkpoint_{epoch_num}.pt')

    print("train ends")

    def plot_enc_z(self, epoch_num):
        num_z, num_labels = load_enc_eval_data(self.eval_loader_1,
                                               lambda x: self.model.batch_encode_to_z(x))
        eval_path = os.path.join(self.train_result_path, f'{epoch_num}_z.png')
        plot_z_against_label(num_z, num_labels, eval_path, self.eval_help)

    def plot_plus_z(self, epoch_num):
        all_enc_z, all_plus_z = load_plusZ_eval_data(
            self.loader,
            lambda x: self.model.batch_encode_to_z(x),
            self.model.plus,
            self.latent_code_1
        )
        eval_path = os.path.join(self.train_result_path, f'{epoch_num}_plus_z.png')
        plot_plusZ_against_label(all_enc_z, all_plus_z, eval_path, self.eval_help)

    def bi_plus_loss(self, za, zb, zc, imgs_c, vis_imgs: VisImgs = None):
        plus_loss1 = self.plus_loss(za, zb, zc, imgs_c, vis_imgs)
        plus_loss2 = self.plus_loss(zb, za, zc, imgs_c)
        plus_loss = (
            plus_loss1[0] + plus_loss2[0],
            plus_loss1[1] + plus_loss2[1]
        )
        return plus_loss

    def plus_loss(self, za, zb, zc, imgs_c, vis_imgs: VisImgs = None):
        z_s = za[..., self.latent_code_1:] if random.random() > 0.5 else zb[..., self.latent_code_1:]
        z_ab_content, e_q_loss = self.model.plus(za[..., 0:self.latent_code_1], zb[..., 0:self.latent_code_1])
        z_ab = torch.cat((z_ab_content, z_s), -1)
        recon_c = self.model.batch_decode_from_z(z_ab)
        recon_loss = nn.MSELoss()(recon_c, imgs_c) * self.plus_recon_loss_scalar
        if vis_imgs is not None:
            vis_imgs.plus_c = recon_c[0]
        if self.z_plus_loss_scalar > self.min_loss_scalar:
            z_loss = self.mean_mse(z_ab, zc) * self.z_plus_loss_scalar
        else:
            z_loss = torch.zeros(1)[0]
        return recon_loss, z_loss + e_q_loss * self.VQPlus_eqLoss_scalar

    def commutative_z_loss(self, z_a, z_b):
        z_ab, e_q_loss_ab = self.model.plus(z_a, z_b)
        z_ba, e_q_loss_ba = self.model.plus(z_b, z_a)
        loss = self.mean_mse(z_ab, z_ba) * self.commutative_z_loss_scalar + self.VQPlus_eqLoss_scalar * (
                    e_q_loss_ab + e_q_loss_ba)
        return loss

    def associative_z_loss(self):
        trans_dim = np.ones(self.latent_code_1, dtype=int)
        z_a, _ = make_translation_batch(self.K, dim=trans_dim, t_range=self.assoc_aug_range)
        z_b, _ = make_translation_batch(self.K, dim=trans_dim, t_range=self.assoc_aug_range)
        z_c, _ = make_translation_batch(self.K, dim=trans_dim, t_range=self.assoc_aug_range)
        z_ab, e_q_loss_ab = self.model.plus(z_a, z_b)
        z_abc1, e_q_loss_abc1 = self.model.plus(z_ab, z_c)
        z_bc, e_q_loss_bc = self.model.plus(z_b, z_c)
        z_abc2, e_q_loss_abc2 = self.model.plus(z_a, z_bc)
        accoc_plus_loss = self.mean_mse(z_abc1, z_abc2) * self.associative_z_loss_scalar
        e_q_loss = self.VQPlus_eqLoss_scalar * (e_q_loss_ab + e_q_loss_abc1 + e_q_loss_bc + e_q_loss_abc2)
        return accoc_plus_loss + e_q_loss

    def operation_loss_z(self, z):
        # idx_1 = torch.randperm(z.size(0))
        # z1 = z[idx_1, ...] + torch.randn_like(z)

        za, zb, zc = split_into_three(z)
        # zd = z1[0:za.size(0)].detach()

        loss = torch.zeros(1)[0].to(DEVICE)
        if self.commutative_z_loss_scalar > self.min_loss_scalar:
            loss += self.commutative_z_loss(za, zb)
        if self.associative_z_loss_scalar > self.min_loss_scalar:
            loss += self.associative_z_loss()
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


if __name__ == "__main__":
    trainer = PlusTrainer(CONFIG)
    trainer.train()
