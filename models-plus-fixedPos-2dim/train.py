import os
import random
from typing import List

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.optim as optim
from dataloader_plus import Dataset
from loss_counter import LossCounter
from model import S3Plus
from shared import *
from train_config import CONFIG


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
    sizes = [3, int(tensor.size(0)/3), *tensor.size()[1:]]
    new_tensor = tensor.reshape(*sizes)
    return new_tensor[0], new_tensor[1], new_tensor[2]


class PlusTrainer:
    def __init__(self, config, is_train=True):
        dataset = Dataset(config['train_data_path'])
        self.batch_size = config['batch_size']
        self.loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model = S3Plus(config).to(DEVICE)
        self.train_result_path = config['train_result_path']
        self.train_record_path = config['train_record_path']
        self.model_path = config['model_path']
        self.learning_rate = config['learning_rate']
        self.scheduler_base_num = config['scheduler_base_num']
        self.is_save_img = config['is_save_img']
        self.isVAE = config['kld_loss_scalar'] > 0.00001
        self.kld_loss_scalar = config['kld_loss_scalar']
        self.log_interval = config['log_interval']
        self.checkpoint_interval = config['checkpoint_interval']
        self.max_iter_num = config['max_iter_num']
        self.latent_code_1 = config['latent_code_1']
        self.z_std_loss_scalar = config['z_std_loss_scalar']
        self.sub_batch = int(self.batch_size / 3)
        self.sum_mse = nn.MSELoss(reduction='sum')
        self.z_minus_loss_scalar = config['z_minus_loss_scalar']
        self.z_plus_loss_scalar = config['z_plus_loss_scalar']
        self.commutative_z_loss_scalar = config['commutative_z_loss_scalar']
        self.associative_z_loss_scalar = config['associative_z_loss_scalar']

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
                                          'KLD',
                                          'plus',
                                          'loss_oper'])
        start_epoch = train_loss_counter.load_iter_num(self.train_record_path)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: self.scheduler_func(start_epoch))
        for epoch_num in range(start_epoch, self.max_iter_num):
            print(f'Epoch: {epoch_num}')
            is_log = (epoch_num % self.log_interval == 0 and epoch_num != 0)
            for batch_ndx, sample in enumerate(self.loader):
                optimizer.zero_grad()
                data, labels = sample
                sizes = data[0].size()
                data_all = torch.stack(data, dim=0).reshape(3*sizes[0], sizes[1], sizes[2], sizes[3])
                z_all, mu, logvar = self.model.batch_encode_to_z(data_all, is_VAE=self.isVAE)
                z_content = z_all[..., 0:self.latent_code_1]
                za, zb, zc = split_into_three(z_all)
                recon = self.model.batch_decode_from_z(z_all)
                vae_loss = self.vae_loss(data_all, recon, mu, logvar)
                plus_loss = self.plus_loss(za, zb, data[2]) + self.plus_loss(zb, za, data[2])
                operations_loss = self.operation_loss_z(z_content)
                loss = self.loss_func(vae_loss, plus_loss, operations_loss, train_loss_counter)
                loss.backward()
                optimizer.step()
                if self.is_save_img and batch_ndx == 0 and is_log:
                    save_image(recon[0], os.path.join(self.train_result_path, f'{epoch_num}.png'))
                    # save_image(data[0], os.path.join(self.train_result_path, f'{epoch_num}_data_{labels[0]}'))

            # scheduler.step()

            if is_log:
                self.model.save_tensor(self.model.state_dict(), self.model_path)
                print(train_loss_counter.make_record(epoch_num))
                train_loss_counter.record_and_clear(self.train_record_path, epoch_num)
                # self.save_result_imgs(recon_list, f'{i}_{str(I_sample_points)}', z_rpm.size(1) - 1)

            if epoch_num % self.checkpoint_interval == 0 and epoch_num != 0:
                self.model.save_tensor(self.model.state_dict(), f'checkpoint_{epoch_num}.pt')

    print("train ends")

    def plus_loss(self, za, zb, imgs_c):
        z_s = za[..., self.latent_code_1:] if random.random() > 0.5 else zb[..., self.latent_code_1:]
        z_ab_content = self.model.plus(za[..., 0:self.latent_code_1], zb[..., 0:self.latent_code_1])
        z_ab = torch.cat((z_ab_content, z_s), -1)
        recon_c = self.model.batch_decode_from_z(z_ab)
        recon_loss = nn.BCELoss(reduction='sum')(recon_c, imgs_c)
        return recon_loss

    def commutative_z_loss(self, z_a, z_b):
        z_ab = self.model.plus(z_a, z_b)
        z_ba = self.model.plus(z_b, z_a)
        loss = self.sum_mse(z_ab, z_ba)
        return loss * self.commutative_z_loss_scalar

    def associative_z_loss(self, z_a, z_b, z_c):
        z_ab = self.model.plus(z_a, z_b)
        z_abc1 = self.model.plus(z_ab, z_c)
        z_bc = self.model.plus(z_b, z_c)
        z_abc2 = self.model.plus(z_a, z_bc)
        loss = self.sum_mse(z_abc1, z_abc2)
        return loss * self.associative_z_loss_scalar

    def operation_loss_z(self, z):
        idx_1 = torch.randperm(z.size(0))
        idx_2 = torch.randperm(z.size(0))
        idx_3 = torch.randperm(z.size(0))
        z1 = z[idx_1, ...] + torch.rand_like(z)
        z2 = z[idx_2, ...] + torch.rand_like(z)
        z3 = z[idx_3, ...] + torch.rand_like(z)
        loss = torch.zeros(1)[0].to(DEVICE)
        if self.commutative_z_loss_scalar > 0.00001:
            loss += self.commutative_z_loss(z1, z2)
        if self.associative_z_loss_scalar > 0.00001:
            loss += self.associative_z_loss(z1, z2, z3)
        return loss

    def vae_loss(self, data, recon, mu, logvar):
        recon_loss = nn.BCELoss(reduction='sum')(recon, data)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar)) * self.kld_loss_scalar
        return recon_loss, KLD

    def loss_func(self, vae_loss, plus_loss, operations_loss, loss_counter):
        xloss_ED, KLD = vae_loss
        loss = torch.zeros(1)[0].to(DEVICE)
        loss += xloss_ED + KLD
        loss += plus_loss
        loss += operations_loss
        loss_counter.add_values([xloss_ED.item(),
                                 KLD.item(),
                                 plus_loss.item(),
                                 operations_loss.item()
                                 ])
        return loss


if __name__ == "__main__":
    trainer = PlusTrainer(CONFIG)
    trainer.train()
