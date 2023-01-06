import os
from typing import List

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.optim as optim
from dataloader import SingleImgDataset
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


class PlusTrainer:
    def __init__(self, config, is_train=True):
        dataset = SingleImgDataset(config['train_data_path'])
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
                data = data.to(DEVICE)
                z, mu, logvar = self.model.batch_encode_to_z(data, is_VAE=self.isVAE)
                recon = self.model.batch_decode_from_z(z)
                vae_loss = self.vae_loss(data, recon, mu, logvar)
                operations_loss = self.operation_loss(z)
                loss = self.loss_func(vae_loss, operations_loss, train_loss_counter)
                loss.backward()
                optimizer.step()
                if self.is_save_img and batch_ndx == 0 and is_log:
                    save_image(recon[0], os.path.join(self.train_result_path, f'{epoch_num}_recon_{labels[0]}'))
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

    def operation_loss(self, z):
        z_a = z[0:self.sub_batch, 0:self.latent_code_1]
        z_b = z[self.sub_batch:self.sub_batch * 2, 0:self.latent_code_1]
        z_c = z[self.sub_batch * 2:self.sub_batch * 3, 0:self.latent_code_1]
        z_ab, z_ba, loss_ab = self.binary_operation_loss(z_a, z_b)
        z_ac, z_ca, loss_ac = self.binary_operation_loss(z_a, z_c)
        z_bc, z_cb, loss_bc = self.binary_operation_loss(z_b, z_c)

        z_abc = self.model.plus(z_ab, z_c)
        z_bac = self.model.plus(z_ba, z_c)
        z_acb = self.model.plus(z_ac, z_b)
        z_cab = self.model.plus(z_ca, z_b)
        z_bca = self.model.plus(z_bc, z_a)
        z_cba = self.model.plus(z_cb, z_a)
        std_loss = self.std_loss([z_abc, z_bac, z_acb, z_cab, z_bca, z_cba])

        loss = loss_ab + loss_bc + loss_bc + std_loss
        return loss

    def binary_operation_loss(self, z_a, z_b):
        z_ab = self.model.plus(z_a, z_b)
        z_ba = self.model.plus(z_b, z_a)
        plus_loss = self.sum_mse(z_ba, z_ab) * self.z_plus_loss_scalar
        minus_loss = self.minus_loss(z_ab, z_a, z_b) + self.minus_loss(z_ba, z_a, z_b)
        loss = plus_loss + minus_loss
        return z_ab, z_ba, loss

    def minus_loss(self, z_ab, z_a, z_b):
        z_abSa = self.model.minus(z_ab, z_a)
        z_abSb = self.model.minus(z_ab, z_b)
        loss = (self.sum_mse(z_abSb, z_a) + self.sum_mse(z_abSa, z_b)) * self.z_minus_loss_scalar
        return loss

    def std_loss(self, tensors: List[torch.Tensor]):
        tensor = torch.stack(tensors, dim=0)
        stds = torch.std(tensor)
        std_sum = torch.sum(stds) * self.z_std_loss_scalar
        return std_sum

    def vae_loss(self, data, recon, mu, logvar):
        recon_loss = nn.BCELoss(reduction='sum')(recon, data)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar)) * self.kld_loss_scalar
        return recon_loss, KLD

    def loss_func(self, vae_loss, operations_loss, loss_counter):
        xloss_ED, KLD = vae_loss
        loss = torch.zeros(1)[0].to(DEVICE)
        loss += xloss_ED + KLD
        loss += operations_loss
        loss_counter.add_values([xloss_ED.item(),
                                 KLD.item(),
                                 operations_loss.item()
                                 ])
        return loss


if __name__ == "__main__":
    trainer = PlusTrainer(CONFIG)
    trainer.train()
