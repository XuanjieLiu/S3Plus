import os
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.optim as optim
from dataloader import Dataset
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
        dataset = Dataset(config['train_data_path'])
        self.loader = DataLoader(dataset, batch_size=32, shuffle=True)
        self.model = S3Plus(config).to(DEVICE)
        self.train_result_path = config['train_result_path']
        self.train_record_path = config['train_record_path']
        self.model_path = config['model_path']
        self.learning_rate = config['learning_rate']
        self.scheduler_base_num = config['scheduler_base_num']
        self.is_save_img = config['is_save_img']
        self.isVAE = config['is_save_img'] > 0.00001
        self.kld_loss_scalar = config['kld_loss_scalar']
        self.log_interval = config['log_interval']
        self.checkpoint_interval = config['checkpoint_interval']
        self.max_iter_num = config['max_iter_num']


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
                                          'KLD'])
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
                loss = self.loss_func(vae_loss,
                                      train_loss_counter)
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

    def vae_loss(self, data, recon, mu, logvar):
        recon_loss = nn.BCELoss(reduction='sum')(recon, data)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar)) * self.kld_loss_scalar
        return recon_loss, KLD

    def loss_func(self, vae_loss, loss_counter):
        xloss_ED, KLD = vae_loss
        loss = torch.zeros(1)[0].to(DEVICE)
        loss += xloss_ED + KLD
        loss_counter.add_values([xloss_ED.item(),
                                 KLD.item()
                                 ])
        return loss


if __name__ == "__main__":
    trainer = PlusTrainer(CONFIG)
    trainer.train()
