import random
import matplotlib.pyplot as plt
import matplotlib
import torch
from shared import *
import os
import random
import sys

from torch import optim
from simple_FC import SimpleFC
from dataloader_plus import Dataset
from torch.utils.data import DataLoader
from loss_counter import LossCounter, RECORD_PATH_DEFAULT
from train import split_into_three


def decimal_to_base(n, base):
    if not 2 <= base <= 36:
        raise ValueError("Base must be between 2 and 36")
    if n == 0:
        return "0"

    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    result = ""
    while n > 0:
        remainder = n % base
        result = digits[remainder] + result
        n //= base
    return result


def add_len(n: str, len_to_be: int):
    s = n
    while len(s) < len_to_be:
        s = f'0{s}'
    return s


class DiyCodebook:
    def __init__(self, n_dim, dim_size, v_range=(-1., 1.)):
        self.n_dim = n_dim
        self.dim_size = dim_size
        self.v_range = v_range
        self.dim_points = self.init_dim_points()
        self.linear_book = self.init_linear_book()
        self.random_book = self.init_random_book()
        self.choose_book = {
            'linear': self.linear_book,
            'random': self.random_book
        }

    def init_dim_points(self):
        interval = (self.v_range[1] - self.v_range[0]) / (self.dim_size - 1)
        points = [self.v_range[0]]
        base = self.v_range[0]
        for i in range(0, self.dim_size-1):
            base += interval
            points.append(base)
        return points

    def init_linear_book(self):
        book = []
        for i in range(0, pow(self.dim_size, self.n_dim)):
            trans_num = decimal_to_base(i, self.dim_size)
            num_str = add_len(trans_num, self.n_dim)
            num_quan = [self.dim_points[int(s)] for s in num_str]
            num_tensor = torch.Tensor(num_quan)
            book.append(num_tensor)
        book_tensor = torch.stack(book)
        return book_tensor.to(DEVICE)

    def init_random_book(self):
        rand_idx = torch.randperm(self.linear_book.size(0))
        rand_book = self.linear_book[rand_idx, ...]
        return rand_book

    def plot_book(self, book, n_row):
        n_col = self.dim_size
        fig, axs = plt.subplots(n_row, n_col, sharey="all")
        x = [n + 1 for n in range(0, book[0].size(0))]
        for i in range(0, n_row):
            for j in range(0, n_col):
                book_idx = i * n_col + j
                y = book[book_idx].detach().tolist()
                axs[i, j].scatter(x, y)
                axs[i, j].set_title(f'Num {book_idx}')
                axs[i, j].set(ylabel="value", yticks=self.dim_points)
                axs[i, j].set(xlabel="dimension")
        for ax in axs.flat:
            ax.label_outer()
            ax.grid()
        plt.show()

def name_appd(name: str, path:str):
    return f'{name}_{path}'

def is_need_train(train_config):
    loss_counter = LossCounter([])
    train_record_path = train_config['train_record_path']
    task_name = task_name = train_config['task_name']
    iter_num = loss_counter.load_iter_num(name_appd(task_name, train_record_path))
    if train_config['max_iter_num'] > iter_num:
        print("Continue training")
        return True
    else:
        print("No more training is needed")
        return False

class OtherTask:
    def __init__(self, other_task_config):
        self.other_task_config = other_task_config
        self.num_dim = other_task_config['num_dim']
        self.dim_size = other_task_config['dim_size']
        self.diy_book = DiyCodebook(self.num_dim, self.dim_size)
        self.num_class = other_task_config['num_class']
        self.simple_fc = SimpleFC(other_task_config['fc_network_config'], self.num_dim*2, self.num_class).to(DEVICE)
        task_name = other_task_config['task_name']
        self.fc_model_path = name_appd(task_name, other_task_config['fc_model_path'])
        train_set = Dataset(other_task_config['train_data_path'])
        eval_set = Dataset(other_task_config['eval_data_path'])
        self.batch_size = other_task_config['batch_size']
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size)
        self.eval_loader = DataLoader(eval_set, batch_size=self.batch_size)
        self.train_result_path = name_appd(task_name, other_task_config['train_record_path'])
        self.eval_result_path = name_appd(task_name, other_task_config['eval_record_path'])
        self.learning_rate = other_task_config['learning_rate']
        self.max_iter_num = other_task_config['max_iter_num']
        self.log_interval = other_task_config['log_interval']
        self.CE_loss = torch.nn.CrossEntropyLoss(reduction='mean')

    def resume(self):
        if os.path.exists(self.fc_model_path):
            self.simple_fc.load_state_dict(torch.load(self.fc_model_path))
            print(f"FC Model is loaded")
        else:
            print("New FC model is initialized")

    def load_pretrained(self):
        pretrained = VQVAE(self.train_config).to(DEVICE)
        pretrained_path = self.other_task_config['pretrained_path']
        if os.path.exists(pretrained_path):
            pretrained.load_state_dict(pretrained.load_tensor(pretrained_path))
            print(f"Pretrained Model is loaded")
            return pretrained
        else:
            print("No pretrained parameters")
            exit()

    def train(self):
        self.load_pretrained()
        self.simple_fc.train()
        self.resume()
        train_loss_counter = LossCounter(['loss_z',
                                          'accu'], self.train_result_path)
        eval_loss_counter = LossCounter(['loss_z',
                                         'accu'], self.eval_result_path)
        start_epoch = train_loss_counter.load_iter_num(self.train_result_path)
        optimizer = optim.Adam(self.simple_fc.parameters(), lr=self.learning_rate)
        for epoch_num in range(start_epoch, self.max_iter_num):
            is_log = (epoch_num % self.log_interval == 0 and epoch_num != 0)
            self.one_epoch(epoch_num, train_loss_counter, self.train_loader, is_log, optimizer)
            if is_log:
                torch.save(self.simple_fc.state_dict(), self.fc_model_path)
                self.simple_fc.eval()
                self.one_epoch(epoch_num, eval_loss_counter, self.eval_loader, True, None)
                self.simple_fc.train()

    def fc_comp(self, sample):
        data, labels = sample
        tensor_y = data_name_2_labels(labels[2]).to(DEVICE)
        sizes = data[0].size()
        data_all = torch.stack(data, dim=0).reshape(3 * sizes[0], sizes[1], sizes[2], sizes[3])
        e_all, e_q_loss, z_all = self.pretrained.batch_encode_to_z(data_all)
        e_content = e_all[..., 0:self.latent_code_1]
        ec_a, ec_b, ec_c = split_into_three(e_content)
        ec_ab = self.simple_fc.composition(ec_a, ec_b)
        loss = self.CE_loss(ec_ab, tensor_y)
        accu = (ec_ab.argmax(1) == tensor_y).float().mean().item()
        return ec_ab, tensor_y, loss, accu

    def one_epoch(self, epoch_num, loss_counter: LossCounter, data_loader,
                  is_log, optimizer: torch.optim.Optimizer = None):
        for batch_ndx, sample in enumerate(data_loader):
            print(f'Epoch: {epoch_num}')
            if optimizer is not None:
                optimizer.zero_grad()
            ec_ab, tensor_y, loss, accu = self.fc_comp(sample)
            loss_counter.add_values([
                loss.item(),
                accu
            ])
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        if is_log:
            print(loss_counter.make_record(epoch_num))
            loss_counter.record_and_clear(RECORD_PATH_DEFAULT, epoch_num)


if __name__ == "__main__":
    diy_codebook = DiyCodebook(3, 4)
    print(diy_codebook.dim_points)
    print(diy_codebook.linear_book)
    diy_codebook.plot_book(diy_codebook.random_book, 4)
