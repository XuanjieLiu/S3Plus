import os

from torch import optim

from VQVAE import VQVAE
from shared import *
from simple_FC import SimpleFC
from dataloader_plus import MultiImgDataset
from torch.utils.data import DataLoader
from loss_counter import LossCounter, RECORD_PATH_DEFAULT
from train import split_into_three
from dataMaker_fixedPosition_plusPair import data_name_2_labels
from common_func import parse_label
from evaler_common import OperResult


def name_appd(name: str, path:str):
    return f'{name}_{path}'

def is_need_train(train_config):
    loss_counter = LossCounter([])
    train_record_path = train_config['train_record_path']
    task_name = train_config['task_name']
    iter_num = loss_counter.load_iter_num(name_appd(task_name, train_record_path))
    if train_config['max_iter_num'] > iter_num:
        print("Continue training")
        return True
    else:
        print("No more training is needed")
        return False

class OtherTask:
    def __init__(self, train_config, other_task_config):
        self.other_task_config = other_task_config
        self.train_config = train_config
        self.pretrained = self.load_pretrained()
        self.latent_code_1 = self.pretrained.latent_code_1
        self.num_class = other_task_config['num_class']
        self.simple_fc = None
        # self.simple_fc = SimpleFC(other_task_config['fc_network_config'], self.latent_code_1*2, self.num_class).to(DEVICE)
        task_name = other_task_config['task_name']
        self.fc_model_path = name_appd(task_name, other_task_config['fc_model_path'])
        train_set = MultiImgDataset(other_task_config['train_data_path'])
        eval_set = MultiImgDataset(other_task_config['eval_data_path'])
        self.batch_size = other_task_config['batch_size']
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size)
        self.eval_loader = DataLoader(eval_set, batch_size=self.batch_size)
        self.train_result_path = name_appd(task_name, other_task_config['train_record_path'])
        self.eval_result_path = name_appd(task_name, other_task_config['eval_record_path'])
        self.learning_rate = other_task_config['learning_rate']
        self.max_iter_num = other_task_config['max_iter_num']
        self.log_interval = other_task_config['log_interval']
        self.CE_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.MSE_loss = torch.nn.MSELoss(reduction='mean')

    def resume(self):
        self.simple_fc = SimpleFC(self.other_task_config['fc_network_config'], self.latent_code_1 * 2,
                                  self.num_class).to(DEVICE)
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
            print(f"No pretrained parameters found in {pretrained_path}")
            exit()

    def train(self):
        self.resume()
        self.simple_fc.train()
        train_loss_counter = LossCounter(['loss_z',
                                          'accu',
                                          'loss_recon'], self.train_result_path)
        eval_loss_counter = LossCounter(['loss_z',
                                         'accu',
                                         'loss_recon'], self.eval_result_path)
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
        classify_ab = self.simple_fc.classify_composition(ec_a, ec_b)
        recon_z_ab = self.simple_fc.recon_composition(ec_a, ec_b)
        loss_classify = self.CE_loss(classify_ab, tensor_y)
        loss_recon = self.MSE_loss(recon_z_ab, ec_c.detach())
        accu = (classify_ab.argmax(1) == tensor_y).float().mean().item()
        return classify_ab, tensor_y, loss_classify, loss_recon, accu, recon_z_ab

    def eval_dec_view(self, data_path, result_path):
        self.resume()
        os.makedirs(result_path, exist_ok=True)
        dataset = MultiImgDataset(data_path)
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        styles = [('-').join(f.split('-')[2:]) for f in dataset.f_list]
        all_oper_result = []
        for batch_ndx, sample in enumerate(loader):
            data, labels = sample
            label_a = [parse_label(x) for x in labels[0]]
            label_b = [parse_label(x) for x in labels[1]]
            label_c = [parse_label(x) for x in labels[2]]
            plus_result = []
            classify_ab, tensor_y, loss_classify, loss_recon, accu, recon_z_ab = self.fc_comp(sample)
            e_ab, e_q_loss = self.pretrained.vq_layer(recon_z_ab)
            recon_ab = self.pretrained.batch_decode_from_z(e_ab)
            for i in range(0, recon_z_ab.size(0)):
                plus_result.append(OperResult(label_a[i], label_b[i], label_c[i], e_ab[i], recon_ab[i]))
            all_oper_result.extend(plus_result)

        for i in range(len(all_oper_result)):
            all_oper_result[i].style = styles[i]
            all_oper_result[i].save_recon(result_path)

    def one_epoch(self, epoch_num, loss_counter: LossCounter, data_loader,
                  is_log, optimizer: torch.optim.Optimizer = None):
        for batch_ndx, sample in enumerate(data_loader):
            print(f'Epoch: {epoch_num}')
            if optimizer is not None:
                optimizer.zero_grad()
            ec_ab, tensor_y, loss_classify, loss_recon, accu, recon_z_ab = self.fc_comp(sample)
            loss_counter.add_values([
                loss_classify.item(),
                accu,
                loss_recon.item()
            ])
            if optimizer is not None:
                loss = loss_classify + loss_recon
                loss.backward()
                optimizer.step()
        if is_log:
            print(loss_counter.make_record(epoch_num))
            loss_counter.record_and_clear(RECORD_PATH_DEFAULT, epoch_num)







