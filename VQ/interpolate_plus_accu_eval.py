import sys
import os
from importlib import reload
from torch.utils.data import DataLoader
from dataloader import SingleImgDataset, load_enc_eval_data
from dataloader_plus import Dataset
from VQ.VQVAE import VQVAE
from shared import *
from dataMaker_fixedPosition_plusPair import data_name_2_num
from common_func import EXP_ROOT

HALF = 0.5

def gen_interpolated_nums(start, end):
    i = start
    nums = []
    while i <= end:
        nums.append(i)
        i += 1
    return nums


def gen_all_plus_pairs(nums):
    a = []
    b = []
    c = []
    for i in range(0, len(nums)):
        for j in range(len(nums)-1-i, -1, -1):
            a.append(nums[i])
            b.append(nums[j])
            sum_ab = nums[i]+nums[j]
            c.append(sum_ab)
    return a, b, c


def remove_duplicate_pairs(a_list, b_list, c_list):
    new_a = []
    new_b = []
    new_c = []
    str_buffer_list = []
    for i in range(len(a_list)):
        str_buffer = f'{a_list[i]}-{b_list[i]}'
        if str_buffer not in str_buffer_list:
            str_buffer_list.append(str_buffer)
            new_a.append(a_list[i])
            new_b.append(b_list[i])
            new_c.append(c_list[i])
    return new_a, new_b, new_c


class InterpolatePlusAccuEval:
    def __init__(self, config, model: VQVAE=None, data_set_path=None):
        self.config = config
        self.model = VQVAE(config).to(DEVICE) if model is None else model
        self.model.eval()
        self.data_set_path = data_set_path if data_set_path is not None else config['single_img_eval_set_path']
        single_img_eval_set = SingleImgDataset(self.data_set_path)
        self.single_img_eval_loader = DataLoader(single_img_eval_set, batch_size=32)
        self.train_set = Dataset(config['train_data_path'])
        self.min_interpolated_num = None
        self.max_interpolated_num = None
        self.enc_num_dict = None
        self.itp_num_dict = None

        embedding_dim = config['embedding_dim']
        multi_num_embeddings = config['multi_num_embeddings']
        if multi_num_embeddings is None:
            latent_embedding_1 = config['latent_embedding_1']
        else:
            latent_embedding_1 = len(multi_num_embeddings)
        self.latent_code_1 = latent_embedding_1 * embedding_dim

    def is_valid_itp_num(self, num):
        return self.min_interpolated_num <= num <= self.max_interpolated_num


    def gen_train_interpolate_dataset(self):
        train_labels = [item[1] for item in self.train_set.data_list]
        train_a = [data_name_2_num(item[0]) for item in train_labels]
        train_b = [data_name_2_num(item[1]) for item in train_labels]
        train_c = [data_name_2_num(item[2]) for item in train_labels]
        itp_a = []
        itp_b = []
        itp_c = []
        for i in range(len(train_a)):
            a1 = train_a[i] - HALF
            b1 = train_b[i] + HALF
            if self.is_valid_itp_num(a1) and self.is_valid_itp_num(b1):
                itp_a.append(a1)
                itp_b.append(b1)
                itp_c.append(train_c[i])
            a2 = train_a[i] + HALF
            b2 = train_b[i] - HALF
            if self.is_valid_itp_num(a2) and self.is_valid_itp_num(b2):
                itp_a.append(a2)
                itp_b.append(b2)
                itp_c.append(train_c[i])
        clean_itp_a, clean_itp_b, clean_itp_c = remove_duplicate_pairs(itp_a, itp_b, itp_c)
        for i in range(len(clean_itp_a)):
            assert clean_itp_a[i] + clean_itp_b[i] == clean_itp_c[i]
        tensor_a = torch.stack([self.itp_num_dict[str(i)] for i in clean_itp_a], dim=0).to(DEVICE)
        tensor_b = torch.stack([self.itp_num_dict[str(i)] for i in clean_itp_b], dim=0).to(DEVICE)
        tensor_c = torch.stack([self.enc_num_dict[str(int(i))] for i in clean_itp_c], dim=0).to(DEVICE)
        return tensor_a, tensor_b, tensor_c


    def model_reload(self, checkpoint_path):
        self.model.load_state_dict(self.model.load_tensor(checkpoint_path))
        print('Model loaded from {}'.format(checkpoint_path))

    def init_num_dict(self):
        num_z, num_labels = load_enc_eval_data(
            self.single_img_eval_loader,
            lambda x:
            self.model.batch_encode_to_z(x)[0]
        )
        self.enc_num_dict = {}
        for i in range(len(num_z)):
            self.enc_num_dict[str(num_labels[i])] = num_z[i][..., 0:self.latent_code_1]
        min_label = min(num_labels)
        max_label = max(num_labels)
        self.min_interpolated_num = min_label + HALF
        self.max_interpolated_num = max_label - self.min_interpolated_num
        self.itp_num_dict = {}
        for i in range(min_label, max_label):
            self.itp_num_dict[str(i + HALF)] = \
                self.enc_num_dict[str(i)] + (self.enc_num_dict[str(i + 1)] - self.enc_num_dict[str(i)]) * HALF

    def gen_all_interpolate_dataset(self):
        interpolated_nums = gen_interpolated_nums(self.min_interpolated_num, self.max_interpolated_num)
        a, b, c = gen_all_plus_pairs(interpolated_nums)
        tensor_a = torch.stack([self.itp_num_dict[str(i)] for i in a], dim=0).to(DEVICE)
        tensor_b = torch.stack([self.itp_num_dict[str(i)] for i in b], dim=0).to(DEVICE)
        tensor_c = torch.stack([self.enc_num_dict[str(int(i))] for i in c], dim=0).to(DEVICE)
        return tensor_a, tensor_b, tensor_c

    def eval_interpolated_plus(self, tensor_a, tensor_b, tensor_c):
        interpolated_sum = self.model.plus(tensor_a, tensor_b)[0]
        interpolated_sum_idx = self.model.find_indices(interpolated_sum, False)
        c_idx = self.model.find_indices(tensor_c, False)
        total = len(interpolated_sum_idx)
        correct = sum([1 if interpolated_sum_idx[i] == c_idx[i] else 0 for i in range(total)])
        accu = correct / total
        return accu

    def eval(self, checkpoint_path):
        self.model_reload(checkpoint_path)
        self.init_num_dict()
        self.gen_train_interpolate_dataset()
        all_itp_a, all_itp_b, all_itp_c = self.gen_all_interpolate_dataset()
        train_itp_a, train_itp_b, train_itp_c = self.gen_train_interpolate_dataset()
        accu_all = self.eval_interpolated_plus(all_itp_a, all_itp_b, all_itp_c)
        accu_train = self.eval_interpolated_plus(train_itp_a, train_itp_b, train_itp_c)
        print(f'All interpolated plus accu: {accu_all}')
        print(f'Train interpolated plus accu: {accu_train}')
        return accu_all, accu_train




if __name__ == "__main__":
    EXP_NAME = '2023.09.25_100vq_Zc[1]_Zs[0]_edim2_[0-20]_plus1024_1'
    SUB_EXP = 16
    CHECK_POINT = './checkpoint_60000.pt'
    exp_path = os.path.join(EXP_ROOT, EXP_NAME)
    check_point_path = os.path.join(exp_path, str(SUB_EXP), CHECK_POINT)
    sys.path.append(exp_path)
    os.chdir(exp_path)
    print(f'Exp path: {exp_path}')
    t_config = __import__('train_config')
    reload(t_config)
    evaler = InterpolatePlusAccuEval(t_config.CONFIG)
    # evaler.model_reload(check_point_path)
    # evaler.gen_interpolate_dataset()
    accu_all, accu_train = evaler.eval(check_point_path)





