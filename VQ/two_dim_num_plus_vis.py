import sys
import os
from importlib import reload

import torch
from torch.utils.data import DataLoader
from dataloader import SingleImgDataset, load_enc_eval_data
from VQ.VQVAE import VQVAE
from shared import *
import matplotlib.markers
import matplotlib.pyplot as plt
from VQ.eval_common import EvalHelper
from matplotlib import collections as matcoll
from two_dim_num_vis import plot_num_position_in_two_dim_repr, draw_scatter_gird
from common_func import sort_X_by_Y, sorted_idx, calc_tensor_seq_limits

matplotlib.use('AGG')


# inner_point = (b - a) * k + a
def k_inner_point(a, b, k):
    return (b - a) * k + a


def gen_k_list(decimal, num, base):
    assert decimal * num < 1, f'decimal * num should be less than 1, but got {decimal * num}'
    k_list = [round(decimal * i, 1) + base for i in range(1, num+1)]
    return k_list


class MumEval:
    def __init__(self, config, model_path, data_set_path):
        self.config = config
        self.embedding_dim = config['embedding_dim']
        self.multi_num_embeddings = config['multi_num_embeddings']
        if self.multi_num_embeddings is None:
            self.latent_embedding_1 = config['latent_embedding_1']
        else:
            self.latent_embedding_1 = len(self.multi_num_embeddings)
        self.latent_code_1 = self.latent_embedding_1 * self.embedding_dim
        dataset = SingleImgDataset(data_set_path)
        self.batch_size = config['batch_size']
        self.loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model = VQVAE(config).to(DEVICE)
        if model_path is not None:
            self.reload_model(model_path)
        self.model.eval()

    def reload_model(self, model_path):
        self.model.load_state_dict(self.model.load_tensor(model_path))

    def plus_z1seq_by_repeat_z2(self, z1_seq, z2):
        z2_repeat = z2.repeat(len(z1_seq), 1)
        e_plus, e_q_loss, z_plus = self.model.plus(z1_seq, z2_repeat)
        return e_plus.detach(), z_plus.detach()

    def integer_plus(self, num_zc, num_labels, plus_times=5):
        assert plus_times <= len(num_zc), f'plus_times: {plus_times} should be less than len(num_zc): {len(num_zc)}'
        e_seq = []
        z_seq = []
        label_seq = []
        labels = []
        assert len(num_zc) == len(num_labels), f'len num_zc: {len(num_zc)} != len num_labels: {len(num_labels)}'
        for i in range(plus_times):
            e_plus, z_plus = self.plus_z1seq_by_repeat_z2(num_zc, num_zc[i])
            label_plus = [label + num_labels[i] for label in num_labels]
            e_seq.append(e_plus)
            z_seq.append(z_plus)
            label_seq.append(label_plus)
            labels.append(num_labels[i])
        return e_seq, z_seq, label_seq, labels

    def draw_integer_plus(self, num_zc, num_labels, result_path=None):
        if result_path is not None:
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
        e_seq, z_seq, label_seq, labels = self.integer_plus(num_zc, num_labels)
        labels = [f'0{num}' if num < 10 else str(num) for num in labels]
        all_e_seq = [num_zc] + e_seq
        all_z_seq = [num_zc] + z_seq
        all_label_seq = [num_labels] + label_seq
        all_labels = ['0_ori'] + [str(item) for item in labels]

        def draw(all_zc_seq, all_label_seq, all_labels, result_path):
            os.makedirs(result_path, exist_ok=True)
            tensor_e_seq = torch.cat(all_zc_seq, dim=0).cpu().detach()
            x_limits, y_limits = calc_tensor_seq_limits(tensor_e_seq)
            for i in range(len(all_zc_seq)):
                save_path = os.path.join(result_path, f'{all_labels[i]}.png')
                plot_num_position_in_two_dim_repr(all_zc_seq[i].cpu().detach(), all_label_seq[i], save_path, x_limits, y_limits)

        e_all_labels = [f'e_{label}' for label in all_labels]
        e_result_path = os.path.join(result_path, 'e')
        draw(all_e_seq, all_label_seq, e_all_labels, e_result_path)

        z_all_labels = [f'z_{label}' for label in all_labels]
        z_result_path = os.path.join(result_path, 'z')
        draw(all_z_seq, all_label_seq, z_all_labels, z_result_path)

    def gen_decimal_between_a_b(self, a, b, k_list, num_zc, num_labels):
        a_idx = num_labels.index(a)
        b_idx = num_labels.index(b)
        assert a_idx != -1, f'Cannot find {a} in num_labels'
        assert b_idx != -1, f'Cannot find {b} in num_labels'
        a_zc = num_zc[a_idx]
        b_zc = num_zc[b_idx]
        zc_list = [k_inner_point(a_zc, b_zc, k) for k in k_list]
        decimal_list = [k_inner_point(a, b, k) for k in k_list]
        return zc_list, decimal_list

    def draw_decimal_plus(self, num_zc, num_labels, k_list=None, result_path=None):
        if result_path is not None:
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
        if k_list is None:
            k_list = [round(0.2 * i, 1) for i in range(1, 5)]
        zc_list, decimal_list = self.gen_decimal_between_a_b(0, 1, k_list, num_zc, num_labels)
        zc_plus_list = [self.plus_z1seq_by_repeat_z2(num_zc, zc)[1] for zc in zc_list]
        label_plus_list = [[num + decimal for num in num_labels] for decimal in decimal_list]
        all_z_seq = torch.cat([num_zc] + zc_plus_list, dim=0).cpu().detach()
        all_labels = num_labels + [item for sublist in label_plus_list for item in sublist]
        sorted_label_idx = sorted_idx(all_labels)
        sorted_all_z_seq = all_z_seq[sorted_label_idx].cpu().detach()
        limits_x, limits_y = calc_tensor_seq_limits(all_z_seq)
        sorted_all_labels = [all_labels[i] for i in sorted_label_idx]
        int_X = [item[0] for item in num_zc.cpu().detach()]
        int_Y = [item[1] for item in num_zc.cpu().detach()]
        for i in range(0, len(num_zc)):
            plt.scatter(int_X[i], int_Y[i], marker=f'${num_labels[i]}$', s=60)
            draw_scatter_gird(plt.gca(), int_X[i], int_Y[i])
        color_list = ['orange', 'green', 'blue', 'red', 'black', 'purple', 'pink', 'brown', 'gray']
        for i in range(0, len(k_list)):
            decimal_z = zc_plus_list[i].cpu().detach()
            decimal_X = [item[0] for item in decimal_z]
            decimal_Y = [item[1] for item in decimal_z]
            plt.scatter(decimal_X, decimal_Y, marker='o', s=5, color=color_list[i], label=f'x+{k_list[i]}')
        all_X = [item[0] for item in sorted_all_z_seq]
        all_Y = [item[1] for item in sorted_all_z_seq]
        plt.plot(all_X, all_Y, linestyle='dashed', linewidth=0.5)
        plt.xlabel('z1')
        plt.ylabel('z2')
        plt.xlim(limits_x)
        plt.ylim(limits_y)
        plt.legend()
        if result_path is None:
            plt.show()
        else:
            plt.savefig(result_path)
            plt.cla()
            plt.clf()
            plt.close()


        print('a')

    def eval_multi_decimal_plus(self, k_lists=None, result_path=None):
        num_z, num_labels = load_enc_eval_data(
            self.loader,
            lambda x:
            self.model.batch_encode_to_z(x)[0]
        )
        num_z_c = num_z[:, :self.latent_code_1]
        sorted_num_idx = sorted_idx(num_labels)
        sorted_num_z_c = num_z_c[sorted_num_idx]
        sorted_num_labels = [num_labels[i] for i in sorted_num_idx]
        if result_path is not None:
            result_dir = os.path.join(result_path, 'multi_decimal_plus')
            os.makedirs(result_dir, exist_ok=True)
        else:
            result_dir = None
        for i in range(0, len(k_lists)):
            base = int(k_lists[i][0])
            result_name = os.path.join(result_dir, f'{base}.png') if result_path is not None else None
            self.draw_decimal_plus(sorted_num_z_c, sorted_num_labels, k_lists[i], result_name)

    def num_eval_two_dim_int_plus(self, result_path=None):
        num_z, num_labels = load_enc_eval_data(
            self.loader,
            lambda x:
                self.model.batch_encode_to_z(x)[0]
        )
        num_z_c = num_z[:, :self.latent_code_1]
        sorted_num_idx = sorted_idx(num_labels)
        sorted_num_z_c = num_z_c[sorted_num_idx]
        sorted_num_labels = [num_labels[i] for i in sorted_num_idx]
        self.draw_integer_plus(sorted_num_z_c, sorted_num_labels, result_path)



if __name__ == "__main__":
    matplotlib.use('tkagg')
    DATASET_PATH = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../dataset/(0,20)-FixedPos-oneStyle')
    EXP_ROOT_PATH = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/exp')
    EXP_NAME = '2023.12.17_multiStyle_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_2_realPair'
    SUB_EXP = 1
    CHECK_POINT = 'curr_model.pt'
    exp_path = os.path.join(EXP_ROOT_PATH, EXP_NAME)
    sys.path.append(exp_path)
    os.chdir(exp_path)
    print(f'Exp path: {exp_path}')
    t_config = __import__('train_config')
    reload(t_config)
    sys.path.pop()
    model_path = os.path.join(exp_path, str(SUB_EXP), CHECK_POINT)
    evaler = MumEval(t_config.CONFIG, model_path, DATASET_PATH)
    vis_results_path = os.path.join(exp_path, str(SUB_EXP), 'vis_results')
    evaler.num_eval_two_dim_int_plus(vis_results_path)
    k_lists = [gen_k_list(0.2, 4, i) for i in range(0, 6)]
    evaler.eval_multi_decimal_plus(k_lists, vis_results_path)
    # evaler.num_eval_two_dim()
