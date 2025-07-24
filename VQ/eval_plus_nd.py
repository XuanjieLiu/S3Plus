import os
import random
from collections import defaultdict
from typing import List, Tuple, Dict

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.optim as optim
from dataloader_plus import MultiImgDataset
from loss_counter import LossCounter
from VQVAE import VQVAE
from shared import *
from eval_common import draw_scatter_point_line
import matplotlib
import matplotlib.markers
import matplotlib.pyplot as plt
from scipy import stats
from common_func import parse_label, solve_label_emb_one2one_matching

matplotlib.use('AGG')
MODEL_PATH = 'curr_model.pt'
EVAL_ROOT = 'eval_z'

MIN_KS_NUM = 1


class EncZ:
    def __init__(self, label, z):
        self.label = label
        self.z = z


class PlusZ:
    def __init__(self, label_a, label_b, plus_c_z, plus_c_z_cycle=None):
        """
        Represents the result of a plus operation on two embeddings.
        :param label_a: the label of a.
        :param label_b: the label of b.
        :param plus_c_z: the emb of plus operation of a and b.
        :param plus_c_z_cycle: the encoded emb of the reconstruction of plus_c_z.
        """
        self.plus_c_z = plus_c_z
        self.label_c = label_a + label_b
        self.plus_c_z_cycle = plus_c_z_cycle


def plot_plusZ_against_label(
        all_enc_z,
        all_plus_z,
        eval_path,
        n_rows: int = 1,
        n_cols: int = -1,
        is_scatter_lines=False,
        is_gird=False,
        sub_title: str = '',
        y_label: str = '',
        sub_fig_size=5
):
    dim_z = all_enc_z[0].z.size(0)
    n_row = n_rows
    n_col = dim_z if n_cols == -1 else n_cols
    fig, axs = plt.subplots(n_row, n_col, sharey='all', sharex='all',
                            figsize=(n_col * sub_fig_size, n_row * sub_fig_size), squeeze=False)
    enc_x = [ob.label for ob in all_enc_z]
    plus_x = [ob.label_c for ob in all_plus_z]
    for i in range(0, dim_z):
        enc_y = [ob.z.cpu()[i].item() for ob in all_enc_z]
        plus_y = [ob.plus_c_z.cpu()[i].item() for ob in all_plus_z]
        axs.flat[i].scatter(enc_x, enc_y, edgecolors='blue', label='Encoder output', facecolors='none', s=60)
        axs.flat[i].scatter(plus_x, plus_y, edgecolors='red', label='Plus output', facecolors='none', s=20)
        axs.flat[i].set(xlabel='Num of Points on the card', xticks=range(0, max(enc_x) + 1))
        axs.flat[i].set(ylabel=y_label)
        axs.flat[i].set_title(f"{sub_title} ({i + 1})")
        axs.flat[i].grid(is_gird)
        if is_scatter_lines:
            draw_scatter_point_line(axs.flat[i], [*enc_x, *plus_x], [*enc_y, *plus_y])
    for ax in axs.flat:
        ax.label_outer()
    plt.legend()
    plt.savefig(f'{eval_path}.png')
    plt.cla()
    plt.clf()
    plt.close()


class LabelKs:
    def __init__(self, label: int):
        self.label = label
        self.enc_z_list = []
        self.plus_z_list = []

    def calc_min_len(self):
        return min(len(self.enc_z_list), len(self.plus_z_list))

    def calc_ks(self):
        min_len = self.calc_min_len()
        sample1 = random.sample(self.enc_z_list, min_len)
        sample2 = random.sample(self.plus_z_list, min_len)
        ks, p_value = stats.ks_2samp(sample1, sample2)
        return ks, p_value

    def calc_accu(self):
        total = len(self.plus_z_list)
        assert total != 0, f"plus_z_list of card {self.label} is empty."
        assert len(self.enc_z_list) != 0, f"enc_z_list of card {self.label} is empty."
        mode_enc = stats.mode(self.enc_z_list, keepdims=False)[0]
        correct = len(list(filter(lambda x: x == mode_enc, self.plus_z_list)))
        accu = correct / total
        return accu


def calc_ks_enc_plus_z(enc_z: List[EncZ], plus_z: List[PlusZ]):
    min_label = min(enc_z, key=lambda x: x.label).label
    max_label = max(enc_z, key=lambda x: x.label).label
    label_dict = {}
    for i in range(min_label, max_label + 1):
        label_dict[i] = LabelKs(i)
    for item in enc_z:
        label_dict[item.label].enc_z_list.append(item.z.cpu().item())
    for item in plus_z:
        label_dict[item.label_c].plus_z_list.append(item.plus_c_z.cpu().item())
    label_ks_list = list(label_dict.values())
    ks_all = 0
    accu_all = 0
    item_num = 0
    for item in label_ks_list:
        min_len = item.calc_min_len()
        if min_len >= MIN_KS_NUM:
            ks_all += item.calc_ks()[0]
            accu_all += item.calc_accu()
            item_num += 1
    ks_mean = ks_all / item_num
    accu_mean = accu_all / item_num
    return round(ks_mean, 4), round(accu_mean, 4)



def _mode_emb_label_dict(enc_z: List[EncZ]) -> Dict[int, int]:
    """
    根据 enc_z 中每个 embedding 的 label 众数，返回 emb → label 的映射。
    """
    emb_to_labels = defaultdict(list)
    for item in enc_z:
        emb_to_labels[int(item.z.item())].append(item.label)
    # 取众数
    return {
        emb: int(stats.mode(labels, keepdims=False)[0])
        for emb, labels in emb_to_labels.items()
    }


def _one2one_emb_label_dict(enc_z: List[EncZ]) -> Dict[int, int]:
    """
    调用 Hungarian 算法，返回一对一匹配的 emb → label 映射。
    """
    emb_idx_list = [int(item.z.item()) for item in enc_z]
    label_list = [item.label for item in enc_z]
    emb_label_pairs, _ = solve_label_emb_one2one_matching(emb_idx_list, label_list)
    # emb_label_pairs 是 (label, emb) 对
    return {emb: int(label) for label, emb in emb_label_pairs}


def _plus_accuracy(
        emb_label_dict: Dict[int, int],
        plus_z: List[PlusZ]
) -> Tuple[float, float]:
    """
    统一的准确率计算逻辑：对 plus_c_z 与 plus_c_z_cycle 两种情况分别计数。
    """
    n_total = len(plus_z)
    if n_total == 0:
        raise ValueError("plus_z is empty.")
    n_correct = 0
    n_correct_cycle = 0

    for item in plus_z:
        true_label = int(item.label_c)
        z1 = int(item.plus_c_z.item())
        z2 = int(item.plus_c_z_cycle.item())
        if emb_label_dict.get(z1) == true_label:
            n_correct += 1
        if emb_label_dict.get(z2) == true_label:
            n_correct_cycle += 1

    return n_correct / n_total, n_correct_cycle / n_total


def calc_multi_emb_plus_accu(
        enc_z: List[EncZ],
        plus_z: List[PlusZ]
) -> Tuple[float, float]:
    """
    多 Embedding → 多 Label 情形：embedding 通过众数关联 label。
    """
    emb_label_dict = _mode_emb_label_dict(enc_z)
    return _plus_accuracy(emb_label_dict, plus_z)


def calc_one2one_plus_accu(
        enc_z: List[EncZ],
        plus_z: List[PlusZ]
) -> Tuple[float, float]:
    """
    一对一情形：使用 Hungarian 算法匹配 embedding 与 label。
    """
    emb_label_dict = _one2one_emb_label_dict(enc_z)
    return _plus_accuracy(emb_label_dict, plus_z)


def calc_plus_z_self_cycle_consistency(plus_z: List[PlusZ]):
    assert len(plus_z) != 0, "plus_z is empty."
    n_total = len(plus_z)
    n_consistent = 0
    for item in plus_z:
        if int(item.plus_c_z.item()) == int(item.plus_c_z_cycle.item()):
            n_consistent += 1
    return n_consistent / n_total


def calc_plus_z_mode_emb_label_cycle_consistency(enc_z: List[EncZ], plus_z: List[PlusZ]):
    """
    计算 plus_z 中的 plus_c_z 与 plus_c_z_cycle 的一致性。
    通过 enc_z 中的 embedding 众数来确定每个 embedding 对应的 label。
    根据判断 plus_c_z 和 plus_c_z_cycle 是否对应同一个 label 计算 consistency。
    :param enc_z: List of EncZ objects, each containing a label and an embedding index.
    :param plus_z: List of PlusZ objects, each containing a label and embedding indices for the plus operation.
    :return: Tuple of three floats:
        - consistency: float, the proportion of plus_c_z and plus_c_z_cycle that match the same label.
        - recognized_c_z: float, the proportion of plus_c_z that are recognized by enc_z.
        - recognized_c_z_cycle: float, the proportion of plus_c_z_cycle that are recognized by enc_z.
    """
    assert len(plus_z) != 0, "plus_z is empty."
    assert len(enc_z) != 0, "enc_z is empty."
    emb_label_dict = _mode_emb_label_dict(enc_z)
    n_consistent = 0
    n_consistent_total = 0
    n_total = len(plus_z)
    n_recognized_c_z = 0
    n_recognized_c_z_cycle = 0
    for item in plus_z:
        z1 = int(item.plus_c_z.item())
        z2 = int(item.plus_c_z_cycle.item())
        label1 = emb_label_dict.get(z1)
        label2 = emb_label_dict.get(z2)
        if label1 is not None:
            n_recognized_c_z += 1
        if label2 is not None:
            n_recognized_c_z_cycle += 1
        if label1 is not None and label2 is not None:
            n_consistent_total += 1
            if label1 == label2:
                n_consistent += 1
    consistent_rate = n_consistent / n_consistent_total if n_consistent_total > 0 else 0
    return consistent_rate, n_recognized_c_z / n_total, n_recognized_c_z_cycle / n_total


class VQvaePlusEval:
    def __init__(self, config, model_path=None, loaded_model: VQVAE = None):
        self.config = config
        self.zc_dim = config['latent_embedding_1'] * config['embedding_dim']
        if loaded_model is not None:
            self.model = loaded_model
        elif model_path is not None:
            self.model = VQVAE(config).to(DEVICE)
            self.reload_model(model_path)
            print(f"Model is loaded from {model_path}")
        else:
            self.model = VQVAE(config).to(DEVICE)
            print("No model is loaded")
        self.model.eval()
        self.isVQStyle = config['isVQStyle']

    def reload_model(self, model_path):
        self.model.load_state_dict(self.model.load_tensor(model_path))

    def load_plusZ_eval_data(self, data_loader: DataLoader, is_find_index=True):
        all_enc_z = []
        all_plus_z = []
        for batch_ndx, sample in enumerate(data_loader):
            enc_z_list = []
            plus_z_list = []
            data, labels = sample
            data = [x.to(DEVICE) for x in data]
            za = self.model.batch_encode_to_z(data[0])[0]
            zb = self.model.batch_encode_to_z(data[1])[0]
            zc = self.model.batch_encode_to_z(data[2])[0]
            z_s = za[..., self.zc_dim:]
            if self.isVQStyle:
                z_s = self.model.vq_layer(z_s)[0]
            za = za[..., 0:self.zc_dim]
            zb = zb[..., 0:self.zc_dim]
            zc = zc[..., 0:self.zc_dim]
            label_a = [parse_label(x) for x in labels[0]]
            label_b = [parse_label(x) for x in labels[1]]
            label_c = [parse_label(x) for x in labels[2]]
            plus_c = self.model.plus(za, zb)[0]
            plus_c_cycle = self.model.batch_encode_to_z(
                self.model.batch_decode_from_z(torch.cat((plus_c, z_s), -1))
            )[0]
            plus_c_cycle = plus_c_cycle[..., 0:self.zc_dim]
            if is_find_index:
                idx_z_a = self.model.find_indices(za, False)
                idx_z_b = self.model.find_indices(zb, False)
                idx_z_c = self.model.find_indices(zc, False)
                idx_plus_c = self.model.find_indices(plus_c, False)
                idx_plus_c_cycle = self.model.find_indices(plus_c_cycle, False)
            else:
                idx_z_a = za
                idx_z_b = zb
                idx_z_c = zc
                idx_plus_c = plus_c
                idx_plus_c_cycle = plus_c_cycle
            for i in range(0, za.size(0)):
                enc_z_list.append(EncZ(label_a[i], idx_z_a[i]))
                enc_z_list.append(EncZ(label_b[i], idx_z_b[i]))
                enc_z_list.append(EncZ(label_c[i], idx_z_c[i]))
                plus_z_list.append(PlusZ(label_a[i], label_b[i], idx_plus_c[i], idx_plus_c_cycle[i]))
            all_enc_z.extend(enc_z_list)
            all_plus_z.extend(plus_z_list)
        return all_enc_z, all_plus_z

    def eval(self, eval_path, dataloader: DataLoader):
        all_enc_z, all_plus_z = self.load_plusZ_eval_data(dataloader)
        plot_plusZ_against_label(all_enc_z, all_plus_z, eval_path)

    def eval_multi_emb_accu(self, eval_data_loader: DataLoader):
        all_enc_z, all_plus_z = self.load_plusZ_eval_data(eval_data_loader, is_find_index=True)
        accu = calc_multi_emb_plus_accu(all_enc_z, all_plus_z)
        return accu

# if __name__ == "__main__":
#     os.makedirs(EVAL_ROOT, exist_ok=True)
#     dataset = Dataset(CONFIG['train_data_path'])
#     loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
#     evaler = VQvaePlusEval(CONFIG, loader, model_path=MODEL_PATH)
#     result_path = os.path.join(EVAL_ROOT, f'plus_eval_{MODEL_PATH}.png')
#     evaler.eval(result_path)
