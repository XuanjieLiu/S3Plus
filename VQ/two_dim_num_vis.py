import sys
import os
import numpy as np
from collections import defaultdict
from importlib import reload
from torch.utils.data import DataLoader
from dataloader import SingleImgDataset, load_enc_eval_data
from VQ.VQVAE import VQVAE, get_all_code_embs
from shared import *
import matplotlib.markers
import matplotlib.pyplot as plt
from VQ.eval_common import EvalHelper
from matplotlib import collections as matcoll
from common_func import add_gaussian_noise
from sklearn.manifold import TSNE


matplotlib.use('AGG')
COLOR_LIST = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'hotpink', 'gray', 'steelblue', 'olive']


def plot_z_against_label(num_z, num_labels, eval_path=None, eval_helper: EvalHelper = None):
    fig, axs = plt.subplots(1, num_z.size(1), figsize=(num_z.size(1) * 7, 5))
    if num_z.size(1) == 1:
        axs = [axs]
    for i in range(0, num_z.size(1)):
        x = num_labels
        y = num_z[:, i].detach().cpu()
        axs[i].scatter(x, y)
        axs[i].set_title(f'z{i + 1}')
        axs[i].set(xlabel='Num of Points on the card', xticks=range(0, 18))
        if eval_helper is not None:
            eval_helper.draw_scatter_point_line_or_grid(axs[i], i, x, y)
            eval_helper.set_axis(axs[i], i)
        else:
            axs[i].grid(True)

    # for ax in axs.flat:
    #     ax.label_outer()
    if eval_path is None:
        plt.show()
    else:
        plt.savefig(eval_path)
        plt.cla()
        plt.clf()
        plt.close()


def num_position_in_two_dim_repr(plt_func: matplotlib.pyplot, num_z, num_labels, all_embs=None):
    assert len(
        num_z[0]) == 2, f"The representation dimension of a number should be two, but got {len(num_z[0])} instead."
    sorted_label = sorted(num_labels)
    sorted_indices = [i[0] for i in sorted(enumerate(num_labels), key=lambda x: x[1])]
    sorted_num_z = [num_z[i] for i in sorted_indices]
    X = [item[0] for item in sorted_num_z]
    Y = [item[1] for item in sorted_num_z]
    max_repeating_num = find_most_frequent_elements_repeating_num(num_labels)
    for i in range(0, len(num_z)):
        plt_func.scatter(X[i], Y[i],
                         marker=f'${sorted_label[i]}$',
                         s=200,
                         alpha=min(1, 1 / max_repeating_num * 1.3),
                         c=COLOR_LIST[sorted_label[i] % len(COLOR_LIST)])
        if all_embs is None:
            draw_scatter_gird(plt_func.gca(), X[i], Y[i])
    plt_func.plot(X, Y, linestyle='dashed', linewidth=0.5)
    if all_embs is not None:
        embs_x = [item[0] for item in all_embs]
        embs_y = [item[1] for item in all_embs]
        plt_func.scatter(embs_x, embs_y, marker='o', s=1, c='navy')
    plt_func.xlabel('z1')
    plt_func.ylabel('z2')


def plot_num_position_in_two_dim_repr(num_z, num_labels, result_path=None, x_limit=None, y_limit=None, all_embs=None):
    plt.figure(figsize=(5, 5))
    num_position_in_two_dim_repr(plt, num_z, num_labels, all_embs)
    if x_limit is not None:
        plt.xlim(x_limit[0], x_limit[1])
    if y_limit is not None:
        plt.ylim(y_limit[0], y_limit[1])
    if x_limit is None and y_limit is None:
        plt.axis('equal')
    if result_path is None:
        plt.show()
    else:
        plt.savefig(f'{result_path}.png')
        plt.cla()
        plt.clf()
        plt.close()


def plot_num_position_in_three_dim_repr(num_z, num_labels, result_path=None, x_limit=None, y_limit=None, all_embs=None):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    assert len(
        num_z[0]) == 3, f"The representation dimension of a number should be three, but got {len(num_z[0])} instead."
    sorted_label = sorted(num_labels)
    sorted_indices = [i[0] for i in sorted(enumerate(num_labels), key=lambda x: x[1])]
    sorted_num_z = [num_z[i] for i in sorted_indices]
    X = [item[0] for item in sorted_num_z]
    Y = [item[1] for item in sorted_num_z]
    Z = [item[2] for item in sorted_num_z]
    max_repeating_num = find_most_frequent_elements_repeating_num(num_labels)
    for i in range(0, len(num_z)):
        ax.scatter(X[i], Y[i], Z[i],
                   marker=f'${sorted_label[i]}$',
                   s=200,
                   alpha=min(1, 1 / max_repeating_num * 1.3),
                   c=COLOR_LIST[sorted_label[i] % len(COLOR_LIST)])
    plt.plot(X, Y, Z, linestyle='dashed', linewidth=0.5)
    if all_embs is not None:
        embs_x = [item[0] for item in all_embs]
        embs_y = [item[1] for item in all_embs]
        embs_z = [item[2] for item in all_embs]
        ax.scatter(embs_x, embs_y, embs_z, marker='o', s=1, c='navy')
    ax.set_xlabel('z1')
    ax.set_ylabel('z2')
    ax.set_zlabel('z3')
    if x_limit is not None:
        ax.set_xlim(x_limit)
    if y_limit is not None:
        ax.set_ylim(y_limit)
    if result_path is None:
        plt.show()
    else:
        plt.savefig(f'{result_path}.png')
        plt.cla()
        plt.clf()
        plt.close()


def find_most_frequent_elements_repeating_num(arr):
    nd_array = np.array(arr)
    unique_elements, counts = np.unique(nd_array, return_counts=True)
    max_count = np.max(counts)
    return max_count


def draw_scatter_gird(ax: plt.Axes, x, y):
    horizontal_line = [(0, y), (x, y)]
    vertical_line = [(x, 0), (x, y)]
    lines = [horizontal_line, vertical_line]
    linecoll = matcoll.LineCollection(lines, linewidths=0.2)
    ax.add_collection(linecoll)


def make_result_name(result_path, nna_score=None):
    if result_path is not None and nna_score is not None:
        result_name = f'{result_path}_nna_{round(nna_score, 2)}'
    elif result_path is not None:
        result_name = f'{result_path}_no_nna'
    else:
        result_name = None
    return result_name


def sample_unique_with_indices(nums):
    """
    从 nums 中采样唯一值，并返回这些值及其对应的索引。
    :param nums: 包含数字的列表或数组。
    :return:
        sampled_values: 采样的唯一值列表。
        sampled_indices: 这些唯一值在原 nums 中的索引列表。
    """
    index_map = defaultdict(list)
    # 建立每个数字对应的所有索引列表
    for idx, val in enumerate(nums):
        index_map[val].append(idx)
    sampled_values = []
    sampled_indices = []
    for val, indices in index_map.items():
        sampled_idx = np.random.choice(indices)
        sampled_values.append(nums[sampled_idx])
        sampled_indices.append(sampled_idx)
    return sampled_values, sampled_indices


def multiple_nearest_neighbor_analysis(num_z_c, num_labels, n_times=10):
    score_list = []
    for i in range(n_times):
        sampled_labels, sampled_indices = sample_unique_with_indices(num_labels)
        sampled_num_z_c = num_z_c[sampled_indices]
        score = nearest_neighbor_analysis(sampled_num_z_c, sampled_labels, verbose=False)
        score_list.append(score)
    mean_score = np.mean(score_list)
    return mean_score


class MumEval:
    def __init__(self, config, model_path=None, loaded_model: VQVAE = None):
        self.config = config
        self.embedding_dim = config['embedding_dim']
        self.multi_num_embeddings = config['multi_num_embeddings']
        if self.multi_num_embeddings is None:
            self.latent_embedding_1 = config['latent_embedding_1']
        else:
            self.latent_embedding_1 = len(self.multi_num_embeddings)
        self.latent_code_1 = self.latent_embedding_1 * self.embedding_dim
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

    def reload_model(self, model_path):
        self.model.load_state_dict(self.model.load_tensor(model_path))

    def num_eval_two_dim(self, data_loader, result_path=None, is_show_all_emb=True, is_draw_graph=True, use_tsne=True):
        num_z_c, num_labels = self.get_num_z_and_labels(data_loader)
        has_duplicate_labels = len(set(num_labels)) < len(num_labels)
        if not has_duplicate_labels:
            nna_score = nearest_neighbor_analysis(num_z_c, num_labels)
        else:
            nna_score = multiple_nearest_neighbor_analysis(num_z_c, num_labels)
        print(f'Nearest neighbor analysis score: {nna_score}')
        if is_draw_graph:
            result_name = make_result_name(result_path, nna_score)
            self.plot_num_position_graph(num_z_c, num_labels, is_show_all_emb, result_name, use_tsne)
        return nna_score


    def get_num_z_and_labels(self, data_loader):
        num_z, num_labels = load_enc_eval_data(
            data_loader,
            lambda x:
            self.model.batch_encode_to_z(x)[0]
        )
        num_z = num_z.cpu().detach().numpy()
        num_z_c = num_z[:, :self.latent_code_1]
        return num_z_c, num_labels


    def plot_num_position_graph(self, num_z_c, num_labels, is_show_all_emb=True, result_name=None, use_tsne=False):
        all_embs = get_all_code_embs(self.model) if is_show_all_emb else None
        if self.latent_code_1 == 2:
            plot_num_position_in_two_dim_repr(num_z_c, num_labels, result_name, all_embs=all_embs)
        elif use_tsne and self.latent_code_1 >= 3:
            tsne = TSNE(n_components=2, perplexity=5, random_state=42)
            trans_num_z_c = tsne.fit_transform(num_z_c)
            plot_num_position_in_two_dim_repr(trans_num_z_c, num_labels, result_name, all_embs=None)
        elif self.latent_code_1 == 3:
            plot_num_position_in_three_dim_repr(num_z_c, num_labels, result_name, all_embs=all_embs)

    def num_eval_two_dim_with_gaussian_noise(self, data_loader, result_path=None, is_show_all_emb=True,
                                             is_draw_graph=True, noise_batch: int = 10, noise_std=0.2):
        nna_score_list = []
        num_z_c_all = np.array([])
        num_labels_all = []
        for i in range(0, noise_batch):
            num_z, num_labels = load_enc_eval_data(
                data_loader,
                lambda x: self.model.batch_encode_to_z(add_gaussian_noise(x, mean=0, std=noise_std))[0]
            )
            num_z = num_z.cpu().detach().numpy()
            num_z_c = num_z[:, :self.latent_code_1]
            has_duplicate_labels = len(set(num_labels)) < len(num_labels)
            if not has_duplicate_labels:
                nna_score = nearest_neighbor_analysis(num_z_c, num_labels)
            else:
                nna_score = multiple_nearest_neighbor_analysis(num_z_c, num_labels)
            nna_score_list.append(nna_score)
            num_z_c_all = np.concatenate((num_z_c_all, num_z_c), axis=0) if num_z_c_all.size else num_z_c
            num_labels_all.extend(num_labels)
        if len(nna_score_list) == 0:
            nna_score_mean = None
        else:
            nna_score_mean = round(np.mean(nna_score_list), 3)
        print(f'Nearest neighbor analysis score: {nna_score_mean}')
        if is_draw_graph and (self.latent_code_1 == 2 or self.latent_code_1 == 3):
            result_name = make_result_name(result_path, nna_score_mean)
            self.plot_num_position_graph(num_z_c_all, num_labels_all, is_show_all_emb, result_name)
        return nna_score_mean


def nearest_neighbor_analysis(num_z_c, num_labels, verbose=False):
    ascend_sorted_label = sorted(num_labels)
    ascend_sorted_indices = [i[0] for i in sorted(enumerate(num_labels), key=lambda x: x[1])]
    ascend_sorted_num_z = [num_z_c[i] for i in ascend_sorted_indices]

    def is_the_next_num_the_closest(sorted_num_z, sorted_label):
        valid_nearest_neighbor = 0
        for i in range(len(sorted_label) - 1):
            z = sorted_num_z[i]
            next_label_z = sorted_num_z[i + 1:]
            distances = np.linalg.norm(next_label_z - z, axis=1)
            if is_the_unique_min_num(distances[0], distances):
                valid_nearest_neighbor += 1
            else:
                if verbose:
                    print(f'Nearest neighbor analysis failed at number {sorted_label[i]} and {sorted_label[i + 1]}.')
        return valid_nearest_neighbor

    ascend_valid_nearest_neighbor = is_the_next_num_the_closest(ascend_sorted_num_z, ascend_sorted_label)
    descend_sorted_label = ascend_sorted_label[::-1]
    descend_sorted_num_z = ascend_sorted_num_z[::-1]
    descend_valid_nearest_neighbor = is_the_next_num_the_closest(descend_sorted_num_z, descend_sorted_label)
    all_valid_nearest_neighbor = ascend_valid_nearest_neighbor + descend_valid_nearest_neighbor
    total_num = (len(num_labels) - 1) * 2
    return all_valid_nearest_neighbor / total_num


def is_the_unique_min_num(num, num_list):
    return num == min(num_list) and list(num_list).count(num) == 1


def test_two_dim_vis():
    matplotlib.use('tkagg')
    EXP_ROOT_PATH = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/exp')

    DATASET_PATH = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../dataset/(0,20)-FixedPos-oneStyle')
    EXP_NAME = '2024.05.31_5vq_Zc[3]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_AssocFullsymmCommu'
    # EXP_NAME = '2024.05.10_100vq_Zc[1]_Zs[0]_edim2_[0-20]_plus1024_1_tripleSet_AssocFullsymmCommu'
    SUB_EXP = 1
    CHECK_POINT = 'checkpoint_50000.pt'

    exp_path = os.path.join(EXP_ROOT_PATH, EXP_NAME)
    sys.path.append(exp_path)
    os.chdir(exp_path)
    print(f'Exp path: {exp_path}')
    t_config = __import__('train_config')
    reload(t_config)
    sys.path.pop()
    model_path = os.path.join(exp_path, str(SUB_EXP), CHECK_POINT)
    single_img_eval_set = SingleImgDataset(DATASET_PATH)
    single_img_eval_loader = DataLoader(single_img_eval_set, batch_size=256)
    evaler = MumEval(t_config.CONFIG, model_path)
    evaler.num_eval_two_dim(single_img_eval_loader)


if __name__ == "__main__":
    test_two_dim_vis()
