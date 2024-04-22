import sys
import os
import numpy as np

sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from eval_plus_nd import VQvaePlusEval, calc_ks_enc_plus_z, plot_plusZ_against_label
from common_func import load_config_from_exp_name, record_num_list, DATASET_ROOT, EXP_ROOT, \
    find_optimal_checkpoint_num_by_train_config
from dataloader_plus import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from VQVAE import VQVAE
from visual_imgs import VisImgs

EVAL_SETS = [
    {
        'name': 'Original_train',
        'path': os.path.join(DATASET_ROOT, 'multi_style_(4,4)_realPairs_plus(0,20)', 'train')
    },
    {
        'name': 'Original_test',
        'path': os.path.join(DATASET_ROOT, 'multi_style_(4,4)_realPairs_plus(0,20)', 'test')
    },
    {
        'name': 'New_color_train',
        'path': os.path.join(DATASET_ROOT, 'multi_style_realPairs_plus_eval_newColor', 'train')
    },
    {
        'name': 'New_color_test',
        'path': os.path.join(DATASET_ROOT, 'multi_style_realPairs_plus_eval_newColor', 'test')
    },
    {
        'name': 'New_shape_train',
        'path': os.path.join(DATASET_ROOT, 'multi_style_realPairs_plus_eval_newShape', 'train')
    },
    {
        'name': 'New_shape_test',
        'path': os.path.join(DATASET_ROOT, 'multi_style_realPairs_plus_eval_newShape', 'test')
    },
    {
        'name': 'New_style_train',
        'path': os.path.join(DATASET_ROOT, 'multi_style_realPairs_plus_eval_newShapeColor', 'train')
    },
    {
        'name': 'New_style_test',
        'path': os.path.join(DATASET_ROOT, 'multi_style_realPairs_plus_eval_newShapeColor', 'test')
    }
]
EXP_NAME_LIST = [
    "2024.04.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyle_AssocFullsymm",
    "2024.04.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyle_AssocFullsymmCommu",
    "2024.04.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyle_Fullsymm",
    "2024.04.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyle_Nothing",
]
EXP_NUM_LIST = [str(i) for i in range(1, 21)]
CHECK_POINT = 'checkpoint_10000.pt'
SAVED_IMG_NUM = 10


def load_dataset():
    dataset_loaders = []
    dataset_names = []
    for item in tqdm(EVAL_SETS, desc="Loading datasets"):
        dataset_names.append(item['name'])
        dataset = Dataset(item['path'])
        loader = DataLoader(dataset, batch_size=256, shuffle=True)
        dataset_loaders.append(loader)
    return dataset_names, dataset_loaders


def save_imgs(model: VQVAE, dataloader: DataLoader, img_num=SAVED_IMG_NUM, save_path: str = '', save_name: str = ''):
    for batch_ndx, sample in enumerate(dataloader):
        data, labels = sample
        assert img_num <= data[0].size(0), f"img_num should be smaller than batch size {data[0].size(0)}"
        recon_a, recon_b, recon_c, recon_ab, e_a, e_b, e_c, e_ab = model.forward(data)
        # idx_a = model.find_indices(e_a, True, False)[..., 0]
        # idx_b = model.find_indices(e_b, True, False)[..., 0]
        # idx_c = model.find_indices(e_c, True, False)[..., 0]
        # idx_ab = model.find_indices(e_ab, True, False)[..., 0]
        for i in range(img_num):
            vis_imgs = VisImgs()
            vis_imgs.gt_a, vis_imgs.gt_b, vis_imgs.gt_c = data[0][i], data[1][i], data[2][i]
            vis_imgs.recon_a, vis_imgs.recon_b, vis_imgs.recon_c, vis_imgs.plus_c = \
                recon_a[i], recon_b[i], recon_c[i], recon_ab[i]
            vis_imgs.save_img(os.path.join(save_path, f'{save_name}_{batch_ndx}_{i}.png'))
        break


if __name__ == "__main__":
    evalset_names, evalset_loaders = load_dataset()
    for exp_name in EXP_NAME_LIST:
        exp_path = os.path.join(EXP_ROOT, exp_name)
        config = load_config_from_exp_name(exp_name)
        plus_evaler = VQvaePlusEval(config)
        accu_results = [[] for i in range(0, len(evalset_names))]
        img_results_folders = []
        for i in range(0, len(evalset_names)):
            img_results_folder = os.path.join(exp_path, evalset_names[i])
            os.makedirs(img_results_folder, exist_ok=True)
            img_results_folders.append(img_results_folder)
        for sub_exp in EXP_NUM_LIST:
            # checkpoint_path = os.path.join(exp_path, sub_exp, CHECK_POINT)
            sub_exp_path = os.path.join(exp_path, sub_exp)
            optimal_checkpoint_num = find_optimal_checkpoint_num_by_train_config(sub_exp_path, config)
            checkpoint_path = os.path.join(exp_path, sub_exp, f'checkpoint_{optimal_checkpoint_num}.pt')
            plus_evaler.reload_model(checkpoint_path)
            for i in range(0, len(evalset_names)):
                # eval_plus_accu
                all_enc_z, all_plus_z = plus_evaler.load_plusZ_eval_data(evalset_loaders[i])
                ks, accu = calc_ks_enc_plus_z(all_enc_z, all_plus_z)
                accu_results[i].append(accu)
                img_path = os.path.join(img_results_folders[i], f'exp_{sub_exp}_accu_{accu}')
                plot_plusZ_against_label(all_enc_z, all_plus_z, img_path, is_scatter_lines=True, y_label="Emb idx")

                # save_plus_imgs
                plus_img_save_path = os.path.join(img_results_folders[i], f'plus_img_exp_{sub_exp}')
                os.makedirs(plus_img_save_path, exist_ok=True)
                save_imgs(plus_evaler.model, evalset_loaders[i], img_num=10, save_path=plus_img_save_path,
                          save_name=f'plus_img')


        for i in range(0, len(evalset_names)):
            record_path = os.path.join(exp_path, f'{evalset_names[i]}.txt')
            record_num_list(record_path, accu_results[i], EXP_NUM_LIST)
