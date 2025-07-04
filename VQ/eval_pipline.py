import sys
import os
import numpy as np
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from importlib import reload
from interpolate_plus_accu_eval import InterpolatePlusAccuEval
from common_func import load_config_from_exp_name, record_num_list, EXP_ROOT, find_optimal_checkpoint_num_by_train_config
from eval_plus_nd import VQvaePlusEval, calc_ks_enc_plus_z, plot_plusZ_against_label, calc_multi_emb_plus_accu
from dataloader_plus import Dataset
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader

EXP_NUM_LIST = [str(i) for i in range(1, 21)]
EXP_NAME_LIST = [
    "2025.05.15_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyle_Fullsymm",
    "2025.05.15_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyle_Nothing",
    "2025.05.19_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyle_Nothing_trainAll",
    "2025.06.10_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyle_PureVQ",
    "2025.06.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyleMahjong_nothing",
    "2025.06.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyleMahjong_PureVQ",
    "2025.06.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyleMahjong_symm",
    "2025.06.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyleMahjong_trainAll",
    "2025.06.18_100vq_Zc[1]_Zs[0]_edim2_[0-20]_plus1024_1_tripleSet_Fullsymm_OnlineBlur",
    "2025.06.18_100vq_Zc[1]_Zs[0]_edim2_[0-20]_plus1024_1_tripleSet_Nothing_trainAll_OnlineBlur",
    "2025.06.16_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_Nothing_OnlineBlur",
    "2025.06.16_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_PureVQ_OnlineBlur",
    "2025.06.13_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_Fullsymm_OnlineBlur",
    "2025.06.13_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_Nothing_trainAll_OnlineBlur",
]

if __name__ == "__main__":
    for exp_name in EXP_NAME_LIST:
        exp_path = os.path.join(EXP_ROOT, exp_name)
        config = load_config_from_exp_name(exp_name)
        eval_config = config['eval_config']
        pipline_result_path = os.path.join(exp_path, eval_config['pipline_result_path'])
        os.makedirs(pipline_result_path, exist_ok=True)
        optimal_checkpoint_finding_config = eval_config['optimal_checkpoint_finding_config']
        # eval plus accu
        if eval_config.get('plus_eval_config') is not None:
            evaler = VQvaePlusEval(config)
            plus_eval_config = eval_config['plus_eval_config']
            plus_eval_set_path_list = plus_eval_config['eval_set_path_list']
            # 如果有多个 set_path, 创建多个 dataset 并合并
            datasets = ConcatDataset([Dataset(path) for path in plus_eval_set_path_list])
            data_loader = DataLoader(datasets, batch_size=config['batch_size'], shuffle=False)
            one2n_accu_result_path = os.path.join(pipline_result_path, f"{plus_eval_config['one2n_accu_result_name']}.txt")
            one2one_accu_result_path = os.path.join(pipline_result_path, f"{plus_eval_config['one2one_accu_result_name']}.txt")
            one2n_accu_list = []
            one2one_accu_list = []
            for sub_exp in EXP_NUM_LIST:
                sub_exp_path = os.path.join(exp_path, sub_exp)
                optimal_checkpoint_num = find_optimal_checkpoint_num_by_train_config(sub_exp_path, config, optimal_checkpoint_finding_config)
                print(f'Optimal checkpoint number for {sub_exp}: {optimal_checkpoint_num}')
                checkpoint_path = os.path.join(exp_path, sub_exp, f'checkpoint_{optimal_checkpoint_num}.pt')
                evaler.reload_model(checkpoint_path)
                all_enc_z, all_plus_z = evaler.load_plusZ_eval_data(data_loader, is_find_index=True)
                one2n_accu_list.append(calc_multi_emb_plus_accu(all_enc_z, all_plus_z))
                one2one_accu_list.append(calc_ks_enc_plus_z(all_enc_z, all_plus_z)[1])
            record_num_list(os.path.join(pipline_result_path, one2n_accu_result_path), one2n_accu_list, EXP_NUM_LIST)
            record_num_list(os.path.join(pipline_result_path, one2one_accu_result_path), one2one_accu_list, EXP_NUM_LIST)
