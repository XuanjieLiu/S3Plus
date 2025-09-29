import os

data_root = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../../../dataset')
EVAL_SET = f"{data_root}/multi_style_pairs(0,20)_mahjong/test"
TRAIN_SET = f"{data_root}/multi_style_pairs(0,20)_mahjong/train"
SINGLE_IMG_SET = f"{data_root}/(0,20)-FixedPos-mahjong"
IS_BLUR = False
AUGMENT_TIMES = 1
CONFIG = {
    'train_data_path': TRAIN_SET,
    'single_img_eval_set_path': SINGLE_IMG_SET,
    'plus_eval_set_path': EVAL_SET,
    'plus_eval_set_path_2': None,
    'is_random_split_data': False, # 是否随机划分数据集. 如果为True, 则eval_data_path, plus_eval_set_path_2会被忽略. 数据从train_data_path中随机划分
    'train_data_ratio': 0.8,  # 如果is_random_split_data为True, 则表示从train_data_path中随机划分出多少比例的数据作为训练集
    'is_online_blur': False,
    'latent_embedding_1': 4,
    'latent_embedding_2': 0,
    'multi_num_embeddings': None,
    'latent_code_2': 0,
    'embedding_dim': 1,
    'is_plot_zc_value': True,
    'embeddings_num': 10,
    'is_plot_vis_num': False,
    'kld_loss_scalar': 0.0,
    'checkpoint_interval': 2000,
    'learning_rate': 1e-4,
    'scheduler_base_num': 0.99999,
    'max_iter_num': 50001,
    'model_path': 'curr_model.pt',
    'train_result_path': 'TrainingResults/',
    'eval_result_path': 'EvalResults/',
    'train_record_path': "Train_record.txt",
    'eval_record_path': "Eval_record.txt",
    'plus_accu_record_path': "plus_eval.txt",
    'log_interval': 500,
    'eval_interval': 2000,
    'is_save_img': True,
    'batch_size': 128,
    'is_commutative_train': False,
    'is_commutative_all': False,
    'z_plus_loss_scalar': 0.02,
    'commutative_z_loss_scalar': 0.0,
    'associative_z_loss_scalar': 0.02,
    'plus_mse_scalar': -1,
    'plus_recon_loss_scalar': 3,
    'min_loss_scalar': 0.00001,
    'K': 1024,
    'assoc_aug_range': (-3, 3),
    'commitment_scalar': 0.0025,
    'embedding_scalar': 0.01,
    'isVQStyle': False,
    'plus_by_embedding': True,
    'plus_by_zcode': False,
    'VQPlus_eqLoss_scalar': 0.5,
    'is_zc_based_assoc': True,
    'is_rand_z_assoc': False,
    'is_assoc_on_e': True,
    'is_assoc_on_z': False,
    'is_assoc_within_batch': True,
    'is_switch_digital': False,
    'is_symm_assoc': True,
    'is_full_symm': True,
    'is_pure_assoc': False,
    'is_twice_oper': False,
    'network_config': {
        'enc_dec': {
            'img_channel': 3,
            'last_H': 4,
            'last_W': 4,
            'first_ch_num': 16,
        },
        'enc_fc': {
            'n_units': 128,
            'n_hidden_layers': 2,
        },
        'plus': {
            'plus_unit': 1024,
            'n_hidden_layers': 2,
        }
    },
    'eval_config': {
        'pipeline_result_path': 'PIPELINE_EVAL',
        'optimal_checkpoint_finding_config': {
            'optimal_checkpoint_num': 'find_by_keys',
            'record_name': 'Train_record.txt',
            'keys': ['plus_recon', 'plus_z', 'loss_oper', 'loss_ED'],
            'iter_after': 0.0,
        },
        'plus_eval_configs': [
            {
                'name': 'eval_set',
                'eval_set_path_list': [
                    EVAL_SET,
                ],
                'is_blur': IS_BLUR,  # 是否在评估时模糊处理图像
                'augment_times': AUGMENT_TIMES,
            },
            {
                'name': 'train_set',
                'eval_set_path_list': [
                    TRAIN_SET,
                ],
                'is_blur': IS_BLUR,  # 是否在评估时模糊处理图像
                'augment_times': AUGMENT_TIMES,
            },
        ],
        'emb_matching_rate_configs': [
            {
                'name': 'emb_matching_rate',
                'eval_set_path_list': [
                    SINGLE_IMG_SET,
                ],
                'is_blur': IS_BLUR,  # 是否在评估时模糊处理图像
                'augment_times': AUGMENT_TIMES,
            },
        ],
        'orderliness_configs': [
            {
                'name': 'orderliness',
                'img_dir_name': 'orderliness',
                'eval_set_path_list': [
                    SINGLE_IMG_SET,
                ],
                'is_blur': IS_BLUR,  # 是否在评估时模糊处理图像
                'augment_times': AUGMENT_TIMES,
            },
        ],
        'interpolate_configs': [
            {
                'name': 'train_interpolate',
                'eval_set_path_list': [
                    TRAIN_SET,
                ],
                'is_blur': IS_BLUR,  # 是否在评估时模糊处理图像
                'augment_times': AUGMENT_TIMES,
                'interpolate_num': 10,
            },
        ],
    },
}
