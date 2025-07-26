import os

data_root = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../../../dataset')
TRAIN_SET = f"{data_root}/multi_style_pairs(0,20)_mahjong/train"
EVAL_SET = f"{data_root}/multi_style_pairs(0,20)_mahjong/test"
SINGLE_IMG_SET = f"{data_root}/(0,20)-FixedPos-mahjong"
AUGMENT_TIMES = 16
IS_BLUR = True
BLUR_CONFIG = {  # 模糊处理配置, 如果is_blur为True, 则使用此配置
    'kernel_size_choices': (5, 7, 9),
    'sigma_range': (0.5, 3.0),
    'p_no_blur': 0.00,
}
CONFIG = {
    'train_data_path': TRAIN_SET,
    'single_img_eval_set_path': SINGLE_IMG_SET,
    'plus_eval_set_path': EVAL_SET,
    'num_sub_exp': 10,  # 子实验数量
    'num_workers': 8,  # 数据加载的线程数
    'is_random_split_data': False, # 是否随机划分数据集. 如果为True, 则eval_data_path, plus_eval_set_path_2会被忽略. 数据从train_data_path中随机划分
    'train_data_ratio': 0.8,  # 如果is_random_split_data为True, 则表示从train_data_path中随机划分出多少比例的数据作为训练集
    'latent_embedding_1': 2,
    'latent_embedding_2': 0,
    'multi_num_embeddings': None,
    'latent_code_2': 2,
    'embedding_dim': 1,
    'is_plot_zc_value': False,
    'embeddings_num': 10,
    'is_plot_vis_num': False,
    'kld_loss_scalar': 0.0,
    'checkpoint_interval': 200,
    'learning_rate': 1e-4,
    'scheduler_base_num': 0.99999,
    'max_iter_num': 2001,
    'model_path': 'curr_model.pt',
    'train_result_path': 'TrainingResults/',
    'eval_result_path': 'EvalResults/',
    'train_record_path': "Train_record.txt",
    'eval_record_path': "Eval_record.txt",
    'plus_accu_record_path': "plus_eval.txt",
    'log_interval': 50,
    'is_save_img': True,
    'batch_size': 256,
    'is_commutative_train': False,
    'is_commutative_all': False,
    'z_plus_loss_scalar': 0.1,
    'commutative_z_loss_scalar': 0.0,
    'associative_z_loss_scalar': 0.1,
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
    'is_assoc_within_batch': False,
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
    'is_blur': IS_BLUR,  # 是否在评估时模糊处理图像
    'blur_config': BLUR_CONFIG,
    'augment_times': AUGMENT_TIMES,
    'eval_config': {
        'pipeline_result_path': 'PIPELINE_EVAL',
        'optimal_checkpoint_finding_config': {
            'optimal_checkpoint_num': 'find_by_keys',
            'record_name': 'Train_record.txt',
            'keys': ['plus_z', 'plus_recon', 'loss_oper', 'loss_ED'],
            'iter_after': 0.5,
        },
        'plus_eval_configs': [
            {
                'name': 'eval_set',
                'eval_set_path_list': [
                    EVAL_SET,
                ],
                'is_blur': True,  # 是否在评估时模糊处理图像
                'blur_config': {  # 模糊处理配置, 如果is_blur为True, 则使用此配置
                    'kernel_size_choices': (5, 7, 9),
                    'sigma_range': (0.5, 3.0),
                    'p_no_blur': 0.00,
                },
                'augment_times': 16,
            },
        ],
        'emb_matching_rate_configs': [
            {
                'name': 'emb_matching_rate',
                'eval_set_path_list': [
                    SINGLE_IMG_SET,
                ],
                'is_blur': True,  # 是否在评估时模糊处理图像
                'blur_config': {  # 模糊处理配置, 如果is_blur为True, 则使用此配置
                    'kernel_size_choices': (5, 7, 9),
                    'sigma_range': (0.5, 3.0),
                    'p_no_blur': 0.00,
                },
                'augment_times': 16,
            },
        ],
        'orderliness_configs': [
            {
                'name': 'orderliness',
                'img_dir_name': 'orderliness',
                'eval_set_path_list': [
                    SINGLE_IMG_SET,
                ],
                'is_blur': True,  # 是否在评估时模糊处理图像
                'blur_config': {  # 模糊处理配置, 如果is_blur为True, 则使用此配置
                    'kernel_size_choices': (5, 7, 9),
                    'sigma_range': (0.5, 3.0),
                    'p_no_blur': 0.00,
                },
                'augment_times': 16,
            },
        ],
    },
}
