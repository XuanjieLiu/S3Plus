import os

data_root = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../../../dataset')
CONFIG = [
    {
        'task_name': "minus_16_1.1",
        'train_data_path': f"{data_root}/single_style_pairs_minus(0,20)_mahjong/train",
        'eval_data_path': f"{data_root}/single_style_pairs_minus(0,20)_mahjong/test",
        'batch_size': 128,
        'train_record_path': "train_record.txt",
        'eval_record_path': "eval_record.txt",
        'pretrained_path': 'auto',
        'fc_model_path': 'minus_model.pt',
        'max_iter_num': 6001,
        'log_interval': 200,
        'num_class': 21,
        'fc_network_config': {
            'plus_unit': 16,
            'n_hidden_layers': 1,
        },
        'learning_rate': 1e-3,
        'num_sub_exp': 10,  # 子实验数量
        'num_workers': 0,
        'optimal_checkpoint_finding_config': {
            'optimal_checkpoint_num': 'find_by_keys',
            'record_name': 'Train_record.txt',
            'keys': ['plus_recon', 'plus_z', 'loss_oper', 'loss_ED'],
            'iter_after': 0.0,
        },
        'is_blur': False,  # 是否在评估时模糊处理图像
        'blur_config': {  # 模糊处理配置, 如果is_blur为True, 则使用此配置
            'kernel_size_choices': (5, 7, 9),
            'sigma_range': (0.5, 3.0),
            'p_no_blur': 0.00,
        },
        'augment_times': 1,
    },
]
