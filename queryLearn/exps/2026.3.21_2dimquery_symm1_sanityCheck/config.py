import os

data_root = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../../../dataset')
EVAL_SET = f"{data_root}/single_style_pairs_add(0,20)/test"
TRAIN_SET = [
    f"{data_root}/single_style_pairs_add(0,20)/train",
    f"{data_root}/single_style_pairs_mul_mod21(0,20)/train",
    f"{data_root}/single_style_pairs_add(0,20)/test",
    f"{data_root}/single_style_pairs_mul_mod21(0,20)/test",
]
SINGLE_IMG_SET = f"{data_root}/(0,20)-FixedPos-mul_add_mix"
IS_BLUR = False
BLUR_CONFIG = {  # 模糊处理配置, 如果is_blur为True, 则使用此配置
    'kernel_size_choices': (5, 7, 9),
    'sigma_range': (0.5, 3.0),
    'p_no_blur': 0.00,
}
AUGMENT_TIMES = 16
CONFIG = {
    'VQSPS': {
        'EXP_NAME': '2025.05.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_Fullsymm',
        'CHECK_POINT_NAME': '1/checkpoint_48000.pt',
    },
    'query_learner': {
        'in_dim': 2,  # 示例输入维度
        'out_dim': 4,  # 示例输出维度
    },
    'operator': {
        'unit': 1024,
        'n_hidden_layers': 3,
    },
    'train_data_path': TRAIN_SET,
    'single_img_eval_set_path': SINGLE_IMG_SET,
    'plus_eval_set_path': EVAL_SET,
    'is_random_split_data': True,
    'train_data_ratio': 0.8,
    'checkpoint_interval': 2000,
    'checkpoint_after': 28000,
    'learning_rate': 1e-4,
    'scheduler_base_num': 0.99999,
    'max_iter_num': 32001,
    'model_path': 'curr_model.pt',
    'train_result_path': 'TrainingResults/',
    'eval_result_path': 'EvalResults/',
    'train_record_path': "Train_record.txt",
    'eval_record_path': "Eval_record.txt",
    'log_interval': 500,
    'eval_interval': 2000,
    'batch_size': 256,
    'train_sps': False,
    'is_symm': True,
    'is_assoc': False,
    'symm_loss_scalar': 1.,
    'eqLoss_scalar': 0.05,
    'sanity_check': True,
}
