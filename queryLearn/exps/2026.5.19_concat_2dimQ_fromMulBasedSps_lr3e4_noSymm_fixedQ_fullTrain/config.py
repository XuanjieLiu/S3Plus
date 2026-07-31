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
BLUR_CONFIG = {
    'kernel_size_choices': (5, 7, 9),
    'sigma_range': (0.5, 3.0),
    'p_no_blur': 0.00,
}
AUGMENT_TIMES = 16
CONFIG = {
    'num_sub_exp': 3,
    'VQSPS': {
        'EXP_NAME': '2026.04.16_10vq_Zc[2]_Zs[0]_edim4_[0-20]_plus1024_1_mulmod20_Fullsymm_train0.6',
        'CHECK_POINT_NAME': '1/checkpoint_44000.pt',
    },
    'train_sps': False,
    'query_learner': {
        'query_dim': 2,
        'init_queries': [
            [1.0, -1.0],
            [-1.0, 1.0],
        ],
        'out_dim': 2,
        'train_queries': False,
    },
    'query_vis_format': 'png',
    'operator': {
        'condition_mode': 'concat',
        'film_init_identity': True,
        'unit': 2048,
        'n_hidden_layers': 5,
    },
    'train_data_path': TRAIN_SET,
    'single_img_eval_set_path': SINGLE_IMG_SET,
    'plus_eval_set_path': EVAL_SET,
    'is_random_split_data': True,
    'random_split_seed': 20260510,
    'train_data_ratio': 1.0,
    'checkpoint_interval': 2500,
    'checkpoint_after': 5000,
    'learning_rate': 3e-4,
    'optimizer': 'adamw',
    'weight_decay': 1e-4,
    'grad_clip_norm': 1.0,
    'scheduler_base_num': 0.99999,
    'max_iter_num': 60001,
    'model_path': 'curr_model.pt',
    'train_result_path': 'TrainingResults/',
    'eval_result_path': 'EvalResults/',
    'train_record_path': "Train_record.txt",
    'eval_record_path': "Eval_record.txt",
    'log_interval': 500,
    'eval_interval': 2500,
    'batch_size': 256,
    'is_symm': False,
    'is_assoc': False,
    'symm_loss_scalar': 0.05,
    'eqLoss_scalar': 0.05,
    'sanity_check': False,
}
