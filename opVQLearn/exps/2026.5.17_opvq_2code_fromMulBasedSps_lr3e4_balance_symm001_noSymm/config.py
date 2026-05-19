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

CONFIG = {
    'num_sub_exp': 1,
    'VQSPS': {
        'EXP_NAME': '2026.04.16_10vq_Zc[2]_Zs[0]_edim4_[0-20]_plus1024_1_mulmod20_Fullsymm_train0.6',
        'CHECK_POINT_NAME': '1/checkpoint_44000.pt',
    },
    'train_sps': False,
    'op_code': {
        'dim': 2,
        'num_codes': 2,
        'init_codes': [
            [1.0, -1.0],
            [-1.0, 1.0],
        ],
        'commitment_cost': 0.25,
        'embedding_cost': 1.0,
        'loss_scalar': 1.0,
        'softmax_tau': 0.5,
    },
    'op_encoder': {
        'unit': 1024,
        'n_hidden_layers': 3,
    },
    'decoder': {
        'unit': 2048,
        'n_hidden_layers': 5,
    },
    'balance': {
        'use_balance_loss': True,
        'loss_scalar': 0.01,
    },
    'symm': {
        'use_symm_loss': False,
        'loss_scalar': 0.01,
        'include_sps_vq_loss': False,
    },
    'sps_vq_loss_scalar': 0.05,
    'op_vis_format': 'png',
    'train_data_path': TRAIN_SET,
    'single_img_eval_set_path': SINGLE_IMG_SET,
    'plus_eval_set_path': EVAL_SET,
    'is_random_split_data': True,
    'random_split_seed': 20260510,
    'train_data_ratio': 0.7,
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
}
