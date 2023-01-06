import math
import os

data_root = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../dataset')
CONFIG = {
    'train_data_path': f"{data_root}/PlusPair-(1,7)-FixedPos/train",
    'eval_path_1': f"{data_root}/(0,16)-FixedPos-4Color",
    'latent_code_1': 1,
    'latent_code_2': 2,
    'kld_loss_scalar': 0.0,
    'checkpoint_interval': 1000,
    'learning_rate': 1e-4,
    'scheduler_base_num': 0.99999,
    'max_iter_num': 6001,
    'model_path': 'curr_model.pt',
    'train_result_path': 'TrainingResults/',
    'train_record_path': "Train_record.txt",
    'eval_record_path': "Eval_record.txt",
    'log_interval': 100,
    'eval_interval': 1000,
    'is_save_img': True,
    'batch_size': 32,
    'z_std_loss_scalar': 0.5,
    'z_minus_loss_scalar': 0.5,
    'z_plus_loss_scalar': 0.5,
    'commutative_z_loss_scalar': 0.5,
    'associative_z_loss_scalar': 0,
    'min_loss_scalar': 1e-8,
    'K': 1024,
    'assoc_aug_range': (-3, 3),

}
