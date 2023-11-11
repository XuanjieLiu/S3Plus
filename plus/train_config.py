import math
import os

data_root = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../dataset')
CONFIG = {
    'train_data_path': f"{data_root}/PlusPair-(1,7)-FixedPos-oneStyle/train",
    'single_img_eval_set_path': f"{data_root}/(0,20)-FixedPos-oneStyle",
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
    'log_interval': 50,
    'eval_interval': 1000,
    'is_save_img': True,
    'batch_size': 32,
    'plus_recon_loss_scalar': 1.5,
    'z_plus_loss_scalar': 0.01,
    'commutative_z_loss_scalar': 0.01,
    'associative_z_loss_scalar': 0.01,
    'min_loss_scalar': 1e-8,
    'K': 1024,
    'assoc_aug_range': (-3, 3),
    'network_config': {
        'enc_dec': {
            'img_channel': 3,
            'last_H': 4,
            'last_W': 4,
            'first_ch_num': 64,
        },
        'plus': {
            'plus_unit': 4,
        }
    },

}
