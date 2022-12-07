import math

CONFIG = {
    'train_data_path': "../dataset/PlusPair-(1,8)-FixedPos/train",
    'latent_code_1': 1,
    'latent_code_2': 1,
    'kld_loss_scalar': 0.01,
    'checkpoint_interval': 1000,
    'learning_rate': 1e-4,
    'scheduler_base_num': 0.99999,
    'max_iter_num': 100001,
    'model_path': 'curr_model.pt',
    'train_result_path': 'TrainingResults/',
    'train_record_path': "Train_record.txt",
    'eval_record_path': "Eval_record.txt",
    'log_interval': 50,
    'eval_interval': 1000,
    'is_save_img': True,
    'batch_size': 32,
    'z_std_loss_scalar': 10,
    'z_minus_loss_scalar': 10,
    'z_plus_loss_scalar': 10,
    'commutative_z_loss_scalar': 10,
    'associative_z_loss_scalar': 10,

}
