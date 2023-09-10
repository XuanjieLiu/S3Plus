import os

data_root = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../../../dataset')
CONFIG = [
    {
        'task_name': "minus",
        'train_data_path': f"{data_root}/single_style_pairs_minus(0,20)/train",
        'eval_data_path': f"{data_root}/single_style_pairs_minus(0,20)/test",
        'batch_size': 128,
        'train_record_path': "minus_train_record.txt",
        'eval_record_path': "minus_eval_record.txt",
        'pretrained_path': 'curr_model.pt',
        'fc_model_path': 'minus_model.pt',
        'max_iter_num': 10000,
        'log_interval': 200,
        'num_class': 21,
        'fc_network_config': {
            'plus_unit': 128,
            'n_hidden_layers': 1,
        },
        'learning_rate': 1e-3,
    },

]
