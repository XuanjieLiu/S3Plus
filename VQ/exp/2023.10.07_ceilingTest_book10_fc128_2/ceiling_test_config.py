import os

data_root = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../../../dataset')
CONFIG = [
    {
        'task_name': "book10_minus_16_2_linear",
        'train_data_path': f"{data_root}/single_style_pairs_minus(0,20)/train",
        'eval_data_path': f"{data_root}/single_style_pairs_minus(0,20)/test",
        'batch_size': 128,
        'train_record_path': "train_record.txt",
        'eval_record_path': "eval_record.txt",
        'fc_model_path': 'model.pt',
        'max_iter_num': 20001,
        'log_interval': 200,
        'book_type': 'linear',
        'num_dim': 2,
        'dim_size': 10,
        'num_class': 21,
        'fc_network_config': {
            'plus_unit': 128,
            'n_hidden_layers': 2,
        },
        'learning_rate': 1e-3,
    }, {
        'task_name': "book10_minus_16_2_random",
        'train_data_path': f"{data_root}/single_style_pairs_minus(0,20)/train",
        'eval_data_path': f"{data_root}/single_style_pairs_minus(0,20)/test",
        'batch_size': 128,
        'train_record_path': "train_record.txt",
        'eval_record_path': "eval_record.txt",
        'fc_model_path': 'model.pt',
        'max_iter_num': 20001,
        'log_interval': 200,
        'book_type': 'random',
        'num_dim': 2,
        'dim_size': 10,
        'num_class': 21,
        'fc_network_config': {
            'plus_unit': 128,
            'n_hidden_layers': 2,
        },
        'learning_rate': 1e-3,
    }

]
