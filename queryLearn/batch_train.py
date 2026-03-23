import sys
import os
from importlib import reload

sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from queryLearn.QueryLearn import QueryLearn


if len(sys.argv) < 2:
    print("Usage: python batch_train.py EXP_NAME_1 EXP_NAME_2 ...")
    sys.exit()

EXP_ROOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exps')
sys.path.append(EXP_ROOT_PATH)
EXP_NAME_LIST = sys.argv[1:]


if __name__ == '__main__':
    print(f'Experiment names: {EXP_NAME_LIST}')
    for exp_name in EXP_NAME_LIST:
        exp_path = os.path.join(EXP_ROOT_PATH, exp_name)
        os.chdir(exp_path)
        sys.path.append(exp_path)
        print(f'Exp path: {exp_path}')
        t_config = __import__('config')
        reload(t_config)
        sys.path.pop()
        num_sub_exp = t_config.CONFIG.get('num_sub_exp', 1)
        exp_num_list = [str(i) for i in range(1, num_sub_exp + 1)]
        for exp_num in exp_num_list:
            os.makedirs(exp_num, exist_ok=True)
            sub_exp_path = os.path.join(exp_path, exp_num)
            print(f'Sub-Exp path: {sub_exp_path}')
            os.chdir(sub_exp_path)
            t_config.CONFIG['sub_exp_id'] = exp_num
            t_config.CONFIG['_exp_dir'] = exp_path
            t_config.CONFIG['train_result_path'] = t_config.CONFIG.get('train_result_path', 'TrainingResults')
            t_config.CONFIG['eval_result_path'] = t_config.CONFIG.get('eval_result_path', 'EvalResults')
            t_config.CONFIG['train_record_path'] = os.path.join(t_config.CONFIG['train_result_path'], 'Train_record.txt')
            t_config.CONFIG['eval_record_path'] = os.path.join(t_config.CONFIG['eval_result_path'], 'Eval_record.txt')
            t_config.CONFIG['model_path'] = t_config.CONFIG.get('model_path', 'curr_model.pt')
            trainer = QueryLearn(t_config.CONFIG)
            trainer.train()
            os.chdir(exp_path)
