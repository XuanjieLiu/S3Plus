python run_evaluation.py --active_checkpoint "logs/major_nosymm_k24_5/cp_step9901.pt" --data_dir "../data/insnotes_major_val" --future_pred_acc
python run_evaluation.py --active_checkpoint "logs/major_nosymm_k24_6/cp_step9901.pt" --data_dir "../data/insnotes_major_val" --future_pred_acc
python run_evaluation.py --active_checkpoint "logs/major_nosymm_k24_7/cp_step9901.pt" --data_dir "../data/insnotes_major_val" --future_pred_acc
python run_evaluation.py --active_checkpoint "logs/major_nosymm_k24_8/cp_step9901.pt" --data_dir "../data/insnotes_major_val" --future_pred_acc
python run_evaluation.py --active_checkpoint "logs/major_nosymm_k24_9/cp_step9901.pt" --data_dir "../data/insnotes_major_val" --future_pred_acc

python run_evaluation.py --active_checkpoint "logs/major_nosymm_k24_5/cp_step9901.pt" --data_dir "../data/insnotes_major_ood" --future_pred_acc
python run_evaluation.py --active_checkpoint "logs/major_nosymm_k24_6/cp_step9901.pt" --data_dir "../data/insnotes_major_ood" --future_pred_acc
python run_evaluation.py --active_checkpoint "logs/major_nosymm_k24_7/cp_step9901.pt" --data_dir "../data/insnotes_major_ood" --future_pred_acc
python run_evaluation.py --active_checkpoint "logs/major_nosymm_k24_8/cp_step9901.pt" --data_dir "../data/insnotes_major_ood" --future_pred_acc
python run_evaluation.py --active_checkpoint "logs/major_nosymm_k24_9/cp_step9901.pt" --data_dir "../data/insnotes_major_ood" --future_pred_acc
