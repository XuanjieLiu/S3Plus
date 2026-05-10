# i_list=(0 1 3 4)
# step_list=(3001 4801 4801 5601)

# for idx in ${!i_list[@]}; do
#   i=${i_list[$idx]}
#   step=${step_list[$idx]}
  
#   python run_evaluation.py \
#     --active_checkpoint "experiments/transition/logs/major_sax_transition_val6_symm0.2frombl_k24_normed_cos_gru_gr7_${i}/cp_step${step}.pt" \
#     --data_dir "../data/sax_major_val_ood_spectrum_2/sax_major_val6" \
#     --save_results_at "experiments/transition/results/final_1119/source_csv/major_sax_val6_transition_val_1031.csv" \
#     --future_pred_acc \
#     --recon_acc

#   python run_evaluation.py \
#     --active_checkpoint "experiments/transition/logs/major_sax_transition_val6_symm0.2frombl_k24_normed_cos_gru_gr7_${i}/cp_step${step}.pt" \
#     --data_dir "../data/sax_major_val_ood_spectrum_2/sax_major_ood6" \
#     --save_results_at "experiments/transition/results/final_1119/source_csv/major_sax_val6_transition_ood_1031.csv" \
#     --future_pred_acc \
#     --recon_acc
# done

i_list=(0 1 2 3 4)
step_list=(2501 3101 2701 3601 3501)

for idx in ${!i_list[@]}; do
  i=${i_list[$idx]}
  step=${step_list[$idx]}

  python run_evaluation.py \
    --active_checkpoint "experiments/transition/logs/major_sax_transition_val6_nosymm_k24_normed_cos_gru_$i/cp_step${step}.pt" \
    --data_dir "../data/sax_major_val_ood_spectrum_2/sax_major_val6" \
    --save_results_at "experiments/transition/results/final_1119/source_csv/major_sax_val6_transition_val_1031.csv" \
    --future_pred_acc \
    --recon_acc
  python run_evaluation.py \
    --active_checkpoint "experiments/transition/logs/major_sax_transition_val6_nosymm_k24_normed_cos_gru_$i/cp_step${step}.pt" \
    --data_dir "../data/sax_major_val_ood_spectrum_2/sax_major_ood6" \
    --save_results_at "experiments/transition/results/final_1119/source_csv/major_sax_val6_transition_ood_1031.csv" \
    --future_pred_acc \
    --recon_acc
done

# python run_evaluation.py --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_symm0.3k4_k24_normed_0/cp_step5901.pt" --data_dir "../data/sax_major_val_ood_spectrum/sax_major_val6" --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_val_0823.csv" --future_pred_acc --recon_acc
# python run_evaluation.py --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_symm0.3k4_k24_normed_1/cp_step5901.pt" --data_dir "../data/sax_major_val_ood_spectrum/sax_major_val6" --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_val_0823.csv" --future_pred_acc --recon_acc
# python run_evaluation.py --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_symm0.3k4_k24_normed_2/cp_step5901.pt" --data_dir "../data/sax_major_val_ood_spectrum/sax_major_val6" --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_val_0823.csv" --future_pred_acc --recon_acc
# python run_evaluation.py --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_symm0.3k4_k24_normed_3/cp_step5901.pt" --data_dir "../data/sax_major_val_ood_spectrum/sax_major_val6" --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_val_0823.csv" --future_pred_acc --recon_acc
# python run_evaluation.py --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_symm0.3k4_k24_normed_4/cp_step5901.pt" --data_dir "../data/sax_major_val_ood_spectrum/sax_major_val6" --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_val_0823.csv" --future_pred_acc --recon_acc

# python run_evaluation.py --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_symm0.3k4_k24_normed_0/cp_step5901.pt" --data_dir "../data/sax_major_all" --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_all_0823.csv" --future_pred_acc --recon_acc
# python run_evaluation.py --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_symm0.3k4_k24_normed_1/cp_step5901.pt" --data_dir "../data/sax_major_all" --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_all_0823.csv" --future_pred_acc --recon_acc
# python run_evaluation.py --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_symm0.3k4_k24_normed_2/cp_step5901.pt" --data_dir "../data/sax_major_all" --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_all_0823.csv" --future_pred_acc --recon_acc
# python run_evaluation.py --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_symm0.3k4_k24_normed_3/cp_step5901.pt" --data_dir "../data/sax_major_all" --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_all_0823.csv" --future_pred_acc --recon_acc
# python run_evaluation.py --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_symm0.3k4_k24_normed_4/cp_step5901.pt" --data_dir "../data/sax_major_all" --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_all_0823.csv" --future_pred_acc --recon_acc




# python run_evaluation.py --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_nosymm_k24_normed_5/cp_step5901.pt" --data_dir "../data/sax_major_val_ood_spectrum/sax_major_val6" --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_val_0825.csv" --future_pred_acc --recon_acc
# python run_evaluation.py --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_nosymm_k24_normed_6/cp_step5901.pt" --data_dir "../data/sax_major_val_ood_spectrum/sax_major_val6" --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_val_0825.csv" --future_pred_acc --recon_acc
# python run_evaluation.py --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_nosymm_k24_normed_7/cp_step5901.pt" --data_dir "../data/sax_major_val_ood_spectrum/sax_major_val6" --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_val_0825.csv" --future_pred_acc --recon_acc

# python run_evaluation.py --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_nosymm_k24_normed_5/cp_step5901.pt" --data_dir "../data/sax_major_all" --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_all_0825.csv" --future_pred_acc --recon_acc
# python run_evaluation.py --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_nosymm_k24_normed_6/cp_step5901.pt" --data_dir "../data/sax_major_all" --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_all_0825.csv" --future_pred_acc --recon_acc
# python run_evaluation.py --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_nosymm_k24_normed_7/cp_step5901.pt" --data_dir "../data/sax_major_all" --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_all_0825.csv" --future_pred_acc --recon_acc