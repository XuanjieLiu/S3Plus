# i_list=(0 1 2 3 4 5 6 7)
# step_list=(5901 4901 5101 5901 5901 5501 5901 5701)
# # i_list=(4 5 6 7)
# # step_list=(5901 5501 5901 5701)

# for idx in ${!i_list[@]}; do
#   i=${i_list[$idx]}
#   step=${step_list[$idx]}
  
#   python run_evaluation.py \
#     --probing 1 \
#     --active_checkpoint "experiments/transition/logs/major_sax_transition_val3_symm0.2_k24_normed_cos_gru_gr7_${i}/cp_step${step}.pt" \
#     --data_dir "../data/sax_major_all" \
#     --save_results_at "experiments/transition/results/archive/csv/major_sax_val3_transition_probe_1102.csv"
# done

i_list=(0 1 2 3 4 5 6 7 8 9)
step_list=(3901 3901 3901 3901 3901 3901 3901 3901 3901 3901)

for idx in ${!i_list[@]}; do
  i=${i_list[$idx]}
  step=${step_list[$idx]}

  python run_evaluation.py \
    --probing 1 \
    --active_checkpoint "experiments/transition/logs/major_sax_transition_val3_nosymm_k24_normed_cos_gru_gr7_$i/cp_step${step}.pt" \
    --data_dir "../data/sax_major_all" \
    --save_results_at "experiments/transition/results/archive/csv/major_sax_val3_transition_probe_1102.csv"
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