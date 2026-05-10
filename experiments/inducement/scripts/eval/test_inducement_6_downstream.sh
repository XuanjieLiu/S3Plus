for i in {0..4}; do
  python run_evaluation.py \
    --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_symm0.3k4_k24_normed_${i}_minordown_val6/cp_step5901.pt" \
    --data_dir "../data/sax_minordown_val_ood_spectrum/sax_minordown_val6" \
    --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_minordown_val_0906.csv" \
    --future_pred_acc \
    --recon_acc
done
for i in {0..4}; do
  python run_evaluation.py \
    --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_symm0.3k4_k24_normed_${i}_minordown_val6/cp_step5901.pt" \
    --data_dir "../data/sax_minordown_all" \
    --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_minordown_all_0906.csv" \
    --future_pred_acc \
    --recon_acc
done

# python run_evaluation.py \
#   --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_symm0.3k1_k24_normed_0_minordown_val6/cp_step5901.pt" \
#   --data_dir "../data/sax_minordown_val_ood_spectrum/sax_minordown_val6" \
#   --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_minordown_val_0906.csv" \
#   --future_pred_acc \
#   --recon_acc
# python run_evaluation.py \
#   --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_symm0.3k1_k24_normed_1_minordown_val6/cp_step5901.pt" \
#   --data_dir "../data/sax_minordown_val_ood_spectrum/sax_minordown_val6" \
#   --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_minordown_val_0906.csv" \
#   --future_pred_acc \
#   --recon_acc
# python run_evaluation.py \
#   --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_symm0.3k1_k24_normed_2_minordown_val6/cp_step5901.pt" \
#   --data_dir "../data/sax_minordown_val_ood_spectrum/sax_minordown_val6" \
#   --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_minordown_val_0906.csv" \
#   --future_pred_acc \
#   --recon_acc
# python run_evaluation.py \
#   --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_symm0.3k1_k24_normed_3_minordown_val6/cp_step5901.pt" \
#   --data_dir "../data/sax_minordown_val_ood_spectrum/sax_minordown_val6" \
#   --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_minordown_val_0906.csv" \
#   --future_pred_acc \
#   --recon_acc
# python run_evaluation.py \
#   --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_symm0.3k1_k24_normed_4_minordown_val6/cp_step5901.pt" \
#   --data_dir "../data/sax_minordown_val_ood_spectrum/sax_minordown_val6" \
#   --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_minordown_val_0906.csv" \
#   --future_pred_acc \
#   --recon_acc

# python run_evaluation.py \
#   --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_symm0.3k1_k24_normed_0_minordown_val6/cp_step5901.pt" \
#   --data_dir "../data/sax_minordown_all" \
#   --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_minordown_all_0906.csv" \
#   --future_pred_acc \
#   --recon_acc
# python run_evaluation.py \
#   --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_symm0.3k1_k24_normed_1_minordown_val6/cp_step5901.pt" \
#   --data_dir "../data/sax_minordown_all" \
#   --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_minordown_all_0906.csv" \
#   --future_pred_acc \
#   --recon_acc
# python run_evaluation.py \
#   --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_symm0.3k1_k24_normed_2_minordown_val6/cp_step5901.pt" \
#   --data_dir "../data/sax_minordown_all" \
#   --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_minordown_all_0906.csv" \
#   --future_pred_acc \
#   --recon_acc
# python run_evaluation.py \
#   --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_symm0.3k1_k24_normed_3_minordown_val6/cp_step5901.pt" \
#   --data_dir "../data/sax_minordown_all" \
#   --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_minordown_all_0906.csv" \
#   --future_pred_acc \
#   --recon_acc
# python run_evaluation.py \
#   --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_symm0.3k1_k24_normed_4_minordown_val6/cp_step5901.pt" \
#   --data_dir "../data/sax_minordown_all" \
#   --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_minordown_all_0906.csv" \
#   --future_pred_acc \
#   --recon_acc

# -------------------------------------------

# python run_evaluation.py \
#   --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_nosymm_k24_normed_0_minordown_val6/cp_step5901.pt" \
#   --data_dir "../data/sax_minordown_val_ood_spectrum/sax_minordown_val6" \
#   --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_minordown_val_0906.csv" \
#   --future_pred_acc \
#   --recon_acc
# python run_evaluation.py \
#   --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_nosymm_k24_normed_1_minordown_val6/cp_step5901.pt" \
#   --data_dir "../data/sax_minordown_val_ood_spectrum/sax_minordown_val6" \
#   --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_minordown_val_0906.csv" \
#   --future_pred_acc \
#   --recon_acc
# python run_evaluation.py \
#   --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_nosymm_k24_normed_2_minordown_val6/cp_step5901.pt" \
#   --data_dir "../data/sax_minordown_val_ood_spectrum/sax_minordown_val6" \
#   --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_minordown_val_0906.csv" \
#   --future_pred_acc \
#   --recon_acc
# python run_evaluation.py \
#   --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_nosymm_k24_normed_3_minordown_val6/cp_step5901.pt" \
#   --data_dir "../data/sax_minordown_val_ood_spectrum/sax_minordown_val6" \
#   --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_minordown_val_0906.csv" \
#   --future_pred_acc \
#   --recon_acc
# python run_evaluation.py \
#   --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_nosymm_k24_normed_4_minordown_val6/cp_step5901.pt" \
#   --data_dir "../data/sax_minordown_val_ood_spectrum/sax_minordown_val6" \
#   --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_minordown_val_0906.csv" \
#   --future_pred_acc \
#   --recon_acc

# python run_evaluation.py \
#   --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_nosymm_k24_normed_0_minordown_val6/cp_step5901.pt" \
#   --data_dir "../data/sax_minordown_all" \
#   --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_minordown_all_0906.csv" \
#   --future_pred_acc \
#   --recon_acc
# python run_evaluation.py \
#   --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_nosymm_k24_normed_1_minordown_val6/cp_step5901.pt" \
#   --data_dir "../data/sax_minordown_all" \
#   --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_minordown_all_0906.csv" \
#   --future_pred_acc \
#   --recon_acc
# python run_evaluation.py \
#   --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_nosymm_k24_normed_2_minordown_val6/cp_step5901.pt" \
#   --data_dir "../data/sax_minordown_all" \
#   --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_minordown_all_0906.csv" \
#   --future_pred_acc \
#   --recon_acc
# python run_evaluation.py \
#   --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_nosymm_k24_normed_3_minordown_val6/cp_step5901.pt" \
#   --data_dir "../data/sax_minordown_all" \
#   --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_minordown_all_0906.csv" \
#   --future_pred_acc \
#   --recon_acc
# python run_evaluation.py \
#   --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_nosymm_k24_normed_4_minordown_val6/cp_step5901.pt" \
#   --data_dir "../data/sax_minordown_all" \
#   --save_results_at "experiments/inducement/results/csv/major_sax_val6_induced_minordown_all_0906.csv" \
#   --future_pred_acc \
#   --recon_acc