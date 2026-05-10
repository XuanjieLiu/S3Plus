# python -m pdb run_evaluation.py --probing 1 --active_checkpoint "experiments/inducement/logs/major_sax_induced_val6_symm0.3k4_k24_normed_0/cp_step5901.pt" --data_dir "../data/sax_major_all" --save_results_at "experiments/inducement/results/csv/major_sav_val6_induced_probe_0914.csv"

for i in {0..4}; do
    python run_evaluation.py \
        --probing 1 \
        --active_checkpoint "experiments/inducement/logs/major_sax_induced_val3_symm0.3k4_k24_normed_${i}/cp_step5901.pt" \
        --data_dir "../data/sax_major_all" \
        --save_results_at "experiments/inducement/results/csv/major_sav_val3_induced_probe_0914.csv"
done
for i in {0..4}; do
    python run_evaluation.py \
        --probing 1 \
        --active_checkpoint "experiments/inducement/logs/major_sax_induced_val3_symm0.3k1_k24_normed_${i}/cp_step5901.pt" \
        --data_dir "../data/sax_major_all" \
        --save_results_at "experiments/inducement/results/csv/major_sav_val3_induced_probe_0914.csv"
done
for i in {0..4}; do
    python run_evaluation.py \
        --probing 1 \
        --active_checkpoint "experiments/inducement/logs/major_sax_induced_val3_nosymm_k24_normed_${i}/cp_step5901.pt" \
        --data_dir "../data/sax_major_all" \
        --save_results_at "experiments/inducement/results/csv/major_sav_val3_induced_probe_0914.csv"
done
