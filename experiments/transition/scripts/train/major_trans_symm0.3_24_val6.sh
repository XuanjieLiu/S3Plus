# interleave with nosymm runs
for i in {0..4}; do
  python run_training.py \
    --config model/transition/configs/cfg_isymm_transition.yaml \
    --name major_sax_transition_val6_symm0.3_k24_normed_cos_gru_gr7_$i \
    --data_dir "../data/sax_major_val_ood_spectrum_2/sax_major_val6" \
    --data_type 6 \
    --loss_config.weights.prior_loss 1 \
    --loss_config.weights.isymm_loss "min(0.3, t/4000)" \
    --loss_config.isymm_k 4 \
    --model_config.d_zs 0 \
    --model_config.n_atoms 24 \
    --model_config.GRU True \
    --batch_size 256 \
    --steps 6000 \
    --optimizer_config.scheduler "cosine_annealing" \
    --precision bfloat16
done