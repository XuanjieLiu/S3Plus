# run 0.3 only

# interleave with nosymm runs
for i in {0..4}; do
  python run_training.py \
    --config model/transition/configs/cfg_isymm_transition.yaml \
    --name major_sax_transition_val6_symm1_k24_normed_cos_ctn_$i \
    --load_checkpoint "experiments/transition/logs/major_sax_transition_val6_symm1_k24_normed_cos_$i/cp_step9901.pt" \
    --data_dir "../data/sax_major_val_ood_spectrum_2/sax_major_val6" \
    --data_type 6 \
    --loss_config.weights.prior_loss 1 \
    --loss_config.weights.isymm_loss 1 \
    --loss_config.isymm_k 4 \
    --model_config.d_zs 0 \
    --model_config.n_atoms 24 \
    --batch_size 256 \
    --steps 20000 \
    --optimizer_config.scheduler "cosine_annealing" \
    --optimizer_config.lr_anneal_min_factor 0.001

  # python run_training.py \
  #   --config model/transition/configs/cfg_isymm_transition.yaml \
  #   --name major_sax_transition_val6_nosymm_k24_normed_$i \
  #   --data_dir "../data/sax_major_val_ood_spectrum_2/sax_major_val6" \
  #   --data_type 6 \
  #   --loss_config.weights.prior_loss 1 \
  #   --loss_config.weights.isymm_loss 0 \
  #   --loss_config.isymm_k 4 \
  #   --model_config.d_zs 0 \
  #   --model_config.n_atoms 24 \
  #   --batch_size 256 \
  #   --steps 6000
done