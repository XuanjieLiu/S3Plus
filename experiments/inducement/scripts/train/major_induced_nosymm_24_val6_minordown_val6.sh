for i in {0..4}; do
  python run_training.py \
    --config model/inducement/configs/cfg_isymm_induced_downstream.yaml \
    --name major_sax_induced_val6_nosymm_k24_normed_${i}_minordown_val6 \
    --data_dir "../data/sax_minordown_val_ood_spectrum/sax_minordown_val6" \
    --data_type 6 \
    --loss_config.weights.prior_loss 1 \
    --model_config.d_zs 0 \
    --model_config.n_atoms 24 \
    --model_config.ae_checkpoint "experiments/inducement/logs/major_sax_induced_val6_nosymm_k24_normed_${i}/cp_step9901.pt" \
    --batch_size 192 \
    --steps 6000
done