# python run_training.py \
#     --config model/inducement/configs/cfg_isymm_induced_downstream.yaml \
#     --name major_sax_induced_val6_symm0.3k1_k24_normed_0_minordown_val6 \
#     --data_dir "../data/sax_minordown_val_ood_spectrum/sax_minordown_val6" \
#     --data_type 6 \
#     --loss_config.weights.prior_loss 1 \
#     --model_config.d_zs 0 \
#     --model_config.n_atoms 24 \
#     --model_config.ae_checkpoint "experiments/inducement/logs/major_sax_induced_val6_symm0.3k1_k24_normed_0/cp_step3601.pt" \
#     --batch_size 192 \
#     --steps 6000
python run_training.py \
    --config model/inducement/configs/cfg_isymm_induced_downstream.yaml \
    --name major_sax_induced_val6_symm0.3k1_k24_normed_1_minordown_val6 \
    --data_dir "../data/sax_minordown_val_ood_spectrum/sax_minordown_val6" \
    --data_type 6 \
    --loss_config.weights.prior_loss 1 \
    --model_config.d_zs 0 \
    --model_config.n_atoms 24 \
    --model_config.ae_checkpoint "experiments/inducement/logs/major_sax_induced_val6_symm0.3k1_k24_normed_1/cp_step5901.pt" \
    --batch_size 192 \
    --steps 6000
# python run_training.py \
#     --config model/inducement/configs/cfg_isymm_induced_downstream.yaml \
#     --name major_sax_induced_val6_symm0.3k1_k24_normed_2_minordown_val6 \
#     --data_dir "../data/sax_minordown_val_ood_spectrum/sax_minordown_val6" \
#     --data_type 6 \
#     --loss_config.weights.prior_loss 1 \
#     --model_config.d_zs 0 \
#     --model_config.n_atoms 24 \
#     --model_config.ae_checkpoint "experiments/inducement/logs/major_sax_induced_val6_symm0.3k1_k24_normed_2/cp_step5901.pt" \
#     --batch_size 192 \
#     --steps 6000
# python run_training.py \
#     --config model/inducement/configs/cfg_isymm_induced_downstream.yaml \
#     --name major_sax_induced_val6_symm0.3k1_k24_normed_3_minordown_val6 \
#     --data_dir "../data/sax_minordown_val_ood_spectrum/sax_minordown_val6" \
#     --data_type 6 \
#     --loss_config.weights.prior_loss 1 \
#     --model_config.d_zs 0 \
#     --model_config.n_atoms 24 \
#     --model_config.ae_checkpoint "experiments/inducement/logs/major_sax_induced_val6_symm0.3k1_k24_normed_3/cp_step5901.pt" \
#     --batch_size 192 \
#     --steps 6000
python run_training.py \
    --config model/inducement/configs/cfg_isymm_induced_downstream.yaml \
    --name major_sax_induced_val6_symm0.3k1_k24_normed_4_minordown_val6 \
    --data_dir "../data/sax_minordown_val_ood_spectrum/sax_minordown_val6" \
    --data_type 6 \
    --loss_config.weights.prior_loss 1 \
    --model_config.d_zs 0 \
    --model_config.n_atoms 24 \
    --model_config.ae_checkpoint "experiments/inducement/logs/major_sax_induced_val6_symm0.3k1_k24_normed_4/cp_step4901.pt" \
    --batch_size 192 \
    --steps 6000