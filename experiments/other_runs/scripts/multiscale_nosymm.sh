python run_training.py --config cfg_isymm.yaml --name multiscale_nosymm_k60 --loss_config.weights.isymm_loss 0 --model_config.n_atoms 60 --batch_size 128

python run_training.py --config cfg_isymm.yaml --name multiscale_nosymm_k12 --loss_config.weights.isymm_loss 0 --model_config.n_atoms 12 --batch_size 128
python run_training.py --config cfg_isymm.yaml --name multiscale_nosymm_k24 --loss_config.weights.isymm_loss 0 --model_config.n_atoms 24 --batch_size 128
python run_training.py --config cfg_isymm.yaml --name multiscale_nosymm_k36 --loss_config.weights.isymm_loss 0 --model_config.n_atoms 36 --batch_size 128
python run_training.py --config cfg_isymm.yaml --name multiscale_nosymm_k60 --loss_config.weights.isymm_loss 0 --model_config.n_atoms 60 --batch_size 128
