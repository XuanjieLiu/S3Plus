#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=s3plus
#SBATCH --output=slurm_train.out
#SBATCH --error=slurm_train.err
#SBATCH --partition=cscc-gpu-p 
#SBATCH --qos=cscc-gpu-qos


exp_list=$1
echo "The Exp list is: $exp_list"


nvidia-smi -a
echo $CUDA_VISIBLE_DEVICES
hostname


source activate xxxxj


cd /l/users/xuanjie.liu/S3Plus/VQ
python batch_train.py $exp_list
