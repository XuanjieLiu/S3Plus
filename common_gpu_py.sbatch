#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=s3plus
#SBATCH --output=slurm_common_gpu.out
#SBATCH --error=slurm_common_gpu.err
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos


py_run_file=$1
args=$2


nvidia-smi -a
echo $CUDA_VISIBLE_DEVICES
hostname


source activate xxxxj


cd /l/users/xuanjie.liu/S3Plus
python $py_run_file $args