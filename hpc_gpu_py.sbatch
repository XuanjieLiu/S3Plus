#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=my_python_job
#SBATCH --output=hpc_run_logs/%x-%j.out


py_run_file=$1
args=$2


nvidia-smi -a
echo $CUDA_VISIBLE_DEVICES
hostname

source ~/.bashrc
conda activate xuanjie


cd /home/xuanjie.liu/Projects/S3Plus
python $py_run_file $args