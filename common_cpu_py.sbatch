#!/bin/bash
#SBATCH -J S3plus_eval
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -N 1
#SBATCH --time=1:00:00
#SBATCH --output=slurm_common_cpu.out
#SBATCH --error=slurm_common_cpu.err
#SBATCH -p cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos

py_run_file=$1
args=$2

source activate xxxxj


cd /l/users/xuanjie.liu/S3Plus
python $py_run_file $args
