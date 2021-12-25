#!/bin/bash
#SBATCH --job-name RUN_EXP
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes 1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time 72:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --output "out/slurm-%A_%a.log"
#SBATCH --mem=32G
#SBATCH --cpus-per-task=10

module load gcc/8.4.0-cuda cuda/10.2.89
echo "${@:1}"
python -u "${@:1}"