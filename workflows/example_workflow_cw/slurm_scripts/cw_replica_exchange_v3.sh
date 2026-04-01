#!/bin/bash

#SBATCH --job-name=cw_rev3
#SBATCH --output=outputs/logfiles/cw_replica_exchange_v3_output.txt
#SBATCH --export=ALL
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

source ~/.bashrc
conda activate Argus
echo "Starting replica exchange v3 (12 chains, beta_hot=0.1, geometric)"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
time python -u run_analysis.py configs/replica_exchange_v3_config.ini
