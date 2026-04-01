#!/bin/bash

#SBATCH --job-name=cw_rev2
#SBATCH --output=outputs/logfiles/cw_replica_exchange_v2_output.txt
#SBATCH --export=ALL
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

source ~/.bashrc
conda activate Argus
echo "Starting replica exchange v2 (12 chains, 5 HMC steps, 10 leapfrog, 0.5x step size)"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
time python -u run_analysis.py configs/replica_exchange_v2_config.ini
