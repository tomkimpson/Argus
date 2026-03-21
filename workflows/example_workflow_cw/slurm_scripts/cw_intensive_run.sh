#!/bin/bash

#SBATCH --job-name=cw_intensive
#SBATCH --output=outputs/logfiles/cw_intensive_run_output.txt
#SBATCH --export=ALL
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

source ~/.bashrc
conda activate Argus
time python -u run_analysis.py configs/intensive_cw_config.ini
