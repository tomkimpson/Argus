#!/bin/bash

#SBATCH --job-name=cw_smc
#SBATCH --output=outputs/logfiles/cw_tempered_smc_output.txt
#SBATCH --export=ALL
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

source ~/.bashrc
conda activate Argus
time python -u run_analysis.py configs/tempered_smc_config.ini
