#!/bin/bash

#SBATCH --job-name=cw_8chD
#SBATCH --output=outputs/logfiles/cw_8chain_D_output.txt
#SBATCH --export=ALL
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --time=09:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

source ~/.bashrc
conda activate Argus
time python -u run_analysis.py configs/phase_reparam_8chain_D_config.ini
