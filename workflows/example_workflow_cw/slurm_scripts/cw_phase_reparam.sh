#!/bin/bash

#SBATCH --job-name=cw_phase
#SBATCH --output=outputs/logfiles/cw_phase_reparam_output.txt
#SBATCH --export=ALL
#SBATCH --gres=gpu:p100:2
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

source ~/.bashrc
conda activate Argus
time python -u run_analysis.py configs/phase_reparam_config.ini
