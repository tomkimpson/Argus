#!/bin/bash

#SBATCH --job-name=cw_full_run
#SBATCH --output=outputs/logfiles/cw_full_run_output.txt
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --time=1:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

source ~/.bashrc
conda activate Argus
time python -u run_analysis.py configs/full_cw_config.ini
