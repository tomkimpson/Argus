#!/bin/bash

#SBATCH --job-name=cw_fixfgw
#SBATCH --output=outputs/logfiles/cw_fixed_fgw_output.txt
#SBATCH --export=ALL
#SBATCH --gres=gpu:p100:2
#SBATCH --time=8:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

source ~/.bashrc
conda activate Argus
time python -u run_analysis.py configs/fixed_fgw_config.ini
