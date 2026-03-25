#!/bin/bash

#SBATCH --job-name=cw_fpt
#SBATCH --output=outputs/logfiles/cw_fixed_fgw_pulsar_term_output.txt
#SBATCH --export=ALL
#SBATCH --gres=gpu:p100:2
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

source ~/.bashrc
conda activate Argus
time python -u run_analysis.py configs/fixed_fgw_pulsar_term_config.ini
