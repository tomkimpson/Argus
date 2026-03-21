#!/bin/bash

#SBATCH --job-name=cw_light4c
#SBATCH --output=outputs/logfiles/cw_light_4chain_output.txt
#SBATCH --export=ALL
#SBATCH --gres=gpu:p100:2
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

source ~/.bashrc
conda activate Argus
time python -u run_analysis.py configs/light_4chain_cw_config.ini
