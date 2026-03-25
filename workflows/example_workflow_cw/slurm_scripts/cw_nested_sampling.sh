#!/bin/bash

#SBATCH --job-name=cw_nested
#SBATCH --output=outputs/logfiles/cw_nested_sampling_output.txt
#SBATCH --export=ALL
#SBATCH --gres=gpu:p100:2
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

source ~/.bashrc
conda activate Argus
time python -u run_analysis.py configs/nested_sampling_config.ini
