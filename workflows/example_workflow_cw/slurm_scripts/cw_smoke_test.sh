#!/bin/bash

#SBATCH --job-name=cw_smoke_test
#SBATCH --output=outputs/logfiles/cw_smoke_test_output.txt
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

source ~/.bashrc
conda activate Argus
time python -u run_analysis.py configs/smoke_test_config.ini
