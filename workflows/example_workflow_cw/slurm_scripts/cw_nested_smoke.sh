#!/bin/bash

#SBATCH --job-name=ns_smoke
#SBATCH --output=outputs/logfiles/cw_nested_smoke_output.txt
#SBATCH --export=ALL
#SBATCH --gres=gpu:p100:1
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

source ~/.bashrc
conda activate Argus
time python -u run_analysis.py configs/nested_smoke_test_config.ini
