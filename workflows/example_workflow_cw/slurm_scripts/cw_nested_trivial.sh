#!/bin/bash

#SBATCH --job-name=ns_triv
#SBATCH --output=outputs/logfiles/cw_nested_trivial_output.txt
#SBATCH --export=ALL
#SBATCH --gres=gpu:p100:1
#SBATCH --time=2:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

source ~/.bashrc
conda activate Argus
time python -u run_analysis.py configs/nested_trivial_config.ini
