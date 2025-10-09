#!/bin/bash

#SBATCH --job-name=example_run
#SBATCH --output=outputs/logfiles/example_run_output.txt
#SBATCH --export=ALL
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

# Create output directories if they don't exist
mkdir -p outputs/logfiles

source ~/.bashrc
conda activate argus-env
time python -u run_analysis.py configs/example_config.ini