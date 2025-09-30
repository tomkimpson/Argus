#!/bin/bash

#SBATCH --job-name=example_run_lite
#SBATCH --output=outputs/logfiles/example_run_lite_output.txt
#SBATCH --export=ALL
#SBATCH --gres=gpu:4
#SBATCH --time=2:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

source ~/.bashrc
conda activate argus-pta
time python -u run_analysis.py configs/example_config_lite.ini