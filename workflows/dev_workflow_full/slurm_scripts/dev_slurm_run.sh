#!/bin/bash 
 
#SBATCH --job-name=dev_run
#SBATCH --output=outputs/logfiles/dev_run_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

source ~/.bashrc
conda activate Argus
time python -u run_analysis.py configs/dev_config.ini