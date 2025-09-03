#!/bin/bash 
 
#SBATCH --job-name=dev_run
#SBATCH --output=outputs/logfiles/dev_run_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

source ~/.bashrc
conda activate Argus
time python -u run_analysis.py configs/dev_config.ini