#!/bin/bash 
 
#SBATCH --job-name=dev_example_run
#SBATCH --output=outputs/logfiles/dev_example_run_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00 
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

source ~/.bashrc
conda activate Argus
time python -u run_analysis.py configs/dev_example_config.ini