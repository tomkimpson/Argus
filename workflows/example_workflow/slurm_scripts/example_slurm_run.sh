#!/bin/bash 
 
#SBATCH --job-name=example_run
#SBATCH --output=outputs/logfiles/example_run_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:4
#SBATCH --time=96:00:00 
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

source ~/.bashrc
conda activate Argus
time python -u run_analysis.py configs/example_config.ini