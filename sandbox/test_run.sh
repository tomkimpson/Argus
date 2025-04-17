#!/bin/bash 
 
#SBATCH --job-name=test_run
#SBATCH --output=test_run_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=0:10:00 
#SBATCH --mem=4G


source ~/.bashrc
conda activate Argus
time python test_run.py