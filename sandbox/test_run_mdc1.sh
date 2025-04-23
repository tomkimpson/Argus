#!/bin/bash 
 
#SBATCH --job-name=test_run_mdc1 
#SBATCH --output=test_run_mdc1_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=0:10:00 
#SBATCH --mem=4G


source ~/.bashrc
conda activate Argus
time python test_run_mdc1.py