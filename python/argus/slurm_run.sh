#!/bin/bash 
 
#SBATCH --job-name=dev_test
#SBATCH --output=outputs/logfiles/dev_test_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00 
#SBATCH --mem=8G


source ~/.bashrc
conda activate Argus
time python -u main.py configs/config.ini