#!/bin/bash 
 
#SBATCH --job-name=three_params_test_001
#SBATCH --output=outputs/logfiles/three_params_test_001_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=1:30:00 
#SBATCH --mem=8G


source ~/.bashrc
conda activate Argus
time python -u main.py configs/three_params_test_001.ini