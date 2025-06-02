#!/bin/bash 
 
#SBATCH --job-name=ha_test_001
#SBATCH --output=outputs/logfiles/ha_test_001_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00 
#SBATCH --mem=8G


source ~/.bashrc
conda activate Argus
time python -u main.py configs/ha_test_001.ini