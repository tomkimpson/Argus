#!/bin/bash 
 
#SBATCH --job-name=ha_test_002
#SBATCH --output=outputs/logfiles/ha_test_002_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=1:30:00 
#SBATCH --mem=8G


source ~/.bashrc
conda activate Argus
time python -u main.py configs/ha_test_002.ini