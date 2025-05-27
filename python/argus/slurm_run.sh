#!/bin/bash 
 
#SBATCH --job-name=nested_sampling_test
#SBATCH --output=outputs/logfiles/nested_sampling_test_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00 
#SBATCH --mem=8G


source ~/.bashrc
conda activate Argus
time python -u main.py config.ini