#!/bin/bash 
 
#SBATCH --job-name=numpyro_test_001
#SBATCH --output=outputs/logfiles/numpyro_test_001_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00 
#SBATCH --mem=8G


source ~/.bashrc
conda activate Argus
time python -u main.py configs/config_numpyro.ini