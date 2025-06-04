#!/bin/bash 
 
#SBATCH --job-name=numpyro_test_002
#SBATCH --output=outputs/logfiles/numpyro_test_002_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00 
#SBATCH --mem=8G


source ~/.bashrc
conda activate Argus
time python -u main.py configs/config_numpyro_test.ini