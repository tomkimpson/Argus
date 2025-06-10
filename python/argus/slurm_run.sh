#!/bin/bash 
 
#SBATCH --job-name=numpyro_test_012
#SBATCH --output=outputs/logfiles/numpyro_test_012_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:15 
#SBATCH --mem=8G


source ~/.bashrc
conda activate Argus
time python -u main.py configs/config_numpyro_test_012.ini