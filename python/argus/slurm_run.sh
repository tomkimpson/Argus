#!/bin/bash 
 
#SBATCH --job-name=numpyro_test_011
#SBATCH --output=outputs/logfiles/numpyro_test_011_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:15 
#SBATCH --mem=8G


source ~/.bashrc
conda activate Argus
time python -u main.py configs/config_numpyro_test_011.ini