#!/bin/bash 
 
#SBATCH --job-name=numpyro_test_010
#SBATCH --output=outputs/logfiles/numpyro_test_010_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:2
#SBATCH --time=2:00:15 
#SBATCH --mem=8G


source ~/.bashrc
conda activate Argus
time python -u main.py configs/config_numpyro_test_010.ini