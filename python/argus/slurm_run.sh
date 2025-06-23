#!/bin/bash 
 
#SBATCH --job-name=numpyro_test_018
#SBATCH --output=outputs/logfiles/numpyro_test_018_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00 
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

source ~/.bashrc
conda activate Argus
time python -u main.py configs/config_numpyro_test_018.ini