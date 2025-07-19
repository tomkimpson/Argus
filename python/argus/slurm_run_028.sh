#!/bin/bash 
 
#SBATCH --job-name=numpyro_test_028
#SBATCH --output=outputs/logfiles/numpyro_test_028_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:4
#SBATCH --time=96:00:00 
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

source ~/.bashrc
conda activate Argus
time python -u main.py configs/config_numpyro_test_028.ini