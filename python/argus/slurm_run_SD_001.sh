#!/bin/bash 
 
#SBATCH --job-name=savage_dickey_test_001
#SBATCH --output=outputs/logfiles/savage_dickey_test_001_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00 
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

source ~/.bashrc
conda activate Argus
time python -u main.py configs/savage_dickey_test_001.ini