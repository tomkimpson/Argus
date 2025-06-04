#!/bin/bash 
 
#SBATCH --job-name=jaxns_test_001
#SBATCH --output=outputs/logfiles/jaxns_test_001_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:15 
#SBATCH --mem=8G


source ~/.bashrc
conda activate Argus
time python -u main.py configs/config_jaxns_test.ini