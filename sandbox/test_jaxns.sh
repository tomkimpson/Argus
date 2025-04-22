#!/bin/bash 
 
#SBATCH --job-name=test_jaxns
#SBATCH --output=test_jaxns_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00 
#SBATCH --mem=10G


source ~/.bashrc
conda activate Argus
time python test_jaxns.py