#!/bin/bash 
 
#SBATCH --job-name=sample_adaptive
#SBATCH --output=sample_adaptive_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=4:0:00 
#SBATCH --mem=10G


source ~/.bashrc
conda activate Argus
time python sample_adaptive.py
