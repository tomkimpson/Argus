#!/bin/bash 
 
#SBATCH --job-name=parameter_estimation_example
#SBATCH --output=parameter_estimation_example_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=4:0:00 
#SBATCH --mem=10G


source ~/.bashrc
conda activate Argus
time python parameter_estimation_example.py