#!/bin/bash 
 
#SBATCH --job-name=multi_parameter_estimation_with_nested_sampling3
#SBATCH --output=outputs/logfiles/multi_parameter_estimation_with_nested_sampling_output3.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00 
#SBATCH --mem=8G


source ~/.bashrc
conda activate Argus
time python -u multi_parameter_estimation_with_nested_sampling.py