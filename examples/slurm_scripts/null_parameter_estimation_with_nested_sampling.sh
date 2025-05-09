#!/bin/bash 
 
#SBATCH --job-name=null_parameter_estimation_with_nested_sampling
#SBATCH --output=outputs/logfiles/null_parameter_estimation_with_nested_sampling_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00 
#SBATCH --mem=8G


source ~/.bashrc
conda activate Argus
time python -u null_parameter_estimation_with_nested_sampling.py