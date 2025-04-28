#!/bin/bash 
 
#SBATCH --job-name=parameter_estimation_with_nested_sampling     
#SBATCH --output=outputs/logfiles/parameter_estimation_with_nested_sampling_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00 
#SBATCH --mem=10G


source ~/.bashrc
conda activate Argus
time python parameter_estimation_with_nested_sampling.py