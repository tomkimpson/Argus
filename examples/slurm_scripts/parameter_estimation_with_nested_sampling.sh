#!/bin/bash 
 
#SBATCH --job-name=parameter_estimation_with_nested_sampling_prior_testing
#SBATCH --output=outputs/logfiles/parameter_estimation_with_nested_sampling_prior_testing_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=0:10:00 
#SBATCH --mem=8G
#SBATCH --partition=milan

source ~/.bashrc
conda activate Argus
time python -u parameter_estimation_with_nested_sampling.py