#!/bin/bash 
 
#SBATCH --job-name=parameter_estimation_with_nested_sampling_just_ha_time_test
#SBATCH --output=outputs/logfiles/parameter_estimation_with_nested_sampling_just_ha_time_test_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00 
#SBATCH --mem=8G
#SBATCH --partition=milan

source ~/.bashrc
conda activate Argus
time python -u parameter_estimation_with_nested_sampling.py