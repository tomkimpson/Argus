#!/bin/bash 
 
#SBATCH --job-name=parameter_estimation_with_nested_sampling_log_test    
#SBATCH --output=outputs/logfiles/parameter_estimation_with_nested_sampling_output_test_logging.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00 
#SBATCH --mem=8G


source ~/.bashrc
conda activate Argus
time python parameter_estimation_with_nested_sampling.py