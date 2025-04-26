#!/bin/bash 
 
#SBATCH --job-name=test_run_mdc2_parameter_estimation_nested_sampling     
#SBATCH --output=test_run_mdc2_parameter_estimation_nested_sampling_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00 
#SBATCH --mem=10G


source ~/.bashrc
conda activate Argus
time python test_run_mdc2_parameter_estimation_nested_sampling.py