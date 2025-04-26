#!/bin/bash 
 
#SBATCH --job-name=test_run_mdc2_parameter_estimation_NUTS     
#SBATCH --output=test_run_mdc2_parameter_estimation_NUTS_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00 
#SBATCH --mem=6G


source ~/.bashrc
conda activate Argus
time python test_run_mdc2_parameter_estimation_NUTS.py