#!/bin/bash 
 
#SBATCH --job-name=test_run_mdc2_parameter_estimation_NUTS     
#SBATCH --output=test_run_mdc2_parameter_estimation_NUTS_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00 
#SBATCH --mem=8G


source ~/.bashrc
conda activate Argus
time python test_run_mdc2_parameter_estimation_NUTS.py