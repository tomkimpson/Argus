#!/bin/bash 
 
#SBATCH --job-name=test_run_mdc2_parameter_estimation     
#SBATCH --output=test_run_mdc2_parameter_estimation_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:4
#SBATCH --time=4:00:00 
#SBATCH --mem=24G


source ~/.bashrc
conda activate Argus
time python test_run_mdc2_parameter_estimation.py