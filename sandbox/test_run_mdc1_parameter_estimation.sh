#!/bin/bash 
 
#SBATCH --job-name=test_run_mdc1_parameter_estimation     
#SBATCH --output=test_run_mdc1_parameter_estimation_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00 
#SBATCH --mem=6G


source ~/.bashrc
conda activate Argus
time python test_run_mdc1_parameter_estimation.py