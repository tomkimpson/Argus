#!/bin/bash 
 
#SBATCH --job-name=test_parameter_estimation_ha
#SBATCH --output=test_parameter_estimation_ha_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00 
#SBATCH --mem=6G


source ~/.bashrc
conda activate Argus
time python test_parameter_estimation_ha.py