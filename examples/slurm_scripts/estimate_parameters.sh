#!/bin/bash 
 
#SBATCH --job-name=estimate_parameters_just_ha
#SBATCH --output=outputs/logfiles/estimate_parameters_just_ha_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00 
#SBATCH --mem=8G


source ~/.bashrc
conda activate Argus
time python -u estimate_parameters.py