#!/bin/bash 
 
#SBATCH --job-name=tmp_parameter_estimation
#SBATCH --output=outputs/logfiles/tmp_parameter_estimation_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=0:10:00 
#SBATCH --mem=8G


source ~/.bashrc
conda activate Argus
time python -u tmp_parameter_estimation.py