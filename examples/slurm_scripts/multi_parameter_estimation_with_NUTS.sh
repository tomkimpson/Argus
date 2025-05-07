#!/bin/bash
 
#SBATCH --job-name=multi_parameter_estimation_with_NUTS
#SBATCH --output=outputs/logfiles/multi_parameter_estimation_with_NUTS_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00 
#SBATCH --mem=8G


source ~/.bashrc
conda activate Argus
time python multi_parameter_estimation_with_NUTS.py