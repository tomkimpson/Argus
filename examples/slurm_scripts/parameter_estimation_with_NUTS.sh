#!/bin/bash
timestamp=$(date +'%Y%m%d_%H%M%S') 
 
#SBATCH --job-name=parameter_estimation_with_NUTS_${timestamp}
#SBATCH --output=outputs/logfiles/parameter_estimation_with_NUTS_output_${timestamp}.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:2
#SBATCH --time=1:00:00 
#SBATCH --mem=8G


source ~/.bashrc
conda activate Argus
time python parameter_estimation_with_NUTS.py