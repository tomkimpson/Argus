#!/bin/bash 
 
#SBATCH --job-name=numpyro_sandbox 
#SBATCH --output=test_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=00:5:00 
#SBATCH --mem=50G


source ~/.bashrc
conda activate Argus
time python benchmark_parameter_estimation.py