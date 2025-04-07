#!/bin/bash 
 
#SBATCH --job-name=SA_benchmark
#SBATCH --output=SA_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=2:0:00 
#SBATCH --mem=10G


source ~/.bashrc
conda activate Argus
time python benchmark_parameter_estimation.py