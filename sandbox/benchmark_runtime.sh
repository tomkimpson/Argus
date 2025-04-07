#!/bin/bash 
 
#SBATCH --job-name=benchmark_runtime
#SBATCH --output=benchmark_runtime.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=0:5:00 
#SBATCH --mem=10G


source ~/.bashrc
conda activate Argus
time python benchmark_runtime_jax.py