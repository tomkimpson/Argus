#!/bin/bash

#SBATCH --job-name=bench_gp
#SBATCH --output=outputs/logfiles/benchmark_gp_likelihood_output.txt
#SBATCH --export=ALL
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

source ~/.bashrc
conda activate Argus

python -u benchmark_gp_likelihood.py
