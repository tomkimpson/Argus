#!/bin/bash

#SBATCH --job-name=cw_smc_lt
#SBATCH --output=outputs/logfiles/cw_smc_light_output.txt
#SBATCH --export=ALL
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

source ~/.bashrc
conda activate Argus
export PYTHONPATH=/fred/oz022/tkimpson/parallel-tempering/python:$PYTHONPATH

# Persistent JAX compilation cache — reuse compiled kernels across runs
export JAX_COMPILATION_CACHE_DIR=/fred/oz022/tkimpson/.jax_cache
mkdir -p $JAX_COMPILATION_CACHE_DIR

time python -u run_analysis.py configs/tempered_smc_light_config.ini
