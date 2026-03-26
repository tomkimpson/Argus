#!/bin/bash

#SBATCH --job-name=cw_smc_hv
#SBATCH --output=outputs/logfiles/cw_smc_heavy_output.txt
#SBATCH --export=ALL
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

source ~/.bashrc
conda activate Argus
export PYTHONPATH=/fred/oz022/tkimpson/parallel-tempering/python:$PYTHONPATH

# Persistent JAX compilation cache — reuse compiled kernels across runs
export JAX_COMPILATION_CACHE_DIR=/fred/oz022/tkimpson/.jax_cache
mkdir -p $JAX_COMPILATION_CACHE_DIR

time python -u run_analysis.py configs/tempered_smc_heavy_config.ini
