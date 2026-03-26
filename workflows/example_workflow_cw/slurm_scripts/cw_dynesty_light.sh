#!/bin/bash

#SBATCH --job-name=cw_dyn_lt
#SBATCH --output=outputs/logfiles/cw_dynesty_light_output.txt
#SBATCH --export=ALL
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

source ~/.bashrc
conda activate Argus
export PYTHONPATH=/fred/oz022/tkimpson/parallel-tempering/python:$PYTHONPATH

# Persistent JAX compilation cache
export JAX_COMPILATION_CACHE_DIR=/fred/oz022/tkimpson/.jax_cache
mkdir -p $JAX_COMPILATION_CACHE_DIR

time python -u run_analysis.py configs/dynesty_light_config.ini
