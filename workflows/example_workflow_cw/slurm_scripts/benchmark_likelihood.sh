#!/bin/bash

#SBATCH --job-name=bench_ll
#SBATCH --output=outputs/logfiles/benchmark_likelihood_output.txt
#SBATCH --export=ALL
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

source ~/.bashrc
conda activate Argus
export PYTHONPATH=/fred/oz022/tkimpson/parallel-tempering/python:$PYTHONPATH
export JAX_COMPILATION_CACHE_DIR=/fred/oz022/tkimpson/.jax_cache
mkdir -p $JAX_COMPILATION_CACHE_DIR

python -u benchmark_likelihood.py
