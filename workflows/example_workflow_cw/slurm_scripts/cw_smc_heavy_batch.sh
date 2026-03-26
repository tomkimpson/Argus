#!/bin/bash

#SBATCH --job-name=cw_smc_hv
#SBATCH --output=outputs/logfiles/cw_smc_heavy_%a_output.txt
#SBATCH --export=ALL
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=42,43,44,45

source ~/.bashrc
conda activate Argus
export PYTHONPATH=/fred/oz022/tkimpson/parallel-tempering/python:$PYTHONPATH

# Persistent JAX compilation cache
export JAX_COMPILATION_CACHE_DIR=/fred/oz022/tkimpson/.jax_cache
mkdir -p $JAX_COMPILATION_CACHE_DIR

SEED=$SLURM_ARRAY_TASK_ID
echo "Running SMC batch with seed=${SEED}"
time python -u run_analysis.py configs/tempered_smc_heavy_seed${SEED}_config.ini
