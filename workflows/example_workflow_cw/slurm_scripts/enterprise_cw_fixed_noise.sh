#!/bin/bash

#SBATCH --job-name=ent_cw_fix
#SBATCH --output=outputs/logfiles/enterprise_cw_fixed_noise_output.txt
#SBATCH --export=ALL
#SBATCH --partition=milan
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

source ~/.bashrc
conda activate Argus

echo "Starting ENTERPRISE CW search (fixed noise)"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"

time python -u enterprise_cw_search.py --mode fixed --n-samples 500000

echo "End time: $(date)"
