#!/bin/bash

#SBATCH --job-name=ent_cw_smp
#SBATCH --output=outputs/logfiles/enterprise_cw_sampled_noise_output.txt
#SBATCH --export=ALL
#SBATCH --partition=milan
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

source ~/.bashrc
conda activate Argus

echo "Starting ENTERPRISE CW search (sampled noise)"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"

time python -u enterprise_cw_search.py --mode sampled --n-samples 1000000

echo "End time: $(date)"
