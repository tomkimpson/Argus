#!/bin/bash

#SBATCH --job-name=prof_pt
#SBATCH --output=outputs/logfiles/profile_pulsar_term_output.txt
#SBATCH --export=ALL
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

source ~/.bashrc
conda activate Argus
time python -u profile_likelihood.py --pulsar-term
