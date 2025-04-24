#!/bin/bash 
 
#SBATCH --job-name=test_run_mdc2_likelihood_curve 
#SBATCH --output=test_run_mdc2_likelihood_curve_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=0:10:00 
#SBATCH --mem=4G


source ~/.bashrc
conda activate Argus
time python test_run_mdc2_likelihood_curve.py