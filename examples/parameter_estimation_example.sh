#!/bin/bash 
 
#SBATCH --job-name=parameter_estimation_example
#SBATCH --output=parameter_estimation_example_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=6:0:00 
#SBATCH --mem=24G


source ~/.bashrc
conda activate Argus
export PYTHONPATH="/fred/oz022/tkimpson/tmp/Argus/python:${PYTHONPATH}"
time python parameter_estimation_example.py