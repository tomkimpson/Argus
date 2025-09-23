#!/bin/bash 
 
#SBATCH --job-name=argus_cli_full_run
#SBATCH --output=outputs/logfiles/cli_full_run_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00 
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

echo "=== Argus CLI Full Inference Workflow Started ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME" 
echo "Start time: $(date)"
echo

# Activate environment (adjust path as needed)
source ~/.bashrc
conda activate Argus

echo "=== Environment Information ==="
echo "Python version: $(python --version)"
echo "Argus version: $(argus --version)"
echo "Working directory: $(pwd)"
echo

# Verify argus CLI is available
if ! command -v argus &> /dev/null; then
    echo "ERROR: argus command not found. Please install the package:"
    echo "  pip install argus-pta"
    echo "  # OR"
    echo "  pip install ."
    exit 1
fi

echo "=== Starting Bayesian Inference Analysis ==="
echo "Configuration file: configs/cli_config.ini"
echo

# Run the analysis using the CLI
# This replaces: python -u run_analysis.py configs/example_config.ini
time argus run configs/cli_config.ini

echo
echo "=== Job Completed ==="
echo "End time: $(date)"
echo "Check outputs/ directory for results"