#!/bin/bash

# Convenience script to submit the Argus CLI full inference job to Slurm

echo "=== Submitting Argus CLI Full Inference Analysis Job ==="
echo

# Check if config file exists
CONFIG_FILE="configs/cli_config.ini"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Configuration file '$CONFIG_FILE' not found!"
    echo "Please create it first using:"
    echo "  argus init -o $CONFIG_FILE"
    echo "Then edit it with your specific parameters."
    exit 1
fi

# Check if argus CLI is available
if ! command -v argus &> /dev/null; then
    echo "WARNING: argus command not found in current environment."
    echo "Make sure to install the package in your compute environment:"
    echo "  pip install argus-pta"
    echo "  # OR from source:"
    echo "  pip install ."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p outputs/logfiles

echo "Configuration file: $CONFIG_FILE"
echo "Slurm script: slurm_scripts/cli_slurm_run.sh"
echo "Output log: outputs/logfiles/cli_full_run_output.txt"
echo

# Submit the job
sbatch slurm_scripts/cli_slurm_run.sh

if [ $? -eq 0 ]; then
    echo "✓ Job submitted successfully!"
    echo
    echo "Monitor progress with:"
    echo "  squeue -u \$USER"
    echo "  tail -f outputs/logfiles/cli_full_run_output.txt"
else
    echo "✗ Job submission failed!"
    exit 1
fi