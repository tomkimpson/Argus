#!/bin/bash

# Helper script to create a configuration file using the Argus CLI

echo "=== Creating Argus Configuration File ==="
echo

# Check if argus CLI is available
if ! command -v argus &> /dev/null; then
    echo "ERROR: argus command not found!"
    echo "Please install the Argus package first:"
    echo "  pip install argus-pta"
    echo "  # OR from source:"
    echo "  pip install ."
    exit 1
fi

# Create configs directory if it doesn't exist
mkdir -p configs

# Generate template config
CONFIG_FILE="configs/my_analysis.ini"
echo "Generating configuration template: $CONFIG_FILE"

argus init -o "$CONFIG_FILE"

if [ $? -eq 0 ]; then
    echo "✓ Configuration template created successfully!"
    echo
    echo "Next steps:"
    echo "1. Edit $CONFIG_FILE with your specific parameters"
    echo "2. Update data paths, analysis settings, etc."
    echo "3. Submit job with: ./submit_job.sh"
    echo
    echo "Example sections to configure:"
    echo "  [Data] - Set data_path and noise_params_path"
    echo "  [Inference] - Set nsamples, nwarmup, nchains"
    echo "  [PriorModel] - Set parameter ranges"
    echo "  [Output] - Set output directory and options"
else
    echo "✗ Failed to create configuration template!"
    exit 1
fi