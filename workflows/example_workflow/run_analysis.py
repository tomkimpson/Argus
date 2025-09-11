"""Main script for running Bayesian inference on pulsar timing data."""

import os
import sys
import argparse
from datetime import datetime

import jax
jax.config.update("jax_enable_x64", True)

# Add the python directory to path to import argus modules
# Go up to project root and then to python directory
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_root, 'python'))

from argus import utils, workflow


def main():
    """Run main entry point for Bayesian inference on pulsar timing data."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run Bayesian inference on pulsar timing data.')
    parser.add_argument('config', type=str, help='Path to the configuration file')
    
    args = parser.parse_args()
    
    # Print JAX configuration and system info
    print("=== JAX VERSION INFO ===")
    print(f"JAX version: {jax.__version__}")
    print("\n=== DEVICE INFO ===")
    print("Default device:", jax.default_backend())

    print("You are working with the development version of the code. Good job.")
    
    # Check GPU availability
    utils.check_gpu_availability()
    
    # Create a single timestamp for both runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run the inference workflow
    output_dir = workflow.run_inference(
        config_path=args.config,
        use_gw=True, 
        timestamp=timestamp
    )
    
    print(f"\nInference complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()