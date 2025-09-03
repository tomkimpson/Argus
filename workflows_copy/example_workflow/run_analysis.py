"""Main script for running Bayesian inference on pulsar timing data."""

import os
import sys
import argparse
from datetime import datetime

import jax
jax.config.update("jax_enable_x64", True)

# Add the parent directory to path to import argus modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    
    # Run the full model comparison workflow
    gw_output_dir, no_gw_output_dir, bayes_factor_results = workflow.run_model_comparison(
        config_path=args.config, 
        timestamp=timestamp
    )
    
    print(f"\nInference complete! Results saved to: {gw_output_dir}")
    if no_gw_output_dir:
        print(f"No-GW results saved to: {no_gw_output_dir}")


if __name__ == "__main__":
    main()