"""Lightweight development script for rapid prototyping of Bayesian inference."""

import os
import sys
import argparse
from datetime import datetime

import jax

# Add the python directory to path to import argus modules
# Go up to project root and then to python directory
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_root, 'python'))

from argus import utils, workflow

jax.config.update("jax_enable_x64", True)


def main():
    """Run lightweight development version for rapid prototyping."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run lightweight Bayesian inference for development.')
    parser.add_argument('config', type=str, help='Path to the configuration file')
    
    args = parser.parse_args()
    
    # Print JAX configuration and system info
    print("=== DEV WORKFLOW - RAPID PROTOTYPING ===")
    print(f"JAX version: {jax.__version__}")
    print("Default device:", jax.default_backend())
    print("Note: This is a lightweight development workflow for rapid prototyping.")
    print("Use the full example_workflow for production runs.\n")
    
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
    
    print(f"\nDev inference complete! Results saved to: {output_dir}")
    
    # Print quick summary for development purposes
    print("\n=== DEV SUMMARY ===")
    print("This was a lightweight development run.")
    print("For production analysis, use the full example_workflow with:")
    print("- More MCMC samples (2000+ vs 200)")
    print("- More warmup samples (1000+ vs 100)")
    print("- More chains (4+ vs 2)")
    print("- Full parameter space exploration")


if __name__ == "__main__":
    main()