"""Example workflow script for Bayesian inference on pulsar timing array data."""

import os
import sys
import argparse
from datetime import datetime

import jax

# Add the python directory to path to import argus modules
# Go up to project root and then to python directory
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(os.path.join(project_root, "python"))

from argus import utils, workflow

jax.config.update("jax_enable_x64", True)


def main():
    """Run full example workflow for Bayesian inference."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run Bayesian inference workflow.")
    parser.add_argument("config", type=str, help="Path to the configuration file")

    args = parser.parse_args()

    # Print JAX configuration and system info
    print("=== EXAMPLE WORKFLOW ===")
    print(f"JAX version: {jax.__version__}")
    print("Default device:", jax.default_backend())
    print("Note: This is the full example workflow.")
    print("For rapid prototyping, use example_workflow_lite.\n")

    # Check GPU availability
    utils.check_gpu_availability()

    # Create a single timestamp for both runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run the inference workflow
    output_dir = workflow.run_inference(
        config_path=args.config, use_gw=True, timestamp=timestamp
    )

    print(f"\nInference complete! Results saved to: {output_dir}")

    # Print quick summary
    print("\n=== SUMMARY ===")
    print("This was a full example workflow run.")
    print("For faster prototyping, use example_workflow_lite with:")
    print("- Fewer MCMC samples (200 vs 2000+)")
    print("- Fewer warmup samples (100 vs 1000+)")
    print("- Fewer chains (2 vs 4+)")
    print("- Reduced parameter space exploration")


if __name__ == "__main__":
    main()
