"""NG15 SGWB workflow script for Bayesian inference on pulsar timing array data.

Cloned from workflows/example_workflow/run_analysis.py. Keeps use_gw=True (GWB
path) and jax_enable_x64. The project-root path walk resolves the repo root and
appends python/ to sys.path unchanged, since this file sits at the same depth
(repo/workflows/<name>/run_analysis.py).
"""

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
    """Run the NG15 SGWB workflow for Bayesian inference."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run Bayesian inference workflow.")
    parser.add_argument("config", type=str, help="Path to the configuration file")

    args = parser.parse_args()

    # Print JAX configuration and system info
    print("=== NG15 SGWB WORKFLOW ===")
    print(f"JAX version: {jax.__version__}")
    print("Default device:", jax.default_backend())
    print("Note: SGWB (GWB+HD+NUTS) recovery on NG15 wideband data.\n")

    # Check GPU availability
    utils.check_gpu_availability()

    # Create a single timestamp for both runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run the inference workflow
    output_dir = workflow.run_inference(
        config_path=args.config, use_gw=True, timestamp=timestamp
    )

    print(f"\nInference complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
