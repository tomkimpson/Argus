"""Runner script for CW Bayesian inference."""

import os
import sys
import argparse
from datetime import datetime

import jax

# Add the python directory to path to import argus modules
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(os.path.join(project_root, "python"))

from argus import utils, workflow

jax.config.update("jax_enable_x64", True)


def main():
    """Run CW inference workflow."""
    parser = argparse.ArgumentParser(
        description="Run CW Bayesian inference on pulsar timing data."
    )
    parser.add_argument("config", type=str, help="Path to the configuration file")
    args = parser.parse_args()

    print("=== ARGUS CW WORKFLOW ===")
    print(f"JAX version: {jax.__version__}")
    print("Default device:", jax.default_backend())

    utils.check_gpu_availability()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = workflow.run_inference(
        config_path=args.config, use_gw=False, timestamp=timestamp
    )

    print(f"\nCW inference complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
