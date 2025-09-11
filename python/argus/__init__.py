"""Argus: Bayesian inference for pulsar timing data analysis.

Argus is a Python package for performing Bayesian parameter estimation
on pulsar timing data using JAX and Kalman filtering techniques.
"""

__version__ = "0.0.0-dev"
__author__ = "Tom Kimpson, J.Hu"

# Import main modules
from . import (
    analysis,
    bayesian_inference,
    cli,
    data_loader,
    gravitational_waves,
    inference_runners,
    io_manager,
    jax_kalman_filter,
    model,
    utils,
    workflow
)

# Expose key classes and functions at package level
from .workflow import get_noise_parameters
from .analysis import compare_inference_methods
from .data_loader import load_pulsar_data
from .bayesian_inference import run_bayesian_inference
from .utils import (
    setup_jax_config,
    get_datetime_string,
    save_dict_to_json
)

__all__ = [
    # Main modules
    "analysis",
    "bayesian_inference",
    "cli", 
    "data_loader",
    "gravitational_waves",
    "inference_runners",
    "io_manager",
    "jax_kalman_filter",
    "model", 
    "utils",
    "workflow",
    # Key functions
    "get_noise_parameters",
    "compare_inference_methods", 
    "load_pulsar_data",
    "run_bayesian_inference",
    "setup_jax_config",
    "get_datetime_string",
    "save_dict_to_json"
]
