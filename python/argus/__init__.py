"""Argus: Bayesian inference for pulsar timing data analysis.

Argus is a Python package for performing Bayesian parameter estimation
on pulsar timing data using JAX and Kalman filtering techniques.
"""

__version__ = "0.0.0-dev"
__author__ = "Tom Kimpson, J.Hu"

# Import main modules
from . import (
    bayesian_inference,
    data_loader,
    gravitational_waves,
    io_manager,
    jax_kalman_filter,
    model,
    parameter_sampling,
    prior_models,
    utils,
    workflow,
)

# Expose key classes and functions at package level
from .utils import get_noise_parameters

__all__ = [
    # Main modules
    "bayesian_inference",
    "data_loader",
    "gravitational_waves",
    "io_manager",
    "jax_kalman_filter",
    "model",
    "parameter_sampling",
    "prior_models",
    "utils",
    "workflow",
    # Key functions
    "get_noise_parameters",
]
