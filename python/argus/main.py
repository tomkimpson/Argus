"""Main script for running Bayesian inference on pulsar timing data using jaxns."""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.scipy.linalg import block_diag
from jax import random
import os
import sys
import json
from datetime import datetime
from flax import struct

# Add the parent directory to path to import argus modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from argus import data_loader
from argus import models
from argus import jax_kalman_filter
from argus import gravitational_waves
from argus import bayesian_inference

import tensorflow_probability.substrates.jax as tfp
tfpd = tfp.distributions

from jaxns import Prior, Model, NestedSampler, TerminationCondition

def run_inference(data_path, output_dir=None, seed=42):
    """
    Run Bayesian inference on pulsar timing data using nested sampling.
    
    Args:
        data_path (str): Path to the data directory
        output_dir (str, optional): Directory to save results. If None, creates a timestamped directory
        seed (int): Random seed for reproducibility
    
    Returns:
        dict: Inference results
    """
    print("Starting Bayesian inference...")
    
    # Create output directory if not provided
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"inference_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and process data
    print("Loading and processing data...")
    processed_pulsar_residuals, pulsar_metadata, pulsar_design_matrices, P_eps_matrices, hd_correlation_matrix = \
        data_loader.get_processed_residuals(data_path)
    
    # Set up the GW model
    print("Setting up GW model...")
    GW_model = models.StochasticGWBackgroundModel(
        pulsar_metadata, 
        hd_correlation_matrix, 
        pulsar_design_matrices
    )
    
    # Set up Kalman filter
    print("Initializing Kalman filter...")
    alpha = 1  # scale factor
    P0 = alpha * block_diag(*P_eps_matrices)
    
    KF = jax_kalman_filter.JaxKalmanFilter(
        model=GW_model,
        observations=processed_pulsar_residuals,
        Peps=P0
    )
    
    # Define likelihood function
    loglik_fn = lambda log10_ha, γa, log10_γp, log10_σp, efac, equad: \
        bayesian_inference.jaxns_log_likelihood(KF, log10_ha, γa, log10_γp, log10_σp, efac, equad)
    
    # Set up the model
    print("Setting up jaxns model...")
    jax_model = Model(
        prior_model=bayesian_inference.gw_prior_model,
        log_likelihood=loglik_fn
    )
    
    # Run nested sampling
    print("Running nested sampling...")
    ns = NestedSampler(
        model=jax_model,
        num_live_points=1000,
        max_samples=1e5,
        num_parallel_samplers=1,
        uncert_improvement_patience=3
    )
    
    termination_reason, state = ns(
        random.PRNGKey(seed),
        term_cond=TerminationCondition()
    )
    
    # Process results
    results = ns.to_results(state, termination_reason)
    
    # Save results
    print("Saving results...")
    results_dict = {
        'log_Z': float(results.log_Z),
        'log_Z_error': float(results.log_Z_error),
        'parameter_means': {k: float(v) for k, v in results.parameter_means.items()},
        'parameter_stds': {k: float(v) for k, v in results.parameter_stds.items()},
        'termination_reason': str(termination_reason)
    }
    
    with open(os.path.join(output_dir, 'inference_results.json'), 'w') as f:
        json.dump(results_dict, f, indent=4)
    
    print(f"Results saved to {output_dir}")
    return results_dict

if __name__ == "__main__":
    # Print JAX configuration
    print("=== JAX VERSION INFO ===")
    print(f"JAX version: {jax.__version__}")
    print("\n=== DEVICE INFO ===")
    print("Default device:", jax.default_backend())
    print("\n=== JAX CONFIG SETTINGS ===")
    for name, value in sorted(jax.config.values.items()):
        print(f"{name}: {value}")
    
    # Check GPU availability
    if any(d.platform == 'gpu' for d in jax.devices()):
        print("\nJAX GPU acceleration is AVAILABLE!")
        print("GPU devices:", [d for d in jax.devices() if d.platform == 'gpu'])
    else:
        print("\nJAX GPU acceleration is NOT available. Using CPU only.")
    
    # Example usage
    data_path = "../data/IPTA_MockDataChallenge2/dataset_2b/"
    results = run_inference(data_path)
    
    print("\nInference Results:")
    print(f"Log Evidence (Z): {results['log_Z']:.2f} ± {results['log_Z_error']:.2f}")
    print("\nParameter Estimates:")
    for param, (mean, std) in zip(results['parameter_means'].keys(), 
                                 zip(results['parameter_means'].values(), 
                                     results['parameter_stds'].values())):
        print(f"{param}: {mean:.3f} ± {std:.3f}")
