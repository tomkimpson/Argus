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
import time
import logging
import argparse
import shutil

# Add the parent directory to path to import argus modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from argus import data_loader
from argus import jax_kalman_filter
from argus import bayesian_inference
from argus import utils

from jaxns import Model, NestedSampler, TerminationCondition

def run_inference(config_path):
    """
    Run Bayesian inference on pulsar timing data using nested sampling.
    
    Args:
        config_path (str): Path to configuration file
    
    Returns:
        dict: Inference results
    """
    start_time = time.time()
    
    # Load configuration
    config = utils.load_config(config_path)
    
    # Get data path from config
    data_path = config.get('Data', 'data_path')
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = config.get('Output', 'base_dir').format(timestamp=timestamp)
    # Change output directory to be in python/argus/outputs/
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', base_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    logger = utils.setup_logging(output_dir, config)
    logger.info("Starting Bayesian inference...")

    # Copy config file to output directory
    config_filename = os.path.basename(config_path)
    output_config_path = os.path.join(output_dir, config_filename)
    shutil.copy2(config_path, output_config_path)
    logger.info(f"Copied config file to {output_config_path}")
    
    # Load and process data
    logger.info("Loading and processing data...")
    excluded_psrs = config.get('Data', 'excluded_psrs').split(',')
    pulsar_data = data_loader.LoadWidebandPulsarData.get_processed_residuals(
        data_path,
        excluded_psrs=[psr.strip() for psr in excluded_psrs if psr.strip()],
    )
    
    # Initialize Kalman filter
    logger.info("Initializing Kalman filter...")
    KF = jax_kalman_filter.JaxKalmanFilter(data=pulsar_data, use_gw=True)
    
    #Set up the jaxns model
    Npsr = len(pulsar_data)  # Get number of pulsars from data

    # Get EFAC and EQUAD values
    noise_params_path = config.get('Data', 'noise_params_path')
    spin_injections_path = config.get('Data', 'spin_injections_path')
    efac_array, equad_array = utils.get_efac_equad_injections(noise_params_path, excluded_psrs)
    sigma_p_array, gamma_p_array = utils.get_psr_noise_injections(spin_injections_path, excluded_psrs)
    
    # Get prior model specifications
    prior_specs = bayesian_inference.get_prior_model_specs(config, Npsr, sigma_p_array, gamma_p_array, efac_array, equad_array)

    # Set up the prior model with configurable parameters using a lambda function
    prior_model = lambda: bayesian_inference.configurable_prior_model(
        Npsr=Npsr,
        **prior_specs
    )

    # Set up the log likelihood function using a lambda function
    loglik_fn = lambda log10_ha, γa, log10_γp, log10_σp, efac, equad: \
        bayesian_inference.jaxns_log_likelihood(KF, log10_ha, γa, log10_γp, log10_σp, efac, equad)
    
    jax_model = Model(prior_model=prior_model, log_likelihood=loglik_fn)

    # Sample from the prior model and evaluate the log likelihood, just to check everything is working ok.
    u = jax_model.sample_U(key=random.PRNGKey(432345987))  # Unit cube sample
    θ = jax_model.transform(u)                       # Transform to physical parameter space

    print("The sampled parameters are:")
    print(θ)

    # Define the expected parameter order for loglik_fn
    param_names = ['log10_ha', 'γa', 'log10_γp', 'log10_σp', 'efac', 'equad']
    
    # Extract parameters in the correct order and convert to float
    params = [θ[name] for name in param_names]
    
    # Evaluate log likelihood with the parameters
    log_likelihood = loglik_fn(*params)
    print("\nLog likelihood value:")
    print(log_likelihood)

    # Initialize and run nested sampling
    logger.info("Initializing nested sampling...")
    ns = NestedSampler(
        model=jax_model,
        num_live_points=config.getint('NestedSampling', 'num_live_points', fallback=100),
        verbose=True
    )

    logger.info("Running nested sampling...")
    term_cond = TerminationCondition(
        dlogZ=config.getfloat('NestedSampling', 'dlogZ', fallback=0.1)
    )
    termination_reason, state = jax.jit(ns)(
        key=random.PRNGKey(432345987),
        term_cond=term_cond
    )

    logger.info("Converting results...")
    results = ns.to_results(termination_reason=termination_reason, state=state)

   
    # Save results
    logger.info("Saving results...")
    results_path = os.path.join(output_dir, f'nested_sampling_results_{timestamp}.json')
    ns.save_results(results, results_path)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run Bayesian inference on pulsar timing data.')
    parser.add_argument('config', type=str, help='Path to the configuration file')
    
    args = parser.parse_args()
    
    # Print JAX configuration
    print("=== JAX VERSION INFO ===")
    print(f"JAX version: {jax.__version__}")
    print("\n=== DEVICE INFO ===")
    print("Default device:", jax.default_backend())
    
    # Check GPU availability
    has_gpu = utils.check_gpu_availability()
    
    # Run inference
    results = run_inference(config_path=args.config)
    