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
import matplotlib.pyplot as plt

# Add the parent directory to path to import argus modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from argus import data_loader
from argus import jax_kalman_filter
from argus import bayesian_inference
from argus import utils

from jaxns import Model, NestedSampler, TerminationCondition
from jaxns import load_results 


def setup_output_directory(config, use_gw, timestamp=None):
    """Setup output directory and logging for the inference run.
    
    Args:
        config: Configuration object
        use_gw (bool): Whether to include gravitational wave model
        timestamp (str): Optional timestamp to use for output directory
    
    Returns:
        tuple: (output_dir, logger)
    """
    # Get output_id from config file
    output_id = config.get('Output', 'output_id', fallback='').strip()
    
    # Determine directory name: use ID if provided, otherwise use timestamp
    if output_id:
        dir_name = output_id
        base_dir = config.get('Output', 'base_dir').format(output_id=output_id)
    else:
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = timestamp
        base_dir = config.get('Output', 'base_dir').format(timestamp=timestamp)
    
    # Create base output directory
    base_output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', base_dir)
    
    if not use_gw:
        # For no-GW runs, nest under the GW directory
        # First ensure the parent (GW) directory exists
        os.makedirs(base_output_dir, exist_ok=True)
        output_dir = os.path.join(base_output_dir, "no_gw")
    else:
        # For GW runs, use the base directory
        output_dir = base_output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging directly in output directory (no logfiles subdirectory)
    logger = utils.setup_logging(output_dir, config)
    logger.info(f"Starting Bayesian inference {'with' if use_gw else 'without'} GW model...")
    
    return output_dir, logger

def setup_data_and_kalman_filter(config, logger, use_gw):
    """Load and process data, initialize Kalman filter.
    
    Args:
        config: Configuration object
        logger: Logger object
        use_gw (bool): Whether to include gravitational wave model
    
    Returns:
        tuple: (pulsar_data, KF)
    """
    logger.info("Loading and processing data...")
    data_path = config.get('Data', 'data_path')
    excluded_psrs = config.get('Data', 'excluded_psrs').split(',')
    pulsar_data = data_loader.LoadWidebandPulsarData.get_processed_residuals(
        data_path,
        excluded_psrs=[psr.strip() for psr in excluded_psrs if psr.strip()],
    )
    
    logger.info("Initializing Kalman filter...")
    KF = jax_kalman_filter.JaxKalmanFilter(data=pulsar_data, use_gw=use_gw)
    
    return pulsar_data, KF

def setup_model(config, KF, pulsar_data):
    """Setup the jaxns model with priors and likelihood.
    
    Args:
        config: Configuration object
        KF: Kalman filter object
        pulsar_data: Processed pulsar data
    
    Returns:
        tuple: (jax_model, param_names)
    """
    Npsr = int(len(pulsar_data['metadata'])) 
    print("The number of pulsars is:")
    print(Npsr)
    
    # Get noise parameters
    noise_params_path = config.get('Data', 'noise_params_path')
    spin_injections_path = config.get('Data', 'spin_injections_path')
    excluded_psrs = config.get('Data', 'excluded_psrs').split(',')
    efac_array, equad_array = utils.get_efac_equad_injections(noise_params_path, excluded_psrs)
    sigma_p_array, gamma_p_array = utils.get_psr_noise_injections(spin_injections_path, excluded_psrs)
    
    # Get prior model specifications
    prior_specs = bayesian_inference.get_prior_model_specs(
        config, Npsr, sigma_p_array, gamma_p_array, efac_array, equad_array
    )

    # Set up the prior model
    print("Setting up the prior model...")
    prior_model = lambda: bayesian_inference.configurable_prior_model(
        Npsr=Npsr,
        **prior_specs
    )

    # Set up the log likelihood function
    print("Setting up the log likelihood function...")
    loglik_fn = lambda log10_ha, γa, log10_γp, log10_σp, efac, equad: \
        bayesian_inference.jaxns_log_likelihood(KF, log10_ha, γa, log10_γp, log10_σp, efac, equad)
    
    print("Setting up the jax model...")
    print("The prior model is:")
    print(prior_model)
    print("The log likelihood function is:")
    print(loglik_fn)
    jax_model = Model(prior_model=prior_model, log_likelihood=loglik_fn)
    
    # Define parameter names for reference
    param_names = ['log10_ha', 'γa', 'log10_γp', 'log10_σp', 'efac', 'equad']
    
    return jax_model, param_names

def test_likelihood_performance(KF, config, logger):
    """Test likelihood evaluation performance using known parameter values.
    
    This function runs a single likelihood evaluation using the same parameter
    values as in test_likelihood_value to provide users with timing and
    likelihood value information before running the full nested sampling.
    
    Args:
        KF: Kalman filter object
        config: Configuration object
        logger: Logger object
        
    Returns:
        float: The computed log likelihood value
    """
    logger.info("=== Likelihood Performance Test ===")
    logger.info("Testing likelihood evaluation with known parameter values...")
    
    # Get the same parameter values used in test_likelihood_value
    noise_params_path = config.get('Data', 'noise_params_path')
    spin_injections_path = config.get('Data', 'spin_injections_path')
    excluded_psrs = config.get('Data', 'excluded_psrs').split(',')
    
    # Get noise parameters (same as test)
    efac_array, equad_array = utils.get_efac_equad_injections(noise_params_path, excluded_psrs)
    sigma_p_array, gamma_p_array = utils.get_psr_noise_injections(spin_injections_path, excluded_psrs)
    
    # Set test parameter values (same as test_likelihood_value)
    γa_test = 1e-9 
    ha_test = 1e-15
    

    # Create parameter object
    test_params = bayesian_inference.Parameters(
        γa=γa_test,
        ha=ha_test,
        γp=gamma_p_array,
        σp=sigma_p_array,
        EFAC=efac_array,
        EQUAD=equad_array
    )
    
    logger.info(f"Test parameters: γa={γa_test}, ha={ha_test}")
    logger.info(f"Number of pulsars: {len(gamma_p_array)}")
    
    # Time the likelihood evaluation
    logger.info("Performing for the first time a likelihood evaluation...")
    start_time = time.perf_counter()
    
    log_likelihood = KF.get_likelihood(test_params)
    # Ensure computation is complete before stopping timer
    log_likelihood.block_until_ready()
    
    end_time = time.perf_counter()
    duration1 = end_time - start_time


    # Time the likelihood evaluation
    logger.info("Performing timed for the second time a likelihood evaluation...")
    start_time = time.perf_counter()
    
    log_likelihood = KF.get_likelihood(test_params)
    # Ensure computation is complete before stopping timer
    log_likelihood.block_until_ready()
    
    end_time = time.perf_counter()
    duration2 = end_time - start_time



    
    # Log results
    logger.info(f"Likelihood evaluation completed in {duration1:.4f} seconds the first time")
    logger.info(f"Likelihood evaluation completed in {duration2:.4f} seconds the second time")
    logger.info(f"Log likelihood value: {float(log_likelihood)}")
    logger.info("=== End Likelihood Performance Test ===")
    
    return float(log_likelihood)

def run_nested_sampling(config, jax_model, logger):
    """Run the nested sampling algorithm.
    
    Args:
        config: Configuration object
        jax_model: JAX model object
        logger: Logger object
    
    Returns:
        tuple: (termination_reason, state, ns)
    """
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
    
    return termination_reason, state, ns

def save_results(ns, termination_reason, state, output_dir, logger):
    """Save the nested sampling results.
    
    Args:
        ns: NestedSampler object
        termination_reason: Termination reason from sampling
        state: Final state from sampling
        output_dir: Output directory path
        logger: Logger object
    
    Returns:
        dict: Results dictionary
    """
    logger.info("Converting results...")
    results = ns.to_results(termination_reason=termination_reason, state=state)
    
    # Save results
    logger.info("Saving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f'nested_sampling_results_{timestamp}.json')
    ns.save_results(results, results_path)
    logger.info(f"Results saved to {results_path}")
    
    # Load results and create corner plot of just ha
    logger.info("Loading results and creating corner plot...")
    loaded_results = load_results(results_path)
    
    # Define parameters to plot and their ranges
    parameters = ['log10_ha']
    ranges = [[-17.0, -14.0]]  # log10_ha range
    
    # Create and save the plot
    plot_path = utils.plot_jaxns_corner(loaded_results, parameters, ranges, output_dir)
    if plot_path:
        logger.info(f"Corner plot saved to {plot_path}")
    
    return results

def calculate_and_save_bayes_factor(gw_output_dir, no_gw_output_dir, logger):
    """Calculate and save the Bayes factor between GW and no-GW models.
    
    Args:
        gw_output_dir (str): Directory containing GW model results
        no_gw_output_dir (str): Directory containing no-GW model results
        logger: Logger object
    
    Returns:
        dict: Dictionary containing log evidences and Bayes factor, or None if calculation failed
    """
    try:
        logger.info("=== Calculating Bayes Factor ===")
        
        # Find the result files in each directory
        gw_results_file = None
        no_gw_results_file = None
        
        # Search for nested sampling results files
        for file in os.listdir(gw_output_dir):
            if file.startswith('nested_sampling_results_') and file.endswith('.json'):
                gw_results_file = os.path.join(gw_output_dir, file)
                break
        
        for file in os.listdir(no_gw_output_dir):
            if file.startswith('nested_sampling_results_') and file.endswith('.json'):
                no_gw_results_file = os.path.join(no_gw_output_dir, file)
                break
        
        if gw_results_file is None:
            logger.error(f"No GW results file found in {gw_output_dir}")
            return None
            
        if no_gw_results_file is None:
            logger.error(f"No no-GW results file found in {no_gw_output_dir}")
            return None
        
        logger.info(f"Loading GW results from: {gw_results_file}")
        logger.info(f"Loading no-GW results from: {no_gw_results_file}")
        
        # Load the results
        gw_results = load_results(gw_results_file)
        no_gw_results = load_results(no_gw_results_file)
        
        # Extract log evidences
        log_Z_gw = float(gw_results.log_Z_mean)
        log_Z_gw_uncert = float(gw_results.log_Z_uncert)
        log_Z_no_gw = float(no_gw_results.log_Z_mean)
        log_Z_no_gw_uncert = float(no_gw_results.log_Z_uncert)
        
        # Calculate log Bayes factor (GW vs no-GW)
        log_bayes_factor = log_Z_gw - log_Z_no_gw
        
        # Calculate uncertainty in Bayes factor (assuming independent uncertainties)
        log_bayes_factor_uncert = (log_Z_gw_uncert**2 + log_Z_no_gw_uncert**2)**0.5
        
        logger.info(f"Log evidence (GW model): {log_Z_gw:.3f} ± {log_Z_gw_uncert:.3f}")
        logger.info(f"Log evidence (no-GW model): {log_Z_no_gw:.3f} ± {log_Z_no_gw_uncert:.3f}")
        logger.info(f"Log Bayes factor (GW vs no-GW): {log_bayes_factor:.3f} ± {log_bayes_factor_uncert:.3f}")
        
        # Create results dictionary
        bayes_factor_results = {
            "log_evidence_gw": log_Z_gw,
            "log_evidence_gw_uncertainty": log_Z_gw_uncert,
            "log_evidence_no_gw": log_Z_no_gw,
            "log_evidence_no_gw_uncertainty": log_Z_no_gw_uncert,
            "log_bayes_factor": log_bayes_factor,
            "log_bayes_factor_uncertainty": log_bayes_factor_uncert,
            "bayes_factor": float(jnp.exp(log_bayes_factor)),
            "calculation_timestamp": datetime.now().isoformat(),
            "gw_results_file": gw_results_file,
            "no_gw_results_file": no_gw_results_file
        }
        
        # Save results to the main GW directory
        bayes_factor_file = os.path.join(gw_output_dir, 'bayes_factor_results.json')
        with open(bayes_factor_file, 'w') as f:
            json.dump(bayes_factor_results, f, indent=2)
        
        logger.info(f"Bayes factor results saved to: {bayes_factor_file}")
        logger.info("=== Bayes Factor Calculation Complete ===")
        
        return bayes_factor_results
        
    except Exception as e:
        logger.error(f"Error calculating Bayes factor: {e}")
        logger.error("Bayes factor calculation failed")
        return None

def run_inference(config_path, use_gw=True, timestamp=None):
    """
    Run Bayesian inference on pulsar timing data using nested sampling.
    
    Args:
        config_path (str): Path to configuration file
        use_gw (bool): Whether to include gravitational wave model
        timestamp (str): Optional timestamp to use for output directory
    
    Returns:
        str: Output directory path
    """
    
    # Load configuration
    config = utils.load_config(config_path)
    
    # Setup output directory and logging
    output_dir, logger = setup_output_directory(config, use_gw, timestamp)
    
    # Copy config file to output directory
    config_filename = os.path.basename(config_path)
    output_config_path = os.path.join(output_dir, config_filename)
    shutil.copy2(config_path, output_config_path)
    logger.info(f"Copied config file to {output_config_path}")
    
    # Setup data and Kalman filter
    pulsar_data, KF = setup_data_and_kalman_filter(config, logger, use_gw)
    
    # Setup model
    jax_model, param_names = setup_model(config, KF, pulsar_data)
    
    # Test likelihood performance with known parameters
    test_likelihood_performance(KF, config, logger)
    
    # Sample from prior and evaluate likelihood for testing
    u = jax_model.sample_U(key=random.PRNGKey(432345987))
    θ = jax_model.transform(u)

    
    params = [θ[name] for name in param_names]
    log_likelihood = jax_model.log_likelihood(*params)
    logger.info("\nLog likelihood for parameters sampled from priorv:")
    logger.info(str(log_likelihood))
    
    # Run nested sampling
    if config.getboolean('NestedSampling', 'run_sampling', fallback=True):
        termination_reason, state, ns = run_nested_sampling(config, jax_model, logger)
        
        # Save results
        results = save_results(ns, termination_reason, state, output_dir, logger)
    else:
        logger.info("Nested sampling is not being run")
    
    return output_dir


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

    print("You are working with the development version of the code. Good job.")
    
    # Check GPU availability
    has_gpu = utils.check_gpu_availability()
    
    # Create a single timestamp for both runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run inference with GW
    print("\nRunning inference with GW model...")
    gw_output_dir = run_inference(config_path=args.config, use_gw=True, timestamp=timestamp)
    
    # Run inference without GW
    print("\nRunning inference without GW model...")
    no_gw_output_dir = run_inference(config_path=args.config, use_gw=False, timestamp=timestamp)
    
    # Calculate and save Bayes factor
    print("\nCalculating Bayes factor...")
    config = utils.load_config(args.config)
    logger = utils.setup_logging(gw_output_dir, config)
    calculate_and_save_bayes_factor(gw_output_dir, no_gw_output_dir, logger)
    