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

# Remove conflicting argus installation from path
conflicting_path = '/fred/oz022/tkimpson/clean/Argus/python'
if conflicting_path in sys.path:
    sys.path.remove(conflicting_path)
    print(f"Removed conflicting path: {conflicting_path}")

# Add the parent directory to path to import argus modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from argus import data_loader
from argus import jax_kalman_filter
from argus import bayesian_inference
from argus import utils

from jaxns import Model, NestedSampler, TerminationCondition 


def setup_output_directory(config, use_gw, timestamp=None):
    """Setup output directory for the inference run.
    
    Args:
        config: Configuration object
        use_gw (bool): Whether to include gravitational wave model
        timestamp (str): Optional timestamp to use for output directory
    
    Returns:
        str: output_dir path
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
    
    print(f"Starting Bayesian inference {'with' if use_gw else 'without'} GW model...")
    
    return output_dir

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

def get_noise_parameters(config):
    """Get noise parameters from configuration and data files.
    
    Args:
        config: Configuration object
    
    Returns:
        tuple: (efac_array, equad_array, sigma_p_array, gamma_p_array)
    """
    noise_params_path = config.get('Data', 'noise_params_path')
    spin_injections_path = config.get('Data', 'spin_injections_path')
    excluded_psrs = config.get('Data', 'excluded_psrs').split(',')
    efac_array, equad_array = utils.get_efac_equad_injections(noise_params_path, excluded_psrs)
    sigma_p_array, gamma_p_array = utils.get_psr_noise_injections(spin_injections_path, excluded_psrs)
    
    return efac_array, equad_array, sigma_p_array, gamma_p_array

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
    
    # Check if we're using JAXNS (which requires the old model setup) or NumPyro
    method = config.get('Inference', 'method', fallback='jaxns').lower()
    
    if method == 'jaxns':
        # Get noise parameters
        efac_array, equad_array, sigma_p_array, gamma_p_array = get_noise_parameters(config)
        
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
    else:
        # For NumPyro, we don't need to setup the model here - it's handled in the inference function
        print("Using NumPyro inference - model setup will be handled during inference...")
        return None, None

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
    
    # Get noise parameters using the common function
    efac_array, equad_array, sigma_p_array, gamma_p_array = get_noise_parameters(config)
    
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

def save_numpyro_results(inf_data, output_dir, logger):
    """Save NumPyro/ArviZ results.
    
    Args:
        inf_data: ArviZ InferenceData object
        output_dir: Output directory path
        logger: Logger object
    
    Returns:
        str: Path to saved results file
    """
    from datetime import datetime
    
    logger.info("Saving NumPyro results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f'numpyro_results_{timestamp}.nc')
    
    # Save to NetCDF format
    inf_data.to_netcdf(results_path)
    logger.info(f"NumPyro results saved to {results_path}")
    
    return results_path

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
    
    # Create and save the plot
    plot_path = utils.corner_plot(results_path, output_dir)
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
        
        # Determine inference methods used
        gw_method = utils.get_inference_method_from_files(gw_output_dir)
        no_gw_method = utils.get_inference_method_from_files(no_gw_output_dir)
        
        if gw_method is None:
            logger.error(f"No inference results found in {gw_output_dir}")
            return None
            
        if no_gw_method is None:
            logger.error(f"No inference results found in {no_gw_output_dir}")
            return None
        
        # Only calculate Bayes factor for nested sampling (JAXNS) since it provides evidence
        if gw_method != 'jaxns' or no_gw_method != 'jaxns':
            logger.warning("Bayes factor calculation requires nested sampling (JAXNS) for both models")
            logger.warning(f"GW method: {gw_method}, no-GW method: {no_gw_method}")
            logger.warning("MCMC methods (NumPyro) do not directly provide model evidence")
            return None
        
        # Find and load JAXNS results
        gw_results_file = utils.find_results_file(gw_output_dir, 'nested_sampling_results_*.json')
        no_gw_results_file = utils.find_results_file(no_gw_output_dir, 'nested_sampling_results_*.json')
        
        if gw_results_file is None:
            logger.error(f"No JAXNS results file found in {gw_output_dir}")
            return None
            
        if no_gw_results_file is None:
            logger.error(f"No JAXNS results file found in {no_gw_output_dir}")
            return None
        
        logger.info(f"Loading GW results from: {gw_results_file}")
        logger.info(f"Loading no-GW results from: {no_gw_results_file}")
        
        # Load the results using utility functions
        gw_results = utils.load_jaxns_results(gw_results_file)
        no_gw_results = utils.load_jaxns_results(no_gw_results_file)
        
        # Extract log evidences
        log_Z_gw, log_Z_gw_uncert = utils.extract_log_evidence(gw_results, 'jaxns')
        log_Z_no_gw, log_Z_no_gw_uncert = utils.extract_log_evidence(no_gw_results, 'jaxns')
        
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
            "no_gw_results_file": no_gw_results_file,
            "inference_method": "jaxns"
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
    
    # Setup output directory
    output_dir = setup_output_directory(config, use_gw, timestamp)
    
    # Setup console logging only
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, config.get('Logging', 'level')))
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
    
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
    
    # Get inference method
    method = config.get('Inference', 'method', fallback='jaxns').lower()
    
    if method == 'jaxns' and jax_model is not None:
        # Sample from prior and evaluate likelihood for testing (JAXNS only)
        u = jax_model.sample_U(key=random.PRNGKey(432345987))
        θ = jax_model.transform(u)
        
        params = [θ[name] for name in param_names]
        log_likelihood = jax_model.log_likelihood(*params)
        logger.info("\nLog likelihood for parameters sampled from prior:")
        logger.info(str(log_likelihood))
        
        # Run nested sampling
        if config.getboolean('NestedSampling', 'run_sampling', fallback=True):
            termination_reason, state, ns = run_nested_sampling(config, jax_model, logger)
            
            # Save results
            results = save_results(ns, termination_reason, state, output_dir, logger)
        else:
            logger.info("Nested sampling is not being run")
    else:
        # For NumPyro, we need to get the noise parameters since they weren't loaded in setup_model
        efac_array, equad_array, sigma_p_array, gamma_p_array = get_noise_parameters(config)
        
        # Run inference using the new dispatcher function that handles both methods
        logger.info(f"Running {method.upper()} inference...")
        results = bayesian_inference.run_inference(
            KF, config, len(pulsar_data['metadata']), 
            sigma_p_array, gamma_p_array, efac_array, equad_array
        )
        
        # Save results based on method
        if method == 'numpyro':
            results_path = save_numpyro_results(results, output_dir, logger)
            
            # Create plots and diagnostics for NUTS
            logger.info("Creating corner plot and diagnostics for NUTS results...")
            
            # Create corner plot
            try:
                plot_path = utils.corner_plot(results_path, output_dir)
                if plot_path:
                    logger.info(f"Corner plot saved to {plot_path}")
                
            except Exception as e:
                logger.error(f"Error creating corner plot: {e}")
            
            # Run diagnostics
            try:
                logger.info("Running MCMC diagnostics...")
                utils.diagnostics(results_path)
                logger.info("MCMC diagnostics completed")
            except Exception as e:
                logger.error(f"Error running diagnostics: {e}")
                
        # JAXNS results are already handled above
    
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
    
    # Load config to check inference method
    config = utils.load_config(args.config)
    method = config.get('Inference', 'method', fallback='jaxns').lower()
    
    # Create a single timestamp for both runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run inference with GW
    print("\nRunning inference with GW model...")
    gw_output_dir = run_inference(config_path=args.config, use_gw=True, timestamp=timestamp)
    
    # For NUTS (NumPyro), only run GW model since it doesn't provide evidence for Bayes factors
    if method == 'numpyro':
        print("NUTS inference complete. Skipping no-GW run since NUTS doesn't provide model evidence.")
    else:
        # Run inference without GW (only for nested sampling methods)
        print("\nRunning inference without GW model...")
        no_gw_output_dir = run_inference(config_path=args.config, use_gw=False, timestamp=timestamp)
        
        # Calculate and save Bayes factor
        print("\nCalculating Bayes factor...")
        logger = logging.getLogger(__name__)
        calculate_and_save_bayes_factor(gw_output_dir, no_gw_output_dir, logger)
    