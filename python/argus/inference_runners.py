"""Inference execution and performance testing utilities for the argus package."""

import time
from jax import random
import jax

from argus import bayesian_inference, utils



def run_numpyro_inference(config, KF, pulsar_data, output_dir, output_id, logger):
    """Run NumPyro NUTS inference pipeline.
    
    Args:
        config: Configuration object
        KF: Kalman filter object
        pulsar_data: Processed pulsar data
        output_dir: Output directory path
        output_id: Output identifier
        logger: Logger object
        
    Returns
    -------
        str: Path to saved results file
    """
    # Get noise parameters
    from argus.workflow import get_noise_parameters
    efac_array, equad_array, sigma_p_array, gamma_p_array = get_noise_parameters(config)
    
    # Get prior model specifications and display them
    n_pulsars = len(pulsar_data['metadata'])
    prior_specs = bayesian_inference.get_prior_model_specs(
        config, n_pulsars, sigma_p_array, gamma_p_array, efac_array, equad_array
    )
    
    # Display prior summary
    bayesian_inference.display_prior_summary(prior_specs, n_pulsars, logger)
    
    # Test likelihood performance
    logger.info("Performing likelihood performance test...")
    bayesian_inference.test_likelihood_performance(KF, config, logger)
    
    # Run inference using the renamed function
    logger.info("Running NUMPYRO inference...")
    results = bayesian_inference.run_nuts_sampling(
        KF, config, len(pulsar_data['metadata']), 
        sigma_p_array, gamma_p_array, efac_array, equad_array
    )
    
    # Save results
    from argus.io_manager import save_numpyro_results
    results_path = save_numpyro_results(results, output_dir, output_id, logger)
    
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
        utils.diagnostics(results_path, output_dir)
        logger.info("MCMC diagnostics completed")
    except Exception as e:
        logger.error(f"Error running diagnostics: {e}")
    
    return results_path