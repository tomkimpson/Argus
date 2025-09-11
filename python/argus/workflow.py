"""Workflow orchestration and high-level functions for the argus package."""

import logging

from argus import data_loader, jax_kalman_filter, bayesian_inference, utils
from argus import io_manager




def setup_data_and_kalman_filter(config, logger, use_gw):
    """Load and process data, initialize Kalman filter.
    
    Args:
        config: Configuration object
        logger: Logger object
        use_gw (bool): Whether to include gravitational wave model
    
    Returns
    -------
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




def run_inference(config_path, use_gw=True, timestamp=None):
    """
    Run Bayesian inference on pulsar timing data.
    
    Args:
        config_path (str): Path to configuration file
        use_gw (bool): Whether to include gravitational wave model
        timestamp (str): Optional timestamp to use for output directory
    
    Returns
    -------
        str: Output directory path
    """
    # Load configuration and resolve any relative paths
    config = utils.load_config(config_path)
    config = utils.resolve_config_paths(config, config_path)
    
    # Get output_id from config
    output_id = io_manager.get_output_id_from_config(config, timestamp)
    
    # Setup output directory
    output_dir = io_manager.setup_output_directory(config, use_gw, timestamp)
    
    # Setup centralized logging with both console and file output
    logger = io_manager.setup_single_logger(config, output_dir, enable_file_logging=True)
    
    # Copy config file to output directory
    io_manager.copy_config_file(config_path, output_dir, logger)
    
    # Setup data and Kalman filter
    pulsar_data, KF = setup_data_and_kalman_filter(config, logger, use_gw)
    
    # Get noise parameters
    efac_array, equad_array, sigma_p_array, gamma_p_array = utils.get_noise_parameters(config)
    
    # Get prior model specifications and display them
    n_pulsars = len(pulsar_data['metadata'])
    prior_specs = bayesian_inference.get_prior_model_specs(
        config, n_pulsars, sigma_p_array, gamma_p_array, efac_array, equad_array
    )
    
    # Display prior summary
    bayesian_inference.display_prior_summary(prior_specs, n_pulsars, logger)
    
    # Test likelihood performance with known parameters
    logger.info("Performing likelihood performance test...")
    bayesian_inference.test_likelihood_performance(KF, config, logger)
    
    # Get inference method
    method = config.get('Inference', 'method', fallback='numpyro').lower()
    
    if method == 'jaxns':
        raise ValueError("JAXNS nested sampling is no longer supported. Please use 'numpyro' for NUTS sampling.")
        
    elif method == 'numpyro':
        # Run NumPyro NUTS inference
        logger.info("Running NUMPYRO inference...")
        results = bayesian_inference.run_nuts_sampling(
            KF, config, len(pulsar_data['metadata']), 
            sigma_p_array, gamma_p_array, efac_array, equad_array
        )
        
        # Save results
        results_path = io_manager.save_numpyro_results(results, output_dir, output_id, logger)
        
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
        
    else:
        raise ValueError(f"Unknown inference method: {method}")
    
    return output_dir


