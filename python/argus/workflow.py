"""Workflow orchestration and high-level functions for the argus package."""

import logging

from argus import data_loader, jax_kalman_filter, bayesian_inference, utils
from argus import io_manager, inference_runners



def get_noise_parameters(config):
    """Get injected noise parameters from configuration and data files.
    
    Args:
        config: Configuration object
    
    Returns
    -------
        tuple: (efac_array, equad_array, sigma_p_array, gamma_p_array)
    """
    noise_params_path = config.get('Data', 'noise_params_path')
    spin_injections_path = config.get('Data', 'spin_injections_path')
    excluded_psrs = config.get('Data', 'excluded_psrs').split(',')
    efac_array, equad_array = utils.get_efac_equad_injections(noise_params_path, excluded_psrs)
    sigma_p_array, gamma_p_array = utils.get_psr_noise_injections(spin_injections_path, excluded_psrs)
    
    return efac_array, equad_array, sigma_p_array, gamma_p_array


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
    
    # Test likelihood performance with known parameters
    inference_runners.test_likelihood_performance(KF, config, logger)
    
    # Get inference method
    method = config.get('Inference', 'method', fallback='numpyro').lower()
    
    if method == 'jaxns':
        raise ValueError("JAXNS nested sampling is no longer supported. Please use 'numpyro' for NUTS sampling.")
        
    elif method == 'numpyro':
        # Run NumPyro inference
        inference_runners.run_numpyro_inference(
            config, KF, pulsar_data, output_dir, output_id, logger
        )
        
    else:
        raise ValueError(f"Unknown inference method: {method}")
    
    return output_dir


def run_model_comparison(config_path, timestamp=None):
    """Run both GW and no-GW models and compare them.
    
    Args:
        config_path (str): Path to configuration file
        timestamp (str): Optional timestamp to use for output directory
        
    Returns
    -------
        tuple: (gw_output_dir, no_gw_output_dir, None)
    """
    # Load config to check if comparison is appropriate
    config = utils.load_config(config_path)
    config = utils.resolve_config_paths(config, config_path)
    method = config.get('Inference', 'method', fallback='numpyro').lower()
    
    # Run inference with GW
    print("\nRunning inference with GW model...")
    gw_output_dir = run_inference(config_path=config_path, use_gw=True, timestamp=timestamp)
    
    print("NUTS inference complete. Model comparison not available with NUTS sampling.")
    print("Note: Model evidence comparison requires nested sampling methods, which are no longer supported.")
    return gw_output_dir, None, None