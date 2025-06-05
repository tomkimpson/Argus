"""Workflow orchestration and high-level functions for the argus package."""

import logging

from argus import data_loader, jax_kalman_filter, bayesian_inference, utils
from argus import io_manager, inference_runners

from jaxns import Model


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


def setup_jaxns_model(config, KF, pulsar_data):
    """Set up the JAXNS model with priors and likelihood.
    
    Args:
        config: Configuration object
        KF: Kalman filter object
        pulsar_data: Processed pulsar data
    
    Returns
    -------
        tuple: (jax_model, param_names)
    """
    Npsr = int(len(pulsar_data['metadata'])) 
    print("The number of pulsars is:")
    print(Npsr)
    
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
    # Load configuration
    config = utils.load_config(config_path)
    
    # Get output_id from config
    output_id = io_manager.get_output_id_from_config(config, timestamp)
    
    # Setup output directory
    output_dir = io_manager.setup_output_directory(config, use_gw, timestamp)
    
    # Setup console logging only
    logger = io_manager.setup_console_logging(config)
    
    # Copy config file to output directory
    io_manager.copy_config_file(config_path, output_dir, logger)
    
    # Setup data and Kalman filter
    pulsar_data, KF = setup_data_and_kalman_filter(config, logger, use_gw)
    
    # Test likelihood performance with known parameters
    inference_runners.test_likelihood_performance(KF, config, logger)
    
    # Get inference method
    method = config.get('Inference', 'method', fallback='jaxns').lower()
    
    if method == 'jaxns':
        # Setup JAXNS model
        jax_model, param_names = setup_jaxns_model(config, KF, pulsar_data)
        
        # Run JAXNS inference
        inference_runners.run_jaxns_inference(
            config, jax_model, param_names, output_dir, output_id, logger
        )
        
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
        tuple: (gw_output_dir, no_gw_output_dir, bayes_factor_results)
    """
    # Load config to check if comparison is appropriate
    config = utils.load_config(config_path)
    method = config.get('Inference', 'method', fallback='jaxns').lower()
    
    # Run inference with GW
    print("\nRunning inference with GW model...")
    gw_output_dir = run_inference(config_path=config_path, use_gw=True, timestamp=timestamp)
    
    if method == 'numpyro':
        print("NUTS inference complete. Skipping no-GW run since NUTS doesn't provide model evidence.")
        return gw_output_dir, None, None
    else:
        # Run inference without GW (only for nested sampling methods)
        print("\nRunning inference without GW model...")
        no_gw_output_dir = run_inference(config_path=config_path, use_gw=False, timestamp=timestamp)
        
        # Calculate and save Bayes factor
        print("\nCalculating Bayes factor...")
        from argus.analysis import calculate_and_save_bayes_factor
        logger = logging.getLogger(__name__)
        bayes_factor_results = calculate_and_save_bayes_factor(gw_output_dir, no_gw_output_dir, logger)
        
        return gw_output_dir, no_gw_output_dir, bayes_factor_results