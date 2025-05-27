"""Utility functions for the argus package."""

import os
import json
import logging
import jax
import jax.numpy as jnp
import pandas as pd
import configparser

def load_config(config_path):
    """Load configuration from file.
    
    Parameters
    ----------
    config_path : str
        Path to the configuration file
        
    Returns
    -------
    configparser.ConfigParser
        Configuration object containing all settings
        
    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def setup_logging(output_dir, config):
    """Set up logging configuration.
    
    Parameters
    ----------
    output_dir : str
        Directory where log files will be stored
    config : configparser.ConfigParser
        Configuration object containing logging settings
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    log_file = os.path.join(output_dir, 'inference.log')
    logging.basicConfig(
        level=getattr(logging, config.get('Logging', 'level')),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def check_gpu_availability():
    """Check for GPU availability and configure JAX accordingly.
    
    Returns
    -------
    bool
        True if GPU is available, False otherwise
    """
    try:
        gpu_devices = jax.devices('gpu')
        if gpu_devices:
            logging.info(f"Found GPU devices: {gpu_devices}")
            return True
        else:
            logging.warning("No GPU device found. Using CPU only.")
            return False
    except Exception as e:
        logging.error(f"Error checking for JAX GPU devices: {e}")
        return False

def get_efac_equad_injections(noise_params_path, excluded_psrs=[]):
    """Load EFAC and EQUAD values from noise parameters file.
    
    Parameters
    ----------
    noise_params_path : str
        Path to the noise parameters JSON file
    excluded_psrs : list of str, optional
        List of pulsar names to exclude from the analysis.
        Default is an empty list.
    
    Returns
    -------
    tuple
        Two JAX arrays containing EFAC and EQUAD values for the included pulsars.
    """
    with open(noise_params_path, "r") as f:
        noise_params = json.load(f)
    
    efac_values = []
    equad_values = []
    
    for psr in noise_params:
        if not any(excluded_psr in psr for excluded_psr in excluded_psrs):
            efac_values.append(noise_params[psr]["efac"])
            equad_values.append(10**noise_params[psr]["equad"])
    
    return jnp.array(efac_values), jnp.array(equad_values)

def get_psr_noise_injections(spin_injections_path, excluded_psrs=[]):
    """Load pulsar noise parameters from pickle file.
    
    Parameters
    ----------
    spin_injections_path : str
        Path to the spin injections pickle file
    excluded_psrs : list of str, optional
        List of pulsar names to exclude from the analysis.
        Default is an empty list.
    
    Returns
    -------
    tuple
        Two JAX arrays containing sigma_p and gamma_p values for the included pulsars.
    """
    df = pd.read_pickle(spin_injections_path)
    
    # Create a mask for pulsars to exclude
    exclude_mask = df['psr'].apply(lambda x: not any(excluded_psr in x for excluded_psr in excluded_psrs))
    df_filtered = df[exclude_mask]
    
    sigma_p_injected = df_filtered['optimal_sigma'].values
    gamma_p_injected = df_filtered['optimal_gamma'].values
    
    return jnp.array(sigma_p_injected), jnp.array(gamma_p_injected) 