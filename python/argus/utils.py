"""Utility functions for the argus package."""

import os
import json
import logging
import jax
import jax.numpy as jnp
import pandas as pd
import configparser
from datetime import datetime
import glob

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
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f'nested_sampling_test_output_{timestamp}.txt')
    
    # Configure logging to both file and console
    logging.basicConfig(
        level=getattr(logging, config.get('Logging', 'level')),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
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






from jaxns import load_results 
import numpy as np 
import corner
import matplotlib.pyplot as plt

def plot_jaxns_corner(results, parameters, ranges, output_dir=None):
    """Create and save a corner plot of the nested sampling results.
    
    Parameters
    ----------
    results : jaxns.Results
        Results from nested sampling
    parameters : list of str
        List of parameter names to plot
    ranges : list of list
        List of [min, max] ranges for each parameter
    output_dir : str, optional
        Directory to save the plot. If None, plot is shown but not saved.
    """
    # Get samples for each selected parameter
    samples = []
    for param in parameters:
        if param not in results.samples:
            raise ValueError(f"Parameter {param} not found in data")
        samples.append(results.samples[param].flatten())

    samples = np.column_stack(samples)

    evidence = results.log_Z_mean.item()
    evidence_std = results.log_Z_uncert.item()

    # Create corner plot
    fig = corner.corner(
        samples,
        labels=parameters,
        color='C0',
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        range=ranges,
        bins=30,
        smooth=1.0,
        smooth1d=1.0,
        plot_datapoints=True,
        fill_contours=True,
        levels=[0.68, 0.95]  # 1 and 2 sigma contours
    )

    # Add evidence information as figure title
    plt.suptitle(f"log(Z) = {evidence:.2f} ± {evidence_std:.2f}", y=1.02, fontsize=14)
    plt.tight_layout()
    
    if output_dir is not None:
        # Create plots directory if it doesn't exist
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(plots_dir, f'corner_plot_{timestamp}.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        return plot_path
    else:
        plt.show()
        return None


def find_results_file(output_dir, file_pattern):
    """Find results file in output directory matching pattern.
    
    Parameters
    ----------
    output_dir : str
        Directory to search for results files
    file_pattern : str
        Glob pattern to match files (e.g., 'nested_sampling_results_*.json')
    
    Returns
    -------
    str or None
        Path to the first matching results file, or None if not found
    """
    pattern_path = os.path.join(output_dir, file_pattern)
    matching_files = glob.glob(pattern_path)
    
    if matching_files:
        # Return the most recent file if multiple matches
        return max(matching_files, key=os.path.getmtime)
    return None


def load_jaxns_results(results_file):
    """Load JAXNS results from JSON file.
    
    Parameters
    ----------
    results_file : str
        Path to JAXNS results JSON file
    
    Returns
    -------
    object
        JAXNS results object
    """
    from jaxns import load_results
    return load_results(results_file)


def load_numpyro_results(results_file):
    """Load NumPyro results from NetCDF file.
    
    Parameters
    ----------
    results_file : str
        Path to NumPyro results NetCDF file
    
    Returns
    -------
    arviz.InferenceData
        ArviZ InferenceData object
    """
    import arviz as az
    return az.from_netcdf(results_file)


def extract_log_evidence(results, method):
    """Extract log evidence from results object.
    
    Parameters
    ----------
    results : object
        Results object (JAXNS or ArviZ)
    method : str
        Inference method ('jaxns' or 'numpyro')
    
    Returns
    -------
    tuple
        (log_evidence, log_evidence_uncertainty)
    """
    if method.lower() == 'jaxns':
        return float(results.log_Z_mean), float(results.log_Z_uncert)
    elif method.lower() == 'numpyro':
        # For NumPyro, we need to estimate evidence using importance sampling or other methods
        # This is a placeholder - proper evidence estimation for MCMC is non-trivial
        # For now, return None to indicate evidence not available
        return None, None
    else:
        raise ValueError(f"Unknown method: {method}")


def get_inference_method_from_files(output_dir):
    """Determine inference method used based on result files in directory.
    
    Parameters
    ----------
    output_dir : str
        Output directory to check
    
    Returns
    -------
    str or None
        'jaxns' if JAXNS results found, 'numpyro' if NumPyro results found, None if neither
    """
    jaxns_file = find_results_file(output_dir, 'nested_sampling_results_*.json')
    numpyro_file = find_results_file(output_dir, 'numpyro_results_*.nc')
    
    if jaxns_file:
        return 'jaxns'
    elif numpyro_file:
        return 'numpyro'
    else:
        return None