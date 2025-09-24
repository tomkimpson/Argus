"""Utility functions for the argus package."""

import os
import json
import logging
import glob
import jax
import jax.numpy as jnp
import pandas as pd
import configparser
from datetime import datetime

import numpy as np
import arviz as az
import corner
import matplotlib.pyplot as plt

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


def resolve_config_paths(config, config_path):
    """Resolve relative paths in configuration relative to config file location.
    
    This function modifies the config object in-place, converting any relative
    paths to absolute paths based on the directory containing the config file.
    Absolute paths are left unchanged.
    
    Parameters
    ----------
    config : configparser.ConfigParser
        Configuration object to modify
    config_path : str
        Path to the configuration file (used as base for relative paths)
        
    Returns
    -------
    configparser.ConfigParser
        The same config object with resolved paths (modified in-place)
    """
    config_dir = os.path.dirname(os.path.abspath(config_path))
    
    # List of config keys that should be treated as file paths
    path_keys = {
        ('Data', 'data_path'),
        ('PriorModel', 'noise_params_path'), 
        ('PriorModel', 'spin_injections_path')
    }
    
    for section, key in path_keys:
        if config.has_option(section, key):
            path = config.get(section, key).strip()
            if path and not os.path.isabs(path):
                # Convert relative path to absolute path
                resolved_path = os.path.abspath(os.path.join(config_dir, path))
                config.set(section, key, resolved_path)
                
    return config


def get_noise_parameters(config, n_pulsars=None):
    """Get injected noise parameters from configuration and data files.

    Args:
        config: Configuration object
        n_pulsars: Number of pulsars (used for default arrays when files not provided)

    Returns
    -------
        tuple: (efac_array, equad_array, sigma_p_array, gamma_p_array)
    """
    # Use fallback for optional paths - these may be commented out when doing inference
    try:
        noise_params_path = config.get('PriorModel', 'noise_params_path')
        # Check if path is empty string or just whitespace
        if not noise_params_path.strip():
            noise_params_path = None
    except:
        noise_params_path = None

    try:
        spin_injections_path = config.get('PriorModel', 'spin_injections_path')
        # Check if path is empty string or just whitespace
        if not spin_injections_path.strip():
            spin_injections_path = None
    except:
        spin_injections_path = None

    excluded_psrs = config.get('Data', 'excluded_psrs').split(',')
    efac_array, equad_array = get_efac_equad_injections(noise_params_path, excluded_psrs, n_pulsars)
    sigma_p_array, gamma_p_array = get_psr_noise_injections(spin_injections_path, excluded_psrs, n_pulsars)

    return efac_array, equad_array, sigma_p_array, gamma_p_array


def setup_logging(output_dir, config):
    """Set up logging configuration.
    
    DEPRECATED: Use io_manager.setup_single_logger() instead for centralized logging.
    This function is kept for backward compatibility.
    
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
    log_file = os.path.join(output_dir, f'inference_test_output_{timestamp}.txt')
    
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

def get_efac_equad_injections(noise_params_path, excluded_psrs=[], n_pulsars=None):
    """Load EFAC and EQUAD values from noise parameters file.

    Parameters
    ----------
    noise_params_path : str or None
        Path to the noise parameters JSON file, or None if not provided
    excluded_psrs : list of str, optional
        List of pulsar names to exclude from the analysis.
        Default is an empty list.
    n_pulsars : int, optional
        Number of pulsars for default arrays when noise_params_path is None

    Returns
    -------
    tuple
        Two JAX arrays containing EFAC and EQUAD values for the included pulsars.
        Returns default arrays with sensible values if noise_params_path is None.
    """
    if noise_params_path is None:
        # Return default arrays when no noise parameters file is provided
        # Use sensible defaults: EFAC=1.0, EQUAD=1e-7 (in seconds)
        if n_pulsars is None:
            return jnp.array([]), jnp.array([])
        efac_default = jnp.ones(n_pulsars)  # EFAC = 1.0 (no scaling)
        equad_default = jnp.full(n_pulsars, 1e-7)  # EQUAD = 100 ns
        return efac_default, equad_default

    with open(noise_params_path, "r") as f:
        noise_params = json.load(f)

    efac_values = []
    equad_values = []

    for psr in noise_params:
        if not any(excluded_psr in psr for excluded_psr in excluded_psrs):
            efac_values.append(noise_params[psr]["efac"])
            equad_values.append(10**noise_params[psr]["equad"])

    return jnp.array(efac_values), jnp.array(equad_values)

def get_psr_noise_injections(spin_injections_path, excluded_psrs=[], n_pulsars=None):
    """Load pulsar noise parameters from pickle file.

    Parameters
    ----------
    spin_injections_path : str or None
        Path to the spin injections pickle file, or None if not provided
    excluded_psrs : list of str, optional
        List of pulsar names to exclude from the analysis.
        Default is an empty list.
    n_pulsars : int, optional
        Number of pulsars for default arrays when spin_injections_path is None

    Returns
    -------
    tuple
        Two JAX arrays containing sigma_p and gamma_p values for the included pulsars.
        Returns default arrays with sensible values if spin_injections_path is None.
    """
    if spin_injections_path is None:
        # Return default arrays when no spin injections file is provided
        # Use sensible defaults from typical PTA analysis ranges
        if n_pulsars is None:
            return jnp.array([]), jnp.array([])
        # gamma_p ~ 1e-8 to 1e-7 (typical red noise spectral index range)
        gamma_p_default = jnp.full(n_pulsars, 1e-8)
        # sigma_p ~ 1e-15 to 1e-13 (typical red noise amplitude range)
        sigma_p_default = jnp.full(n_pulsars, 1e-14)
        return sigma_p_default, gamma_p_default

    df = pd.read_pickle(spin_injections_path)

    # Create a mask for pulsars to exclude
    exclude_mask = df['psr'].apply(lambda x: not any(excluded_psr in x for excluded_psr in excluded_psrs))
    df_filtered = df[exclude_mask]

    sigma_p_injected = df_filtered['optimal_sigma'].values
    gamma_p_injected = df_filtered['optimal_gamma'].values

    return jnp.array(sigma_p_injected), jnp.array(gamma_p_injected) 






def corner_plot(results, output_dir=None):
    """Create a corner plot for log10_ha parameter from inference results.
    
    Parameters
    ----------
    results : str or object
        Either a file path (string) to results file, or a results object.
        For NumPyro: path to NetCDF file or arviz.InferenceData object
    output_dir : str, optional
        Directory to save the plot. If None, plot is shown but not saved.
        
    Returns
    -------
    str or None
        Path to saved plot file, or None if not saved
    """
    # Determine if results is a file path or loaded object
    if isinstance(results, str):
        # It's a file path - determine type and load
        if results.endswith('.json'):
            # JSON results no longer supported
            raise ValueError("JSON results are no longer supported. Use NumPyro NetCDF files.")
        elif results.endswith('.nc'):
            # NumPyro results
            results_obj = az.from_netcdf(results)
            method = 'numpyro'
        else:
            raise ValueError("Results file must be .nc (NumPyro). Only NetCDF files are supported.")
    else:
        # It's a loaded object - determine type
        if hasattr(results, 'log_Z_mean'):
            # Legacy results object no longer supported
            raise ValueError("Legacy results objects are no longer supported. Use NumPyro ArviZ InferenceData objects.")
        elif hasattr(results, 'posterior'):
            # ArviZ InferenceData object
            results_obj = results
            method = 'numpyro'
        else:
            raise ValueError("Unknown results object type. Only NumPyro ArviZ InferenceData objects are supported.")
    
    # Extract log10_ha samples (NumPyro only)
    if method == 'numpyro':
        if 'log10_ha' not in results_obj.posterior:
            raise ValueError("Parameter 'log10_ha' not found in NumPyro results")
        samples = results_obj.posterior['log10_ha'].values.flatten()
    
    title = "Corner Plot Results"
    
    # Create corner plot (1D histogram for single parameter)
    corner.corner(
        samples.reshape(-1, 1),
        labels=['log₁₀(hₐ)'],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        range=[(-17, -14)],
        bins=30,
        smooth=1.0,
        plot_datapoints=True,
        fill_contours=True,
        levels=[0.68, 0.95]
    )
    
    # Add title
    plt.suptitle(title, y=1.02, fontsize=14)
    plt.tight_layout()
    
    if output_dir is not None:
        # Create plots directory if it doesn't exist
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Extract output_id from results file path if available
        if isinstance(results, str):
            # Extract from filename: {output_id}_results.{ext} -> {output_id}
            basename = os.path.basename(results)
            if '_results.' in basename:
                output_id = basename.split('_results.')[0]
            else:
                output_id = 'unknown'
        else:
            # For loaded objects, use timestamp as fallback
            output_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save the plot
        plot_path = os.path.join(plots_dir, f'corner_plot_{output_id}.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        return plot_path
    else:
        plt.show()
        return None




def diagnostics(fname, output_dir=None):
    """Run MCMC diagnostics on NumPyro results.
    
    Args:
        fname (str): Path to NumPyro results NetCDF file
        output_dir (str, optional): Directory to save diagnostics outputs
    """
    inf_data = az.from_netcdf(fname)
    
    # Set up diagnostics output directory and file
    diagnostics_output = []
    if output_dir is not None:
        diagnostics_dir = os.path.join(output_dir, 'numpyro_diagnostics')
        os.makedirs(diagnostics_dir, exist_ok=True)
        diagnostics_file = os.path.join(diagnostics_dir, 'mcmc_diagnostics.txt')
    
    def log_and_print(message):
        """Print message and add to diagnostics output."""
        print(message)
        if output_dir is not None:
            diagnostics_output.append(message)
    
    log_and_print(f"Successfully loaded InferenceData from: {fname}")


    # --- 1. Get Full Summary ---
    log_and_print("\n--- Calculating Full MCMC Summary ---")
    # Calculate summary for ALL variables first
    full_summary_df = az.summary(inf_data, kind='all', round_to=3, hdi_prob=0.94) # Using default HDI

    # --- 2. Filter out Constant Deterministic Variables ---
    # Identify variables with near-zero standard deviation (likely constants)
    # Use a small tolerance instead of exact zero for floating point comparisons
    tolerance = 1e-12
    is_sampled_or_derived = full_summary_df['sd'] > tolerance
    sampled_summary_df = full_summary_df[is_sampled_or_derived]

    log_and_print("\n--- Filtered MCMC Summary (Excluding Constant Deterministics) ---")
    if sampled_summary_df.empty:
        log_and_print("Warning: No variables with standard deviation > tolerance found. Check model or tolerance.")
    else:
        log_and_print(str(sampled_summary_df))

    # --- 3. Diagnostics on Filtered Parameters ---
    if not sampled_summary_df.empty:
        log_and_print("\n--- Interpretation Guidance (Filtered Parameters) ---")
        log_and_print("Checking R-hat and ESS only for variables that showed variation:")
        # Check R-hat
        max_rhat = sampled_summary_df['r_hat'].max()
        log_and_print(f"\nMaximum R-hat value (filtered): {max_rhat:.3f}")
        if max_rhat > 1.05:
            log_and_print("WARNING: Maximum R-hat is high (> 1.05), suggesting potential convergence issues. Consider increasing 'num_warmup'.")
        elif max_rhat > 1.01:
            log_and_print("NOTE: Maximum R-hat is slightly elevated (> 1.01). Check trace plots carefully.")
        else:
            log_and_print("R-hat values look good (<= 1.01).")

        # Check ESS
        min_ess_bulk = sampled_summary_df['ess_bulk'].min()
        min_ess_tail = sampled_summary_df['ess_tail'].min()
        log_and_print(f"Minimum Bulk ESS (filtered): {min_ess_bulk:.0f}")
        log_and_print(f"Minimum Tail ESS (filtered): {min_ess_tail:.0f}")
        if min_ess_bulk < 400 or min_ess_tail < 400:
            log_and_print("WARNING: Minimum ESS is low (< 400). Consider increasing 'num_samples'.")
        else:
            log_and_print("ESS values look sufficient (>= 400).")

        # --- 4. Trace Plots for Filtered Parameters ---
        log_and_print("\n--- Generating Trace Plots (Filtered Parameters) ---")
        # Get the names of the variables to plot from the filtered summary index
        var_names_to_plot = sampled_summary_df.index.tolist()

        if var_names_to_plot:
            try:
                az.plot_trace(inf_data, var_names=var_names_to_plot, compact=True)
                plt.tight_layout()
                
                if output_dir is not None:
                    # Save trace plot to disk
                    trace_plot_path = os.path.join(diagnostics_dir, 'trace_plots.png')
                    plt.savefig(trace_plot_path, bbox_inches='tight', dpi=150)
                    log_and_print(f"Trace plots saved to: {trace_plot_path}")
                    plt.close()
                else:
                    log_and_print("Displaying trace plots...")
                    plt.show()
                    
            except Exception as e:
                log_and_print(f"Could not generate trace plots: {e}")
        else:
            log_and_print("No variables left to plot after filtering.")

        log_and_print("\n--- Trace Plot Interpretation ---")
        log_and_print("(Interpret as before, focusing on these non-constant variables)")

    else:
        log_and_print("\nSkipping detailed diagnostics as no non-constant parameters were found in the summary.")


    # --- 5. Divergent Transitions (Applies to the whole sampler run) ---
    # This check remains the same as it reflects the sampler's overall behavior
    log_and_print("\n--- Checking for Divergent Transitions ---")
    if "sample_stats" in inf_data and "diverging" in inf_data.sample_stats:
        divergences = inf_data.sample_stats["diverging"].sum().item() # .item() gets scalar value
        total_samples = inf_data.posterior.dims["chain"] * inf_data.posterior.dims["draw"]
        log_and_print(f"Total number of divergent transitions: {divergences}")
        log_and_print(f"Total post-warmup samples: {total_samples}")
        if divergences > 0:
            divergence_rate = (divergences / total_samples) * 100
            log_and_print(f"Divergence rate: {divergence_rate:.2f}%")
            log_and_print("WARNING: Divergences indicate potential issues exploring the posterior.")
            log_and_print("Consider:")
            log_and_print("  1. Model Reparameterization (e.g., non-centered parameterization).")
            log_and_print("  2. Increasing 'target_accept_prob' in the NUTS kernel (e.g., kernel = NUTS(model, target_accept_prob=0.90 or 0.95)).")
            log_and_print("  3. Using stronger priors if appropriate.")
        else:
            log_and_print("No divergent transitions found. Good!")
    else:
        log_and_print("Divergence information not found in sample_stats.")

    log_and_print("\n--- Diagnostics Complete ---")
    
    # Save diagnostics output to file if output_dir is provided
    if output_dir is not None:
        with open(diagnostics_file, 'w') as f:
            f.write('\n'.join(diagnostics_output))
        print(f"Diagnostics saved to: {diagnostics_file}")



















def find_results_file(output_dir, file_pattern):
    """Find results file in output directory matching pattern.
    
    Parameters
    ----------
    output_dir : str
        Directory to search for results files
    file_pattern : str
        Glob pattern to match files (e.g., 'inference_results_*.json')
    
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
        Results object (ArviZ only)
    method : str
        Inference method ('numpyro' only)
    
    Returns
    -------
    tuple
        (log_evidence, log_evidence_uncertainty)
    """
    if method.lower() == 'jaxns':
        raise ValueError("JAXNS method no longer supported")
    elif method.lower() == 'numpyro':
        # For NumPyro, we need to estimate evidence using importance sampling or other methods
        # This is a placeholder - proper evidence estimation for MCMC is non-trivial
        # For now, return None to indicate evidence not available
        return None, None
    else:
        raise ValueError(f"Unknown method: {method}")


def load_jaxns_results(results_file):
    """Load JAXNS results - DEPRECATED.
    
    Parameters
    ----------
    results_file : str
        Path to results file
        
    Raises
    ------
    ValueError
        Always raised as JAXNS is no longer supported
    """
    raise ValueError("JAXNS results loading is no longer supported. Use NumPyro instead.")


def get_inference_method_from_files(output_dir):
    """Determine inference method used based on result files in directory.
    
    Parameters
    ----------
    output_dir : str
        Output directory to check
    
    Returns
    -------
    str or None
        'numpyro' if NumPyro results found, None if not found
    """
    numpyro_file = find_results_file(output_dir, '*_results.nc')
    
    if numpyro_file:
        return 'numpyro'
    else:
        return None