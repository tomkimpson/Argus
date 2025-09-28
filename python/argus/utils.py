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

import arviz as az
import corner
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import scienceplots

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

def get_noise_parameters(config):
    """Get injected noise parameters from configuration and data files.
    
    Args:
        config: Configuration object
    
    Returns
    -------
        tuple: (efac_array, equad_array, sigma_p_array, gamma_p_array)
    """
    noise_params_path = config.get('PriorModel', 'noise_params_path')
    spin_injections_path = config.get('PriorModel', 'spin_injections_path')
    excluded_psrs = config.get('Data', 'excluded_psrs').split(',')
    efac_array, equad_array = get_efac_equad_injections(noise_params_path, excluded_psrs)
    sigma_p_array, gamma_p_array = get_psr_noise_injections(spin_injections_path, excluded_psrs)
    
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

def corner_plot(results, output_dir=None, plot_priors=False, smooth_sigma=1.0, nbins=40):
    """Create a publication-quality corner plot for log10_ha and log10_gamma_a parameters.

    This function uses professional scientific plotting styles and creates high-quality
    corner plots suitable for publications. Supports both 1D (log10_ha only) and 2D
    (log10_ha + log10_gamma_a) plots.

    Parameters
    ----------
    results : str or arviz.InferenceData
        Either a file path to NumPyro NetCDF file (.nc) or arviz.InferenceData object
    output_dir : str, optional
        Directory to save the plot. If None, plot is shown but not saved.
    plot_priors : bool, optional
        Whether to overlay prior distributions on 1D histograms (default: False)
    smooth_sigma : float, optional
        Smoothing parameter for histograms (default: 1.0)
    nbins : int, optional
        Number of bins for histograms (default: 40)

    Returns
    -------
    str or None
        Path to saved PNG plot file, or None if not saved. Also saves PDF version.

    Notes
    -----
    - Requires scienceplots package for professional styling
    - Always saves both PNG and PDF versions when output_dir is provided
    - Uses professional blue color scheme with gradient contours
    - Includes minor ticks, grids, and enhanced typography
    """
    # Set up professional styling
    plt.style.use(["science", "no-latex"])  # Add 'no-latex' if LaTeX not available
    plt.rcParams.update({
        "font.size": 12,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "figure.dpi": 100,
        "savefig.dpi": 300,
    })

    # Load results - simplified to handle only NumPyro
    if isinstance(results, str):
        # File path to NetCDF file
        if not results.endswith('.nc'):
            raise ValueError("Results file must be .nc (NumPyro NetCDF file)")
        results_obj = az.from_netcdf(results)
        results_path = results
    else:
        # ArviZ InferenceData object
        if not hasattr(results, 'posterior'):
            raise ValueError("Results object must be ArviZ InferenceData with posterior samples")
        results_obj = results
        results_path = None

    # Extract parameters
    if 'log10_ha' not in results_obj.posterior:
        raise ValueError("Parameter 'log10_ha' not found in results")

    log10_ha = results_obj.posterior['log10_ha'].values.flatten()
    samples_list = [log10_ha]
    labels = [r'$\log_{10} h_a$']

    # Extract log10_gamma_a if available
    if 'log10_gamma_a' in results_obj.posterior.data_vars:
        log10_gamma_a = results_obj.posterior['log10_gamma_a'].values.flatten()
        samples_list.append(log10_gamma_a)
        labels.append(r'$\log_{10} \gamma_a$')
        print("Found log10_gamma_a parameter - creating 2D corner plot")
    else:
        print("log10_gamma_a not found - creating 1D plot for log10_ha only")

    # Combine parameters
    samples = np.column_stack(samples_list)

    # Load config for prior plotting if requested
    config = None
    if plot_priors and results_path:
        config = _load_config_from_results_path(results_path)

    # Print parameter information
    _print_parameter_ranges(samples, labels, config)

    # Define plot ranges (slightly extended from typical prior ranges)
    if len(labels) == 1:
        plot_ranges = [(-17.5, -13.5)]  # log10_ha
    else:
        plot_ranges = [
            (-17.5, -13.5),  # log10_ha
            (-10.5, -7.5)    # log10_gamma_a
        ]

    # Professional color scheme
    posterior_color = "#2E86C1"  # Professional blue
    contour_colors = ["#AED6F1", "#5DADE2", "#2E86C1"]  # Gradient blues
    truth_color = "orange"

    # Create publication-quality corner plot
    fig = corner.corner(
        samples,
        labels=labels,
        show_titles=True,
        title_kwargs={"fontsize": 14, "fontweight": "bold", "pad": 10},
        label_kwargs={"fontsize": 16, "fontweight": "bold"},
        title_fmt=".3f",
        bins=nbins,
        quantiles=[0.16, 0.5, 0.84],  # 68% credible intervals
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2)),  # 1σ and 2σ contours
        plot_density=True,
        plot_datapoints=False,  # Clean look without individual points
        fill_contours=True,
        range=plot_ranges,
        color=posterior_color,
        contour_kwargs={"colors": contour_colors, "linewidths": 0.5},
        #hist_kwargs={"alpha": 0.8, "edgecolor": posterior_color, "linewidth": 1.0},
        max_n_ticks=4,
        use_math_text=True,
        smooth=smooth_sigma,
        smooth1d=smooth_sigma
    )

    # Post-processing improvements
    axes = fig.get_axes()

    # Enhance axis appearance
    for ax in axes:
        if ax is not None:
            # Improve tick appearance
            ax.tick_params(
                which="major",
                labelsize=12,
                width=1.2,
                length=6,
                direction="in",
                top=True,
                right=True,
            )
            ax.tick_params(
                which="minor",
                width=0.8,
                length=3,
                direction="in",
                top=True,
                right=True
            )

            # Add minor ticks
            ax.minorticks_on()

            # Add subtle grid
            ax.grid(True, alpha=0.3, linewidth=0.5, linestyle=":")

    # Add prior overlays if requested
    if plot_priors and config is not None:
        _add_prior_overlays(fig, labels, config)

    # Adjust layout
    plt.tight_layout()

    # Manually adjust layout with reduced spacing
    plt.subplots_adjust(
        hspace=0.05,  # Reduce vertical spacing between subplots
        wspace=0.05,  # Reduce horizontal spacing between subplots
    )

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

        # Generate filename with options
        smooth_suffix = f"_smooth{smooth_sigma}" if smooth_sigma != 1.0 else ""
        priors_suffix = "_priors" if plot_priors else ""
        params_suffix = "_ha_gamma_a" if len(labels) > 1 else "_ha_only"

        # Save PNG for general use
        plot_path = os.path.join(plots_dir, f'corner_plot_{output_id}{params_suffix}{smooth_suffix}{priors_suffix}.png')
        fig.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

        # Also save PDF for publications
        pdf_path = plot_path.replace('.png', '.pdf')
        fig.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')

        plt.close()
        print(f"Corner plot saved to: {plot_path}")
        print(f"PDF version saved to: {pdf_path}")
        return plot_path
    else:
        plt.show()
        return None


def _load_config_from_results_path(results_file):
    """Try to load corresponding config file for the given results file."""
    results_dir = os.path.dirname(results_file)

    # Look for config files in the results directory
    config_candidates = [
        os.path.join(results_dir, 'config.ini'),
        os.path.join(results_dir, f'{os.path.basename(results_dir)}_config.ini'),
    ]

    # Also look in parent directory for argus-style configs
    parent_dir = os.path.dirname(results_dir)
    run_id = os.path.basename(results_file).replace('_results.nc', '').replace('numpyro_test_', '')
    config_candidates.extend([
        os.path.join(parent_dir, 'configs', f'config_numpyro_test_{run_id}.ini'),
        os.path.join('configs', f'config_numpyro_test_{run_id}.ini'),
    ])

    for config_file in config_candidates:
        if os.path.exists(config_file):
            print(f"Loading config from: {config_file}")
            config = configparser.ConfigParser()
            config.read(config_file)
            return config

    print("Warning: No config file found. Prior plotting will be disabled.")
    return None


def _get_log10_ha_prior_pdf(config, x):
    """Get prior PDF for log10_ha parameter."""
    if config is None:
        return None

    try:
        if config.getboolean('PriorModel', 'log10_ha_fixed', fallback=False):
            return None  # Fixed parameter, no prior to plot

        # Get uniform range
        min_val = config.getfloat('PriorModel', 'log10_ha_min', fallback=-16.0)
        max_val = config.getfloat('PriorModel', 'log10_ha_max', fallback=-14.0)

        # Create uniform distribution
        return stats.uniform(loc=min_val, scale=max_val - min_val).pdf(x)
    except (configparser.NoSectionError, configparser.NoOptionError):
        return None


def _get_log10_gamma_a_prior_pdf(config, x):
    """Get prior PDF for log10_gamma_a parameter."""
    if config is None:
        return None

    try:
        if config.getboolean('PriorModel', 'log10_gamma_a_fixed', fallback=False):
            return None  # Fixed parameter, no prior to plot

        # Get uniform range
        min_val = config.getfloat('PriorModel', 'log10_gamma_a_min', fallback=-10.0)
        max_val = config.getfloat('PriorModel', 'log10_gamma_a_max', fallback=-8.0)

        # Create uniform distribution
        return stats.uniform(loc=min_val, scale=max_val - min_val).pdf(x)
    except (configparser.NoSectionError, configparser.NoOptionError):
        return None


def _print_parameter_ranges(samples, labels, config):
    """Print parameter ranges and prior information."""
    print(f"\nParameter ranges from posterior samples:")
    print("=" * 60)
    for i, label in enumerate(labels):
        param_min = samples[:, i].min()
        param_max = samples[:, i].max()
        param_mean = samples[:, i].mean()
        param_std = samples[:, i].std()
        print(f"  {label}:")
        print(f"    Range: [{param_min:.3f}, {param_max:.3f}]")
        print(f"    Mean ± Std: {param_mean:.3f} ± {param_std:.3f}")

    if config is not None:
        print(f"\nPrior ranges from config:")
        print("=" * 60)
        try:
            # log10_ha prior
            if not config.getboolean('PriorModel', 'log10_ha_fixed', fallback=False):
                min_val = config.getfloat('PriorModel', 'log10_ha_min', fallback=-16.0)
                max_val = config.getfloat('PriorModel', 'log10_ha_max', fallback=-14.0)
                print(f"  log10_ha: U({min_val}, {max_val})")
            else:
                fixed_val = config.getfloat('PriorModel', 'log10_ha_value', fallback=-15.0)
                print(f"  log10_ha: Fixed at {fixed_val}")

            # log10_gamma_a prior
            if len(labels) > 1:  # Only if gamma_a is included
                if not config.getboolean('PriorModel', 'log10_gamma_a_fixed', fallback=False):
                    min_val = config.getfloat('PriorModel', 'log10_gamma_a_min', fallback=-10.0)
                    max_val = config.getfloat('PriorModel', 'log10_gamma_a_max', fallback=-8.0)
                    print(f"  log10_gamma_a: U({min_val}, {max_val})")
                else:
                    fixed_val = config.getfloat('PriorModel', 'log10_gamma_a_value', fallback=-9.0)
                    print(f"  log10_gamma_a: Fixed at {fixed_val}")
        except (configparser.NoSectionError, configparser.NoOptionError):
            print("  Config sections not found for prior information")

    print("=" * 60)


def _add_prior_overlays(fig, labels, config):
    """Add prior overlays to corner plot."""
    axes = fig.get_axes()
    ndim = len(labels)

    # Prior overlay for each parameter (diagonal elements)
    for i in range(ndim):
        ax = axes[i * ndim + i]  # Diagonal axis

        # Get parameter name and x-range for prior
        x_min, x_max = ax.get_xlim()
        x_range = np.linspace(x_min, x_max, 200)

        # Get prior PDF based on parameter
        prior_pdf = None
        if i == 0:  # log10_ha
            prior_pdf = _get_log10_ha_prior_pdf(config, x_range)
        elif i == 1 and len(labels) > 1:  # log10_gamma_a
            prior_pdf = _get_log10_gamma_a_prior_pdf(config, x_range)

        # Plot prior if available
        if prior_pdf is not None and np.any(prior_pdf > 0) and np.all(np.isfinite(prior_pdf)):
            # Scale prior to match histogram height
            y_max = ax.get_ylim()[1]
            prior_max = np.max(prior_pdf)
            if prior_max > 0:
                prior_pdf_scaled = prior_pdf * y_max / prior_max * 0.7  # Scale to 70% of max
                ax.plot(x_range, prior_pdf_scaled, 'r--', linewidth=2.5,
                       alpha=0.8, label='Prior')
                ax.legend(fontsize=12)

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