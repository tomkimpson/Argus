#!/usr/bin/env python3
"""
Create corner plot from NUMPYRO results showing log10_ha and 
sigma_p, gamma_p parameters for selected pulsars.

This script supports:
- Optional inclusion of log10_gamma_a parameter (when sampled)
- Overlay of prior distributions on 1D posterior histograms
- Handling of complex prior structures (hierarchical, log-ratio parameterization)
- Various smoothing options for better visualization

Usage: python create_corner_plot.py <run_index> <num_pulsars> [smooth_sigma] [--plot_log10_gamma_a] [--plot_priors] [--efac] [--equad] [--plot_injections]

Examples:
  python create_corner_plot.py 016 2                                # Basic plot
  python create_corner_plot.py 016 2 1.0                            # With smoothing
  python create_corner_plot.py 016 2 0.5 --plot_log10_gamma_a      # Include log10_gamma_a
  python create_corner_plot.py 016 2 1.0 --plot_priors              # With prior overlays
  python create_corner_plot.py 028 2 --efac --equad                 # Include EFAC and EQUAD
  python create_corner_plot.py 028 2 --efac --equad --plot_priors   # EFAC/EQUAD with priors
  python create_corner_plot.py 028 2 --efac --equad --plot_injections # EFAC/EQUAD with injections
  python create_corner_plot.py 016 2 1.0 --plot_log10_gamma_a --plot_priors # Full featured

Arguments:
  run_index     : Run index (e.g., 016, 022, 023) 
  num_pulsars   : Number of pulsars to include in the plot
  smooth_sigma  : Optional smoothing parameter for histograms
  --plot_log10_gamma_a: Include log10_gamma_a parameter if available in posterior
  --plot_priors : Overlay prior distributions on 1D histograms
  --efac        : Include EFAC parameters for each pulsar
  --equad       : Include EQUAD parameters for each pulsar
  --plot_injections : Overlay injection values for EFAC/EQUAD parameters

Features:
- Automatically detects parameter types and applies appropriate priors
- Handles reparameterized parameters (log10_ha with NUTS optimization)
- Supports hierarchical Bayesian priors (gamma_p with population hyperpriors)
- Approximates complex log-ratio parameterizations (sigma_p derived from gamma_p + ratio)
- Scales prior overlays to match histogram heights for visual comparison
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import corner
import os
import configparser
import json
import glob
import re
from scipy import stats
from scipy.integrate import quad


def load_config(run_index):
    """Load configuration file for the given run index."""
    config_file = f"../argus/configs/config_numpyro_test_{run_index}.ini"
    if not os.path.exists(config_file):
        print(f"Warning: Config file {config_file} not found. Prior plotting will be disabled.")
        return None
    
    config = configparser.ConfigParser()
    config.read(config_file)
    return config


def get_log10_ha_prior_pdf(config, x):
    """Get prior PDF for log10_ha parameter.
    
    Note: The posterior samples are in the original log10_ha space (not reparameterized),
    so we plot the original uniform prior regardless of whether reparameterization was used.
    """
    if config.getboolean('PriorModel', 'log10_ha_fixed'):
        return None  # Fixed parameter, no prior to plot
    
    # Get original uniform range - this is the effective prior regardless of reparameterization
    min_val = config.getfloat('PriorModel', 'log10_ha_min')
    max_val = config.getfloat('PriorModel', 'log10_ha_max')
    
    # Create uniform distribution over original range
    return stats.uniform(loc=min_val, scale=max_val - min_val).pdf(x)


def get_log10_gamma_a_prior_pdf(config, x):
    """Get prior PDF for log10_gamma_a parameter."""
    if config.getboolean('PriorModel', 'log10_gamma_a_fixed'):
        return None  # Fixed parameter, no prior to plot
    
    # Get uniform range
    min_val = config.getfloat('PriorModel', 'log10_gamma_a_min')
    max_val = config.getfloat('PriorModel', 'log10_gamma_a_max')
    
    # Create uniform distribution
    return stats.uniform(loc=min_val, scale=max_val - min_val).pdf(x)


def get_log10_gamma_p_prior_pdf(config, x):
    """Get prior PDF for log10_gamma_p parameter."""
    if config.getboolean('PriorModel', 'hierarchical_noise'):
        # Hierarchical case: Use Monte Carlo approximation for efficiency
        # Sample from hyperpriors and approximate the marginal distribution
        mean_min = config.getfloat('PriorModel', 'log10_gamma_p_mean_min')
        mean_max = config.getfloat('PriorModel', 'log10_gamma_p_mean_max')
        std_min = config.getfloat('PriorModel', 'log10_gamma_p_std_min')
        std_max = config.getfloat('PriorModel', 'log10_gamma_p_std_max')
        
        # Monte Carlo approximation
        n_samples = 1000
        np.random.seed(42)  # For reproducibility
        means = np.random.uniform(mean_min, mean_max, n_samples)
        stds = np.random.uniform(std_min, std_max, n_samples)
        
        # Approximate marginal PDF
        pdf_values = np.zeros_like(x)
        for i, x_val in enumerate(x):
            pdf_samples = stats.norm(loc=means, scale=stds).pdf(x_val)
            pdf_values[i] = np.mean(pdf_samples)
        
        return pdf_values
    else:
        # Simple uniform case
        min_val = config.getfloat('PriorModel', 'log10_gamma_p_min')
        max_val = config.getfloat('PriorModel', 'log10_gamma_p_max')
        return stats.uniform(loc=min_val, scale=max_val - min_val).pdf(x)


def get_log10_sigma_p_prior_pdf(config, x):
    """Get prior PDF for log10_sigma_p parameter."""
    if config.getboolean('PriorModel', 'log_ratio_parameterization'):
        # Log-ratio case: log10(σp) = log10(γp) + log10(ratio)
        # Use Monte Carlo sampling to approximate the true hierarchical prior
        print("Log-ratio parameterization detected for sigma_p. Using Monte Carlo approximation of hierarchical prior.")
        
        # Get hyperprior ranges
        gamma_p_mean_min = config.getfloat('PriorModel', 'log10_gamma_p_mean_min', fallback=-9.0)
        gamma_p_mean_max = config.getfloat('PriorModel', 'log10_gamma_p_mean_max', fallback=-7.0)
        gamma_p_std_min = config.getfloat('PriorModel', 'log10_gamma_p_std_min', fallback=0.1)
        gamma_p_std_max = config.getfloat('PriorModel', 'log10_gamma_p_std_max', fallback=1.0)
        
        ratio_mean_min = config.getfloat('PriorModel', 'log10_ratio_mean_min', fallback=-8.0)
        ratio_mean_max = config.getfloat('PriorModel', 'log10_ratio_mean_max', fallback=-4.0)
        ratio_std_min = config.getfloat('PriorModel', 'log10_ratio_std_min', fallback=0.5)
        ratio_std_max = config.getfloat('PriorModel', 'log10_ratio_std_max', fallback=3.0)
        
        # Monte Carlo sampling to approximate prior
        n_samples = 10000
        np.random.seed(42)  # For reproducibility
        
        # Sample hyperparameters
        gamma_p_means = np.random.uniform(gamma_p_mean_min, gamma_p_mean_max, n_samples)
        gamma_p_stds = np.random.uniform(gamma_p_std_min, gamma_p_std_max, n_samples)
        ratio_means = np.random.uniform(ratio_mean_min, ratio_mean_max, n_samples)
        ratio_stds = np.random.uniform(ratio_std_min, ratio_std_max, n_samples)
        
        # Sample individual parameters
        log10_gamma_p_samples = np.random.normal(gamma_p_means, gamma_p_stds)
        log10_ratio_samples = np.random.normal(ratio_means, ratio_stds)
        
        # Compute σp = γp + ratio
        log10_sigma_p_samples = log10_gamma_p_samples + log10_ratio_samples
        
        # Estimate PDF using kernel density estimation
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(log10_sigma_p_samples)
        return kde(x)
    else:
        # Simple uniform case
        min_val = config.getfloat('PriorModel', 'log10_sigma_p_min')
        max_val = config.getfloat('PriorModel', 'log10_sigma_p_max')
        return stats.uniform(loc=min_val, scale=max_val - min_val).pdf(x)


def get_efac_prior_pdf(config, x):
    """Get prior PDF for EFAC parameter."""
    if config.getboolean('PriorModel', 'efac_equad_fixed', fallback=True):
        return None  # Fixed parameter, no prior to plot
    
    # Get uniform range
    min_val = config.getfloat('PriorModel', 'efac_min', fallback=0.1)
    max_val = config.getfloat('PriorModel', 'efac_max', fallback=3.0)
    
    # Create uniform distribution
    return stats.uniform(loc=min_val, scale=max_val - min_val).pdf(x)


def get_equad_prior_pdf(config, x):
    """Get prior PDF for EQUAD parameter."""
    if config.getboolean('PriorModel', 'efac_equad_fixed', fallback=True):
        return None  # Fixed parameter, no prior to plot
    
    # Check if log10 parameterization is used
    if config.getboolean('PriorModel', 'equad_log10_prior', fallback=False):
        # log10(EQUAD) ~ Uniform(min, max)
        min_val = config.getfloat('PriorModel', 'log10_equad_min', fallback=-9.0)
        max_val = config.getfloat('PriorModel', 'log10_equad_max', fallback=-5.0)
        return stats.uniform(loc=min_val, scale=max_val - min_val).pdf(x)
    else:
        # Direct EQUAD ~ Uniform(min, max)
        min_val = config.getfloat('PriorModel', 'equad_min', fallback=1e-9)
        max_val = config.getfloat('PriorModel', 'equad_max', fallback=1e-5)
        return stats.uniform(loc=min_val, scale=max_val - min_val).pdf(x)


def print_prior_ranges(config, labels, plot_log10_gamma_a, plot_efac, plot_equad, num_pulsars):
    """Print prior ranges for all parameters to command line."""
    if config is None:
        print("Prior ranges: Config file not available, cannot display prior information.")
        return
    
    print("\nPrior ranges:")
    print("=" * 50)
    
    param_idx = 0
    
    # log10_ha
    if param_idx < len(labels):
        if not config.getboolean('PriorModel', 'log10_ha_fixed', fallback=False):
            min_val = config.getfloat('PriorModel', 'log10_ha_min', fallback=-16.0)
            max_val = config.getfloat('PriorModel', 'log10_ha_max', fallback=-14.0)
            print(f"  {labels[param_idx]}: U({min_val}, {max_val})")
        else:
            fixed_val = config.getfloat('PriorModel', 'log10_ha_value', fallback=-15.0)
            print(f"  {labels[param_idx]}: Fixed at {fixed_val}")
        param_idx += 1
    
    # log10_gamma_a
    if plot_log10_gamma_a and param_idx < len(labels):
        if not config.getboolean('PriorModel', 'log10_gamma_a_fixed', fallback=False):
            min_val = config.getfloat('PriorModel', 'log10_gamma_a_min', fallback=-10.0)
            max_val = config.getfloat('PriorModel', 'log10_gamma_a_max', fallback=-8.0)
            print(f"  {labels[param_idx]}: U({min_val}, {max_val})")
        else:
            fixed_val = config.getfloat('PriorModel', 'log10_gamma_a_value', fallback=-9.0)
            print(f"  {labels[param_idx]}: Fixed at {fixed_val}")
        param_idx += 1
    
    # sigma_p and gamma_p for each pulsar
    for i in range(num_pulsars):
        # sigma_p
        if param_idx < len(labels):
            if config.getboolean('PriorModel', 'log_ratio_parameterization', fallback=False):
                print(f"  {labels[param_idx]}: Hierarchical (log-ratio parameterization)")
            else:
                min_val = config.getfloat('PriorModel', 'log10_sigma_p_min', fallback=-18.0)
                max_val = config.getfloat('PriorModel', 'log10_sigma_p_max', fallback=-12.0)
                print(f"  {labels[param_idx]}: U({min_val}, {max_val})")
            param_idx += 1
        
        # gamma_p
        if param_idx < len(labels):
            if config.getboolean('PriorModel', 'hierarchical_noise', fallback=False):
                mean_min = config.getfloat('PriorModel', 'log10_gamma_p_mean_min', fallback=-9.0)
                mean_max = config.getfloat('PriorModel', 'log10_gamma_p_mean_max', fallback=-7.0)
                std_min = config.getfloat('PriorModel', 'log10_gamma_p_std_min', fallback=0.1)
                std_max = config.getfloat('PriorModel', 'log10_gamma_p_std_max', fallback=1.0)
                print(f"  {labels[param_idx]}: Hierarchical (μ~U({mean_min},{mean_max}), σ~U({std_min},{std_max}))")
            else:
                min_val = config.getfloat('PriorModel', 'log10_gamma_p_min', fallback=-11.0)
                max_val = config.getfloat('PriorModel', 'log10_gamma_p_max', fallback=-6.0)
                print(f"  {labels[param_idx]}: U({min_val}, {max_val})")
            param_idx += 1
    
    # EFAC for each pulsar
    if plot_efac:
        for i in range(num_pulsars):
            if param_idx < len(labels):
                if not config.getboolean('PriorModel', 'efac_equad_fixed', fallback=True):
                    min_val = config.getfloat('PriorModel', 'efac_min', fallback=0.1)
                    max_val = config.getfloat('PriorModel', 'efac_max', fallback=3.0)
                    print(f"  {labels[param_idx]}: U({min_val}, {max_val})")
                else:
                    print(f"  {labels[param_idx]}: Fixed at 1.0")
                param_idx += 1
    
    # EQUAD for each pulsar
    if plot_equad:
        for i in range(num_pulsars):
            if param_idx < len(labels):
                if not config.getboolean('PriorModel', 'efac_equad_fixed', fallback=True):
                    if config.getboolean('PriorModel', 'equad_log10_prior', fallback=False):
                        min_val = config.getfloat('PriorModel', 'log10_equad_min', fallback=-9.0)
                        max_val = config.getfloat('PriorModel', 'log10_equad_max', fallback=-5.0)
                        print(f"  {labels[param_idx]}: U({min_val}, {max_val}) [log10 space]")
                    else:
                        min_val = config.getfloat('PriorModel', 'equad_min', fallback=1e-9)
                        max_val = config.getfloat('PriorModel', 'equad_max', fallback=1e-5)
                        print(f"  {labels[param_idx]}: U({min_val:.0e}, {max_val:.0e})")
                else:
                    print(f"  {labels[param_idx]}: Fixed at 0.0")
                param_idx += 1
    
    print("=" * 50)


def create_pulsar_name_to_index_mapping(config):
    """Create mapping from pulsar name to index based on data loading order."""
    if config is None:
        return {}
    
    try:
        # Get data path from config
        data_path = config.get('DataModel', 'data_path', fallback='../data/IPTA_MockDataChallenge2/')
        if not data_path.endswith('/'):
            data_path += '/'
        
        # Get excluded pulsars
        excluded_pulsars = config.get('DataModel', 'excluded_pulsars', fallback='').split(',')
        excluded_pulsars = [p.strip() for p in excluded_pulsars if p.strip()]
        
        # Find all .par files and sort them (same logic as data loader)
        par_files = sorted(glob.glob(os.path.join(data_path, '*.par')))
        
        # Extract pulsar names and apply exclusions
        pulsar_names = []
        for par_file in par_files:
            pulsar_name = os.path.basename(par_file).replace('.par', '')
            if pulsar_name not in excluded_pulsars:
                pulsar_names.append(pulsar_name)
        
        # Create mapping
        name_to_index = {name: idx for idx, name in enumerate(pulsar_names)}
        
        print(f"Created pulsar mapping for {len(pulsar_names)} pulsars (excluded: {excluded_pulsars})")
        return name_to_index
        
    except Exception as e:
        print(f"Warning: Could not create pulsar mapping: {e}")
        return {}


def load_injection_data():
    """Load injection data from JSON file."""
    injection_file = "../data/IPTA_MockDataChallenge2/group1_psr_noise.json"
    
    try:
        with open(injection_file, 'r') as f:
            injection_data = json.load(f)
        print(f"Loaded injection data for {len(injection_data)} pulsars")
        return injection_data
    except Exception as e:
        print(f"Warning: Could not load injection data from {injection_file}: {e}")
        return {}


# Parse command line arguments
parser = argparse.ArgumentParser(description='Create corner plot from NUMPYRO results')
parser.add_argument('run_index', type=str, help='Run index (e.g., 016, 022, 023)')
parser.add_argument('num_pulsars', type=int, help='Number of pulsars to include')
parser.add_argument('smooth_sigma', type=float, nargs='?', default=None, help='Smoothing parameter for histograms')
parser.add_argument('--plot_log10_gamma_a', action='store_true', help='Include log10_gamma_a parameter in the plot')
parser.add_argument('--plot_priors', action='store_true', help='Overlay prior distributions on 1D histograms')
parser.add_argument('--efac', action='store_true', help='Include EFAC parameters for each pulsar')
parser.add_argument('--equad', action='store_true', help='Include EQUAD parameters for each pulsar')
parser.add_argument('--plot_injections', action='store_true', help='Overlay injection values for EFAC/EQUAD parameters')

args = parser.parse_args()
run_index = args.run_index
num_pulsars = args.num_pulsars
smooth_sigma = args.smooth_sigma
plot_log10_gamma_a = args.plot_log10_gamma_a
plot_priors = args.plot_priors
plot_efac = args.efac
plot_equad = args.equad
plot_injections = args.plot_injections

# Load config file for prior printing and plotting
config = load_config(run_index)

# Load injection data if requested
injection_data = {}
pulsar_name_to_index = {}
if plot_injections:
    injection_data = load_injection_data()
    pulsar_name_to_index = create_pulsar_name_to_index_mapping(config)

# Load the results
results_file = f"numpyro_test_{run_index}/numpyro_test_{run_index}_results.nc"
print(f"Loading results from: {results_file}")

if not os.path.exists(results_file):
    print(f"Error: Results file {results_file} not found!")
    sys.exit(1)

idata = az.from_netcdf(results_file)

# Get the posterior samples
posterior = idata.posterior

# Extract log10_ha
log10_ha = posterior['log10_ha'].values.flatten()

# Select pulsars (indices 0 to num_pulsars-1)
pulsar_indices = list(range(num_pulsars))
print(f"Creating corner plot for {num_pulsars} pulsars at indices: {pulsar_indices}")

# Start with log10_ha
samples_list = [log10_ha]
labels = [r'$\log_{10} h_a$']

# Add log10_gamma_a if requested and available
if plot_log10_gamma_a:
    if 'log10_gamma_a' in posterior.data_vars:
        log10_gamma_a = posterior['log10_gamma_a'].values.flatten()
        samples_list.append(log10_gamma_a)
        labels.append(r'$\log_{10} \gamma_a$')
        print(f"Added log10_gamma_a parameter to plot")
    else:
        print(f"Warning: log10_gamma_a requested but not found in posterior. Skipping log10_gamma_a.")

# Extract sigma_p and gamma_p for selected pulsars
for i, pulsar_idx in enumerate(pulsar_indices):
    sigma_p = posterior['log10_σp'].isel(log10_σp_dim_0=pulsar_idx).values.flatten()
    gamma_p = posterior['log10_γp'].isel(log10_γp_dim_0=pulsar_idx).values.flatten()
    
    samples_list.append(sigma_p)
    samples_list.append(gamma_p)
    
    labels.append(rf'$\log_{{10}} \sigma_{{p,{i}}}$')
    labels.append(rf'$\log_{{10}} \gamma_{{p,{i}}}$')

# Extract EFAC parameters for selected pulsars if requested
if plot_efac:
    if 'efac' in posterior.data_vars:
        for i, pulsar_idx in enumerate(pulsar_indices):
            efac = posterior['efac'].isel(efac_dim_0=pulsar_idx).values.flatten()
            samples_list.append(efac)
            labels.append(rf'$\mathrm{{EFAC}}_{{p,{i}}}$')
        print(f"Added EFAC parameters for {num_pulsars} pulsars")
    else:
        print(f"Warning: EFAC requested but not found in posterior. Skipping EFAC.")

# Extract EQUAD parameters for selected pulsars if requested  
if plot_equad:
    # Prefer log10_equad if available, otherwise use equad
    if 'log10_equad' in posterior.data_vars:
        for i, pulsar_idx in enumerate(pulsar_indices):
            log10_equad = posterior['log10_equad'].isel(log10_equad_dim_0=pulsar_idx).values.flatten()
            samples_list.append(log10_equad)
            labels.append(rf'$\log_{{10}} \mathrm{{EQUAD}}_{{p,{i}}}$')
        print(f"Added log10_EQUAD parameters for {num_pulsars} pulsars")
    elif 'equad' in posterior.data_vars:
        for i, pulsar_idx in enumerate(pulsar_indices):
            equad = posterior['equad'].isel(equad_dim_0=pulsar_idx).values.flatten()
            samples_list.append(equad)
            labels.append(rf'$\mathrm{{EQUAD}}_{{p,{i}}}$')
        print(f"Added EQUAD parameters for {num_pulsars} pulsars")
    else:
        print(f"Warning: EQUAD requested but neither 'log10_equad' nor 'equad' found in posterior. Skipping EQUAD.")

# Combine all parameters
samples = np.column_stack(samples_list)

print(f"Sample shape: {samples.shape}")
print(f"Parameter ranges:")
for i, label in enumerate(labels):
    print(f"  {label}: [{samples[:, i].min():.3f}, {samples[:, i].max():.3f}]")

# Print prior ranges from config file
print_prior_ranges(config, labels, plot_log10_gamma_a, plot_efac, plot_equad, num_pulsars)

# Define prior ranges from config file - extended to check for railing
prior_ranges = [[-18.5, -13.5]]  # log10_ha - extended from [-16.0, -14.0]

# Add log10_gamma_a range if plotting
if plot_log10_gamma_a and 'log10_gamma_a' in posterior.data_vars:
    prior_ranges.append([-10.0, -8.0])  # log10_gamma_a - typical range

for i in range(num_pulsars):
    prior_ranges.append([-20.5, -11.5])  # log10_sigma_p - extended from [-18.0, -12.0]
    prior_ranges.append([-11.5, -5.5])   # log10_gamma_p - extended from [-11.0, -6.0]

# Add EFAC ranges if plotting
if plot_efac:
    for i in range(num_pulsars):
        prior_ranges.append([0.05, 3.5])  # EFAC - extended from [0.1, 3.0]

# Add EQUAD ranges if plotting
if plot_equad:
    for i in range(num_pulsars):
        if 'log10_equad' in posterior.data_vars:
            prior_ranges.append([-9.5, -4.5])  # log10_EQUAD - extended from [-9.0, -5.0]
        else:
            prior_ranges.append([5e-10, 2e-5])  # EQUAD - extended from [1e-9, 1e-5]

# Configure plot range and smoothing based on smooth_sigma parameter
if smooth_sigma is not None:
    # Use smoothing and prior ranges
    plot_range = prior_ranges
    smooth_option = smooth_sigma
    smooth1d_option = smooth_sigma
    range_desc = "prior bounds"
    smooth_desc = f"sigma={smooth_sigma}"
else:
    # No smoothing, use sample ranges
    plot_range = None  # Let corner.corner determine from data
    smooth_option = None
    smooth1d_option = None
    range_desc = "sample range"
    smooth_desc = "no smoothing"

print(f"Plot configuration: {smooth_desc}, range={range_desc}")

# Create corner plot
fig = corner.corner(
    samples,
    labels=labels,
    show_titles=True,
    title_kwargs={"fontsize": 12},
    label_kwargs={"fontsize": 14},
    quantiles=[0.16, 0.5, 0.84],
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2)),
    plot_density=False,
    plot_datapoints=True,
    fill_contours=True,
    bins=30,
    range=plot_range,
    color='blue',
    hist_kwargs={'color': 'blue'},
    smooth=smooth_option,
    smooth1d=smooth1d_option
)

# Add prior overlays if requested
if plot_priors and config is not None:
    # Get axes from corner plot
    axes = fig.get_axes()
    ndim = len(labels)
    
    # Prior overlay for each parameter (diagonal elements)
    for i in range(ndim):
        ax = axes[i * ndim + i]  # Diagonal axis
        
        # Get parameter name and x-range for prior
        param_name = labels[i]
        x_min, x_max = ax.get_xlim()
        x_range = np.linspace(x_min, x_max, 100)
        
        # Get prior PDF based on parameter label
        prior_pdf = None
        if 'log_{10} h_a' in param_name:  # log10_ha
            prior_pdf = get_log10_ha_prior_pdf(config, x_range)
        elif 'log_{10} \\gamma_a' in param_name:  # log10_gamma_a
            prior_pdf = get_log10_gamma_a_prior_pdf(config, x_range)
        elif 'log_{10} \\sigma_' in param_name:  # log10_sigma_p
            prior_pdf = get_log10_sigma_p_prior_pdf(config, x_range)
        elif 'log_{10} \\gamma_' in param_name:  # log10_gamma_p
            prior_pdf = get_log10_gamma_p_prior_pdf(config, x_range)
        elif 'EFAC' in param_name and 'log_{10}' not in param_name:  # EFAC
            prior_pdf = get_efac_prior_pdf(config, x_range)
        elif 'EQUAD' in param_name:  # EQUAD or log10_EQUAD
            prior_pdf = get_equad_prior_pdf(config, x_range)
        
        # Plot prior if available
        if prior_pdf is not None and np.any(prior_pdf > 0) and np.all(np.isfinite(prior_pdf)):
            # Scale prior to match histogram
            y_max = ax.get_ylim()[1]
            prior_max = np.max(prior_pdf)
            if prior_max > 0:
                prior_pdf_scaled = prior_pdf * y_max / prior_max * 0.8  # Scale to 80% of max
                ax.plot(x_range, prior_pdf_scaled, 'r--', linewidth=2, alpha=0.7, label='Prior')

# Add injection overlays if requested
print("TESTING")
print(plot_injections)
print(injection_data)
print(pulsar_name_to_index)
if plot_injections and injection_data and pulsar_name_to_index:
    # Get axes from corner plot
    axes = fig.get_axes()
    ndim = len(labels)
    
    # Add injection overlays for each parameter (diagonal elements)
    for i in range(ndim):
        ax = axes[i * ndim + i]  # Diagonal axis
        param_name = labels[i]
        print(f"Processing parameter {i}: {param_name}")
        
        # Determine which pulsar this parameter belongs to and parameter type
        injection_value = None
        
        # Extract pulsar index from parameter name 
        if 'EFAC' in param_name:
            print(f"Found EFAC parameter: {param_name}")
            # Extract pulsar index from label like "$\mathrm{EFAC}_{p,0}$"
            match = re.search(r'p,(\d+)', param_name)
            print(f"Regex match: {match}")
            if match:
                pulsar_idx = int(match.group(1))
                print(f"Pulsar index: {pulsar_idx}")
                # Find pulsar name by index
                pulsar_name = None
                for name, idx in pulsar_name_to_index.items():
                    if idx == pulsar_idx:
                        pulsar_name = name
                        break
                print(f"Pulsar name: {pulsar_name}")
                
                if pulsar_name and pulsar_name in injection_data:
                    injection_value = injection_data[pulsar_name]['efac']
                    print(f"EFAC injection value: {injection_value}")
                    
        elif 'EQUAD' in param_name:
            print(f"Found EQUAD parameter: {param_name}")
            # Extract pulsar index from label like "$\log_{10} \mathrm{EQUAD}_{p,0}$"
            match = re.search(r'p,(\d+)', param_name)
            print(f"Regex match: {match}")
            if match:
                pulsar_idx = int(match.group(1))
                print(f"Pulsar index: {pulsar_idx}")
                # Find pulsar name by index
                pulsar_name = None
                for name, idx in pulsar_name_to_index.items():
                    if idx == pulsar_idx:
                        pulsar_name = name
                        break
                print(f"Pulsar name: {pulsar_name}")
                
                if pulsar_name and pulsar_name in injection_data:
                    # Injection EQUAD values are already in log10 space
                    injection_value = injection_data[pulsar_name]['equad']
                    print(f"EQUAD injection value: {injection_value}")
        
        # Plot injection value if available
        if injection_value is not None:
            print(f"Plotting injection value: {injection_value}")
            y_max = ax.get_ylim()[1]
            ax.axvline(injection_value, color='green', linestyle='-', linewidth=2, 
                      alpha=0.8, label='Injection')
        else:
            print(f"No injection value for parameter: {param_name}")
    

# Add title
log10_gamma_a_text = " + log10_gamma_a" if plot_log10_gamma_a and 'log10_gamma_a' in posterior.data_vars else ""
efac_text = " + EFAC" if plot_efac and 'efac' in posterior.data_vars else ""
equad_text = " + EQUAD" if plot_equad and ('log10_equad' in posterior.data_vars or 'equad' in posterior.data_vars) else ""
fig.suptitle(f'Run {run_index} Parameter Posterior Distributions\n(log10_ha{log10_gamma_a_text}{efac_text}{equad_text} + {num_pulsars} Pulsars)', 
             fontsize=16, y=0.98)

# Ensure plots directory exists
plots_dir = f"numpyro_test_{run_index}/plots"
os.makedirs(plots_dir, exist_ok=True)

# Save the plot with appropriate suffix
smooth_suffix = f"_smooth{smooth_sigma}" if smooth_sigma is not None else ""
log10_gamma_a_suffix = "_log10_gamma_a" if plot_log10_gamma_a and 'log10_gamma_a' in posterior.data_vars else ""
efac_suffix = "_efac" if plot_efac and 'efac' in posterior.data_vars else ""
equad_suffix = "_equad" if plot_equad and ('log10_equad' in posterior.data_vars or 'equad' in posterior.data_vars) else ""
priors_suffix = "_priors" if plot_priors else ""
injections_suffix = "_injections" if plot_injections else ""
output_file = f"{plots_dir}/corner_plot_run_{run_index}_{num_pulsars}pulsars{smooth_suffix}{log10_gamma_a_suffix}{efac_suffix}{equad_suffix}{priors_suffix}{injections_suffix}.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Corner plot saved to: {output_file}")

plt.show()