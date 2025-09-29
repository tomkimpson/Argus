#!/usr/bin/env python3
"""
Create corner plot from NUMPYRO results showing log10_ha and 
sigma_p, gamma_p parameters for selected pulsars.

This script supports:
- Optional inclusion of log10_gamma_a parameter (when sampled)
- Optional inclusion of EFAC and EQUAD parameters for individual pulsars
- Professional plotting with scientific styling
- Various smoothing options for better visualization

Usage: python create_corner_plot.py --results-file <path> <num_pulsars> [smooth_sigma] [options]

Examples:
  python create_corner_plot.py --results-file results.nc 2                         # Basic plot
  python create_corner_plot.py --results-file results.nc 2 1.0                     # With smoothing
  python create_corner_plot.py --results-file results.nc 2 0.5 --plot_log10_gamma_a  # Include log10_gamma_a
  python create_corner_plot.py --results-file results.nc 2 --efac --equad          # Include EFAC and EQUAD
  python create_corner_plot.py --results-file results.nc 2 1.0 --plot_log10_gamma_a --efac --equad  # Full featured

Arguments:
  --results-file: Path to NumPyro results file (.nc)
  num_pulsars   : Number of pulsars to include in the plot
  smooth_sigma  : Optional smoothing parameter for histograms
  --plot_log10_gamma_a: Include log10_gamma_a parameter if available in posterior
  --efac        : Include EFAC parameters for each pulsar
  --equad       : Include EQUAD parameters for each pulsar

Features:
- Professional corner plots with publication-quality formatting
- Automatic parameter detection and labeling
- Support for multiple pulsar parameters (sigma_p, gamma_p)
- Optional EFAC and EQUAD parameters for timing noise analysis
- Flexible smoothing and visualization options
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import corner
import os


def setup_professional_styling():
    """Set up professional plotting style similar to utils.py corner_plot function."""
    try:
        plt.style.use(["science", "no-latex"])  # Professional styling
    except OSError:
        # Fallback if scienceplots not available
        plt.style.use("default")
        print("Warning: scienceplots package not found. Using default matplotlib style.")

    # Professional plot parameters
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










# Parse command line arguments
parser = argparse.ArgumentParser(description='Create corner plot from NUMPYRO results')
parser.add_argument('--results-file', type=str, required=True, help='Path to NumPyro results file (.nc)')
parser.add_argument('num_pulsars', type=int, help='Number of pulsars to include')
parser.add_argument('smooth_sigma', type=float, nargs='?', default=None, help='Smoothing parameter for histograms')
parser.add_argument('--plot_log10_gamma_a', action='store_true', help='Include log10_gamma_a parameter in the plot')
parser.add_argument('--efac', action='store_true', help='Include EFAC parameters for each pulsar')
parser.add_argument('--equad', action='store_true', help='Include EQUAD parameters for each pulsar')

args = parser.parse_args()
results_file = args.results_file
num_pulsars = args.num_pulsars
smooth_sigma = args.smooth_sigma
plot_log10_gamma_a = args.plot_log10_gamma_a
plot_efac = args.efac
plot_equad = args.equad


# Load the results
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
    # Always use log10_equad if available, otherwise convert equad to log10
    if 'log10_equad' in posterior.data_vars:
        for i, pulsar_idx in enumerate(pulsar_indices):
            log10_equad = posterior['log10_equad'].isel(log10_equad_dim_0=pulsar_idx).values.flatten()
            samples_list.append(log10_equad)
            labels.append(rf'$\log_{{10}} \mathrm{{EQUAD}}_{{p,{i}}}$')
        print(f"Added log10_EQUAD parameters for {num_pulsars} pulsars")
    elif 'equad' in posterior.data_vars:
        for i, pulsar_idx in enumerate(pulsar_indices):
            equad = posterior['equad'].isel(equad_dim_0=pulsar_idx).values.flatten()
            log10_equad = np.log10(equad)  # Convert to log10
            samples_list.append(log10_equad)
            labels.append(rf'$\log_{{10}} \mathrm{{EQUAD}}_{{p,{i}}}$')
        print(f"Added log10_EQUAD parameters (converted from EQUAD) for {num_pulsars} pulsars")
    else:
        print(f"Warning: EQUAD requested but neither 'log10_equad' nor 'equad' found in posterior. Skipping EQUAD.")

# Combine all parameters
samples = np.column_stack(samples_list)

print(f"Sample shape: {samples.shape}")
print(f"Parameter ranges:")
for i, label in enumerate(labels):
    print(f"  {label}: [{samples[:, i].min():.3f}, {samples[:, i].max():.3f}]")


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

# Add EQUAD ranges if plotting (always log10 scale)
if plot_equad:
    for i in range(num_pulsars):
        prior_ranges.append([-9.5, -4.5])  # log10_EQUAD - extended from [-9.0, -5.0]

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

# Set up professional styling
setup_professional_styling()

# Professional color scheme
posterior_color = "#2E86C1"  # Professional blue
contour_colors = ["#AED6F1", "#5DADE2", "#2E86C1"]  # Gradient blues

# Create corner plot
fig = corner.corner(
    samples,
    labels=labels,
    show_titles=True,
    title_kwargs={"fontsize": 10, "fontweight": "bold", "pad": 10},
    label_kwargs={"fontsize": 16, "fontweight": "bold"},
    title_fmt=".3f",
    bins=40,  # Increased for better resolution
    quantiles=[0.16, 0.5, 0.84],  # 68% credible intervals
    plot_density=True,
    plot_datapoints=False,  # Clean look without individual points
    fill_contours=True,
    range=plot_range,
    color=posterior_color,
    contour_kwargs={"colors": contour_colors, "linewidths": 0.5},
    max_n_ticks=4,
    use_math_text=True,
    smooth=smooth_option,
    smooth1d=smooth1d_option
)

# Post-processing improvements to enhance axis appearance
axes = fig.get_axes()

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




# Adjust layout with professional spacing
plt.tight_layout()

# Manually adjust layout with reduced spacing
plt.subplots_adjust(
    hspace=0.05,  # Reduce vertical spacing between subplots
    wspace=0.05,  # Reduce horizontal spacing between subplots
)

# Save in the same directory as the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Save the plot
output_file = 'example_corner_plot.png' 
# Save PNG
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

print(f"Corner plot saved to: {output_file}")

plt.show()