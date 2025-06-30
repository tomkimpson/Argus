#!/usr/bin/env python3
"""
Create corner plot from NUMPYRO results showing log10_ha and 
sigma_p, gamma_p parameters for selected pulsars.

Usage: python create_corner_plot.py <run_index> <num_pulsars>
Example: python create_corner_plot.py 016 2
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import corner
import os

# Parse command line arguments
if len(sys.argv) != 3:
    print("Usage: python create_corner_plot.py <run_index> <num_pulsars>")
    print("Example: python create_corner_plot.py 016 2")
    sys.exit(1)

run_index = sys.argv[1]
num_pulsars = int(sys.argv[2])

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

# Extract sigma_p and gamma_p for selected pulsars
for i, pulsar_idx in enumerate(pulsar_indices):
    sigma_p = posterior['log10_σp'].isel(log10_σp_dim_0=pulsar_idx).values.flatten()
    gamma_p = posterior['log10_γp'].isel(log10_γp_dim_0=pulsar_idx).values.flatten()
    
    samples_list.append(sigma_p)
    samples_list.append(gamma_p)
    
    labels.append(rf'$\log_{{10}} \sigma_{{p,{i}}}$')
    labels.append(rf'$\log_{{10}} \gamma_{{p,{i}}}$')

# Combine all parameters
samples = np.column_stack(samples_list)

print(f"Sample shape: {samples.shape}")
print(f"Parameter ranges:")
for i, label in enumerate(labels):
    print(f"  {label}: [{samples[:, i].min():.3f}, {samples[:, i].max():.3f}]")

# Define prior ranges from config file
prior_ranges = [[-16.0, -14.0]]  # log10_ha
for i in range(num_pulsars):
    prior_ranges.append([-18.0, -12.0])  # log10_sigma_p
    prior_ranges.append([-11.0, -6.0])   # log10_gamma_p

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
    range=prior_ranges,
    color='blue',
    hist_kwargs={'color': 'blue'},
    smooth=True,
    smooth1d=True
)

# Add title
fig.suptitle(f'Run {run_index} Parameter Posterior Distributions\n(log10_ha + {num_pulsars} Pulsars)', 
             fontsize=16, y=0.98)

# Ensure plots directory exists
plots_dir = f"numpyro_test_{run_index}/plots"
os.makedirs(plots_dir, exist_ok=True)

# Save the plot
output_file = f"{plots_dir}/corner_plot_run_{run_index}_{num_pulsars}pulsars.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Corner plot saved to: {output_file}")

plt.show()