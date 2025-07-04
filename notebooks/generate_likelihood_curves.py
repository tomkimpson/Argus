#!/usr/bin/env python3
"""
Generate likelihood curves for all parameters and save results to disk.

This script creates likelihood curves for each parameter in the model, holding 
all other parameters constant. For vector parameters (like γ_p and σ_p), it 
analyzes a user-specified pulsar index.

The script outputs:
- Individual parameter likelihood curves (plots)
- Summary comparison plot
- Numerical data files (CSV format)
- Analysis report (text file)
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import configparser
from datetime import datetime
import argparse

conflicting_path = '/fred/oz022/tkimpson/clean/Argus/python'
if conflicting_path in sys.path:
    sys.path.remove(conflicting_path)
    print(f"Removed conflicting path: {conflicting_path}")


# Add the python directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python'))

from argus import workflow, bayesian_inference, utils


def fix_config_paths(config, config_path):
    """Fix relative paths in config to be relative to the project root, not config file location."""
    import configparser
    
    # The config paths are designed to work from python/argus/, so we need to find the project root
    # From config file at python/argus/configs/file.ini, go up to project root
    config_dir = os.path.dirname(os.path.abspath(config_path))  # python/argus/configs
    argus_dir = os.path.dirname(config_dir)  # python/argus
    python_dir = os.path.dirname(argus_dir)  # python
    project_root = os.path.dirname(python_dir)  # Argus (project root)
    
    print(f"Config file directory: {config_dir}")
    print(f"Argus module directory: {argus_dir}")
    print(f"Python directory: {python_dir}")
    print(f"Project root: {project_root}")
    
    # Create a new config with corrected paths
    fixed_config = configparser.ConfigParser()
    fixed_config.read_dict(dict(config))
    
    # Fix the data paths by resolving them from the argus directory (where they're designed to work)
    def fix_path(section, key):
        if fixed_config.has_option(section, key):
            relative_path = fixed_config.get(section, key)
            print(f"Original {key}: {relative_path}")
            if not os.path.isabs(relative_path):  # Only fix if it's relative
                # Resolve path as if we were in the argus directory (where config was designed to work)
                absolute_path = os.path.abspath(os.path.join(argus_dir, relative_path))
                print(f"Fixed {key}: {absolute_path}")
                print(f"Path exists: {os.path.exists(absolute_path)}")
                fixed_config.set(section, key, absolute_path)
            else:
                print(f"{key} is already absolute: {relative_path}")
    
    # Fix the known relative paths
    fix_path('Data', 'data_path')
    fix_path('Data', 'noise_params_path')
    fix_path('Data', 'spin_injections_path')

    # Check what files are actually in the data directory
    data_path = fixed_config.get('Data', 'data_path')
    if os.path.exists(data_path):
        files = os.listdir(data_path)
        par_files = [f for f in files if f.endswith('.par')]
        tim_files = [f for f in files if f.endswith('.tim')]
        print(f"Found in data directory: {len(files)} total files")
        print(f"Found {len(par_files)} .par files and {len(tim_files)} .tim files")
        if par_files:
            print(f"Sample .par files: {par_files[:3]}")
        if tim_files:
            print(f"Sample .tim files: {tim_files[:3]}")
    else:
        print(f"Data directory does not exist: {data_path}")

    print("fixed_config [Data] section:")
    for key, value in fixed_config['Data'].items():
        print(f"  {key} = {value}")
    
    return fixed_config


def setup_kalman_filter_from_config(config_path):
    """Set up Kalman filter using the same workflow as the main inference code."""
    from argus.io_manager import setup_single_logger
    
    # Load configuration and fix paths
    config = utils.load_config(config_path)
    print("calling fix_config_paths")
    config = fix_config_paths(config, config_path)

    print(config)
    
    # Initialize the centralized Argus logger system
    logger=setup_single_logger(config, 'likelihood_curves')
    
    # Use the actual workflow function
    pulsar_data, KF = workflow.setup_data_and_kalman_filter(config, logger=logger, use_gw=True)
    
    # Get noise parameters
    efac_array, equad_array, sigma_p_array, gamma_p_array = workflow.get_noise_parameters(config)
    
    return KF, pulsar_data, efac_array, equad_array, sigma_p_array, gamma_p_array, config


def create_baseline_parameters(gamma_p_array, sigma_p_array, efac_array, equad_array):
    """Create baseline parameter values for likelihood curves."""
    return bayesian_inference.Parameters(
        γa=1e-9,  # Fixed GW spectral index
        ha=10**(-15.51),  # GW amplitude baseline
        γp=gamma_p_array,  # Pulsar red noise gamma
        σp=sigma_p_array,  # Pulsar red noise sigma  
        EFAC=efac_array,   # Error factors
        EQUAD=equad_array  # Extra quadrature noise
    )


def likelihood_curve_ha(KF, baseline_params, ha_values, progress_callback=None):
    """Create likelihood curve for GW amplitude ha."""
    log_likelihoods = []

    print("Baseline params before modification:")
    print(baseline_params)

    for i, ha in enumerate(ha_values):
        if progress_callback and i % 10 == 0:
            progress_callback(f"ha curve: {i+1}/{len(ha_values)}")
            
        params = baseline_params.replace(ha=ha)
        ll = KF.get_likelihood(params)
        log_likelihoods.append(float(ll))
    
    return jnp.array(log_likelihoods)


def likelihood_curve_gamma_p(KF, baseline_params, gamma_p_values, pulsar_index, progress_callback=None):
    """Create likelihood curve for pulsar red noise gamma_p for a specific pulsar."""
    log_likelihoods = []

    print("Baseline params before modification:")
    print(baseline_params)

    for i, gamma_p in enumerate(gamma_p_values):
        if progress_callback and i % 10 == 0:
            progress_callback(f"γ_p[{pulsar_index}] curve: {i+1}/{len(gamma_p_values)}")
            
        # Modify only the specified pulsar's gamma_p
        gamma_p_modified = baseline_params.γp.at[pulsar_index].set(gamma_p)
        params = baseline_params.replace(γp=gamma_p_modified)
        ll = KF.get_likelihood(params)
        log_likelihoods.append(float(ll))
    
    return jnp.array(log_likelihoods)


def likelihood_curve_sigma_p(KF, baseline_params, sigma_p_values, pulsar_index, progress_callback=None):
    """Create likelihood curve for pulsar red noise sigma_p for a specific pulsar."""
    log_likelihoods = []

    print("Baseline params before modification:")
    print(baseline_params)

    for i, sigma_p in enumerate(sigma_p_values):
        if progress_callback and i % 10 == 0:
            progress_callback(f"σ_p[{pulsar_index}] curve: {i+1}/{len(sigma_p_values)}")
            
        # Modify only the specified pulsar's sigma_p
        gamma_p_modified = baseline_params.γp.at[pulsar_index].set(7.51e-8)

        sigma_p_modified = baseline_params.σp.at[pulsar_index].set(sigma_p)

        params = baseline_params.replace(σp=sigma_p_modified, γp=gamma_p_modified)
        ll = KF.get_likelihood(params)
        log_likelihoods.append(float(ll))
    
    return jnp.array(log_likelihoods)


def likelihood_curve_efac(KF, baseline_params, efac_values, pulsar_index, progress_callback=None):
    """Create likelihood curve for EFAC for a specific pulsar."""
    log_likelihoods = []

    print("Baseline params before modification:")
    print(baseline_params)

    for i, efac in enumerate(efac_values):
        if progress_callback and i % 10 == 0:
            progress_callback(f"EFAC[{pulsar_index}] curve: {i+1}/{len(efac_values)}")
            
        # Modify only the specified pulsar's EFAC
        efac_modified = baseline_params.EFAC.at[pulsar_index].set(efac)
        params = baseline_params.replace(EFAC=efac_modified)
        ll = KF.get_likelihood(params)
        log_likelihoods.append(float(ll))
    
    return jnp.array(log_likelihoods)


def likelihood_curve_equad(KF, baseline_params, equad_values, pulsar_index, progress_callback=None):
    """Create likelihood curve for EQUAD for a specific pulsar."""
    log_likelihoods = []
    

    print("Baseline params before modification:")
    print(baseline_params)

    for i, equad in enumerate(equad_values):
        if progress_callback and i % 10 == 0:
            progress_callback(f"EQUAD[{pulsar_index}] curve: {i+1}/{len(equad_values)}")
            
        # Modify only the specified pulsar's EQUAD
        equad_modified = baseline_params.EQUAD.at[pulsar_index].set(equad)
        params = baseline_params.replace(EQUAD=equad_modified)
        ll = KF.get_likelihood(params)
        log_likelihoods.append(float(ll))
    
    return jnp.array(log_likelihoods)


def save_likelihood_data(output_dir, param_name, param_values, log_likelihoods, pulsar_index=None):
    """Save likelihood curve data to CSV file."""
    data = {
        'parameter_value': np.array(param_values),
        'log_likelihood': np.array(log_likelihoods)
    }
    
    df = pd.DataFrame(data)
    
    if pulsar_index is not None:
        filename = f"likelihood_curve_{param_name}_pulsar{pulsar_index:02d}.csv"
    else:
        filename = f"likelihood_curve_{param_name}.csv"
    
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved data: {filename}")
    
    return filepath


def plot_likelihood_curve(param_values, log_likelihoods, param_name, param_label, 
                         baseline_value=None, use_logscale=True, pulsar_index=None):
    """Create and return a likelihood curve plot."""
    
    plt.figure(figsize=(10, 6))
    
    if use_logscale:
        plt.semilogx(param_values, log_likelihoods, linewidth=2)
    else:
        plt.plot(param_values, log_likelihoods, linewidth=2)
    
    plt.xlabel(param_label)
    plt.ylabel('Log Likelihood')
    
    if pulsar_index is not None:
        plt.title(f'Likelihood Curve for {param_name} - Pulsar {pulsar_index}')
    else:
        plt.title(f'Likelihood Curve for {param_name}')
    
    plt.grid(True, alpha=0.3)
    
    # Mark maximum
    max_idx = jnp.argmax(log_likelihoods)
    plt.axvline(param_values[max_idx], color='r', linestyle='--', alpha=0.7,
               label=f'Max at {param_values[max_idx]:.2e}')
    
    # Mark baseline if provided
    if baseline_value is not None:
        plt.axvline(baseline_value, color='orange', linestyle=':', alpha=0.7,
                   label=f'Baseline = {baseline_value:.2e}')
    
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf()


def create_summary_plot(results, pulsar_index, baseline_params):
    """Create a summary figure with all likelihood curves."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Likelihood Curves Summary (Pulsar {pulsar_index})', fontsize=16)
    
    # GW amplitude
    axes[0,0].semilogx(results['ha']['values'], results['ha']['likelihoods'], 'b-', linewidth=2)
    axes[0,0].set_xlabel('GW Amplitude ha')
    axes[0,0].set_ylabel('Log Likelihood')
    axes[0,0].set_title('GW Amplitude')
    axes[0,0].grid(True, alpha=0.3)
    max_idx = jnp.argmax(results['ha']['likelihoods'])
    axes[0,0].axvline(results['ha']['values'][max_idx], color='r', linestyle='--', alpha=0.7)
    axes[0,0].axvline(baseline_params.ha, color='orange', linestyle=':', alpha=0.7)
    
    # Gamma_p
    axes[0,1].semilogx(results['gamma_p']['values'], results['gamma_p']['likelihoods'], 'g-', linewidth=2)
    axes[0,1].set_xlabel(f'γ_p[{pulsar_index}] (s⁻¹)')
    axes[0,1].set_ylabel('Log Likelihood')
    axes[0,1].set_title(f'Pulsar {pulsar_index} γ_p')
    axes[0,1].grid(True, alpha=0.3)
    max_idx = jnp.argmax(results['gamma_p']['likelihoods'])
    axes[0,1].axvline(results['gamma_p']['values'][max_idx], color='r', linestyle='--', alpha=0.7)
    axes[0,1].axvline(baseline_params.γp[pulsar_index], color='orange', linestyle=':', alpha=0.7)
    
    # Sigma_p
    axes[0,2].semilogx(results['sigma_p']['values'], results['sigma_p']['likelihoods'], 'm-', linewidth=2)
    axes[0,2].set_xlabel(f'σ_p[{pulsar_index}]')
    axes[0,2].set_ylabel('Log Likelihood')
    axes[0,2].set_title(f'Pulsar {pulsar_index} σ_p')
    axes[0,2].grid(True, alpha=0.3)
    max_idx = jnp.argmax(results['sigma_p']['likelihoods'])
    axes[0,2].axvline(results['sigma_p']['values'][max_idx], color='r', linestyle='--', alpha=0.7)
    axes[0,2].axvline(baseline_params.σp[pulsar_index], color='orange', linestyle=':', alpha=0.7)
    
    # EFAC
    axes[1,0].plot(results['efac']['values'], results['efac']['likelihoods'], 'c-', linewidth=2)
    axes[1,0].set_xlabel(f'EFAC[{pulsar_index}]')
    axes[1,0].set_ylabel('Log Likelihood')
    axes[1,0].set_title(f'Pulsar {pulsar_index} EFAC')
    axes[1,0].grid(True, alpha=0.3)
    max_idx = jnp.argmax(results['efac']['likelihoods'])
    axes[1,0].axvline(results['efac']['values'][max_idx], color='r', linestyle='--', alpha=0.7)
    axes[1,0].axvline(baseline_params.EFAC[pulsar_index], color='orange', linestyle=':', alpha=0.7)
    
    # EQUAD
    axes[1,1].semilogx(results['equad']['values'], results['equad']['likelihoods'], 'y-', linewidth=2)
    axes[1,1].set_xlabel(f'EQUAD[{pulsar_index}]')
    axes[1,1].set_ylabel('Log Likelihood')
    axes[1,1].set_title(f'Pulsar {pulsar_index} EQUAD')
    axes[1,1].grid(True, alpha=0.3)
    max_idx = jnp.argmax(results['equad']['likelihoods'])
    axes[1,1].axvline(results['equad']['values'][max_idx], color='r', linestyle='--', alpha=0.7)
    axes[1,1].axvline(baseline_params.EQUAD[pulsar_index], color='orange', linestyle=':', alpha=0.7)
    
    # Analysis notes
    axes[1,2].text(0.1, 0.5, 
                   f'Analysis for Pulsar Index: {pulsar_index}\n\n'
                   f'Total Pulsars: {len(results["pulsar_names"])}\n\n'
                   'Red dashed lines show\nmaximum likelihood values\n\n'
                   'Orange dotted lines show\nbaseline injection values\n\n'
                   'Individual plots and data\nsaved to output directory',
                   transform=axes[1,2].transAxes, fontsize=12,
                   verticalalignment='center')
    axes[1,2].set_xticks([])
    axes[1,2].set_yticks([])
    axes[1,2].set_title('Analysis Notes')
    
    plt.tight_layout()
    return fig


def generate_analysis_report(results, pulsar_index, baseline_params, output_file):
    """Generate a text report of the analysis results."""
    
    with open(output_file, 'w') as f:
        f.write("LIKELIHOOD CURVE ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Analyzed pulsar index: {pulsar_index}\n")
        f.write(f"Total number of pulsars: {len(results['pulsar_names'])}\n\n")
        
        f.write("BASELINE PARAMETER VALUES\n")
        f.write("-" * 25 + "\n")
        f.write(f"GW amplitude (ha): {float(baseline_params.ha):.2e}\n")
        f.write(f"GW spectral index (γa): {float(baseline_params.γa):.2e}\n")
        f.write(f"γ_p[{pulsar_index}]: {float(baseline_params.γp[pulsar_index]):.2e}\n")
        f.write(f"σ_p[{pulsar_index}]: {float(baseline_params.σp[pulsar_index]):.2e}\n")
        f.write(f"EFAC[{pulsar_index}]: {float(baseline_params.EFAC[pulsar_index]):.3f}\n")
        f.write(f"EQUAD[{pulsar_index}]: {float(baseline_params.EQUAD[pulsar_index]):.2e}\n\n")
        
        f.write("MAXIMUM LIKELIHOOD ESTIMATES\n")
        f.write("-" * 30 + "\n")
        
        for param_name, data in results.items():
            if param_name == 'pulsar_names':
                continue
                
            values = data['values']
            likelihoods = data['likelihoods']
            max_idx = jnp.argmax(likelihoods)
            max_value = values[max_idx]
            max_likelihood = likelihoods[max_idx]
            
            if param_name == 'ha':
                f.write(f"GW amplitude (ha): {max_value:.2e} (LL: {max_likelihood:.2f})\n")
            elif param_name == 'gamma_p':
                f.write(f"γ_p[{pulsar_index}]: {max_value:.2e} (LL: {max_likelihood:.2f})\n")
            elif param_name == 'sigma_p':
                f.write(f"σ_p[{pulsar_index}]: {max_value:.2e} (LL: {max_likelihood:.2f})\n")
            elif param_name == 'efac':
                f.write(f"EFAC[{pulsar_index}]: {max_value:.3f} (LL: {max_likelihood:.2f})\n")
            elif param_name == 'equad':
                f.write(f"EQUAD[{pulsar_index}]: {max_value:.2e} (LL: {max_likelihood:.2f})\n")
        
        f.write(f"\nPULSAR INFORMATION\n")
        f.write("-" * 18 + "\n")
        for i, psr_name in enumerate(results['pulsar_names']):
            f.write(f"Index {i:2d}: {psr_name}\n")
    
    print(f"Saved analysis report: {os.path.basename(output_file)}")


def main():
    parser = argparse.ArgumentParser(description='Generate likelihood curves for parameter estimation')
    parser.add_argument('--config', '-c', 
                       default='../python/argus/configs/config_likelihoods.ini',
                       help='Path to configuration file')
    parser.add_argument('--pulsar-index', '-p', type=int, default=5,
                       help='Pulsar index to analyze (default: 5)')
    parser.add_argument('--output-dir', '-o', 
                       default=None,
                       help='Output directory (default: auto-generated timestamp)')
    parser.add_argument('--n-points', '-n', type=int, default=50,
                       help='Number of points for each likelihood curve (default: 50)')
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"likelihood_curves_{timestamp}"
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Setup progress callback
    def progress_callback(msg):
        print(f"  {msg}")
    
    print("\n=== LIKELIHOOD CURVE GENERATION ===")
    print("Setting up Kalman filter and loading data...")
    
    # Setup model
    KF, pulsar_data, efac_array, equad_array, sigma_p_array, gamma_p_array, config = setup_kalman_filter_from_config(args.config)
    n_pulsars = len(pulsar_data['metadata'])
    
    print(f"Loaded data for {n_pulsars} pulsars")
    print(f"Analyzing pulsar index: {args.pulsar_index}")
    
    if args.pulsar_index >= n_pulsars:
        print(f"Error: pulsar index {args.pulsar_index} >= number of pulsars ({n_pulsars})")
        return
    
    # Get pulsar names
    pulsar_names = [row['name'] for _, row in pulsar_data['metadata'].iterrows()]
    print(f"Selected pulsar: {pulsar_names[args.pulsar_index]}")
    
    # Create baseline parameters
    baseline_params = create_baseline_parameters(gamma_p_array, sigma_p_array, efac_array, equad_array)
    
    # Test likelihood evaluation
    print("\nTesting likelihood evaluation...")
    ll_test = KF.get_likelihood(baseline_params)
    ll_test.block_until_ready()
    print(f"Baseline log likelihood: {float(ll_test):.2f}")
    
    # Store results
    results = {'pulsar_names': pulsar_names}
    
    print(f"\nGenerating likelihood curves with {args.n_points} points each...")
    
    # 1. GW amplitude
    print("1. GW amplitude (ha)")
    print(f"   Baseline ha: {baseline_params.ha:.2e}")
    ha_min, ha_max = 1e-17, 1e-14
    ha_values = jnp.logspace(jnp.log10(ha_min), jnp.log10(ha_max), args.n_points)
    ll_ha = likelihood_curve_ha(KF, baseline_params, ha_values, progress_callback)
    results['ha'] = {'values': ha_values, 'likelihoods': ll_ha}
    
    # Save data and plot
    save_likelihood_data(output_dir, 'ha', ha_values, ll_ha)
    fig = plot_likelihood_curve(ha_values, ll_ha, 'GW Amplitude', 'GW Amplitude ha', 
                               baseline_value=baseline_params.ha)
    fig.savefig(os.path.join(output_dir, 'likelihood_curve_ha.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 2. Pulsar red noise gamma_p
    print(f"2. Pulsar {args.pulsar_index} gamma_p")
    print(f"   Baseline γ_p[{args.pulsar_index}]: {baseline_params.γp[args.pulsar_index]:.2e}")
    gamma_p_baseline = gamma_p_array[args.pulsar_index]
    gamma_p_min = gamma_p_baseline * 0.01
    gamma_p_max = gamma_p_baseline * 100
    gamma_p_values = jnp.logspace(jnp.log10(gamma_p_min), jnp.log10(gamma_p_max), args.n_points)
    ll_gamma_p = likelihood_curve_gamma_p(KF, baseline_params, gamma_p_values, args.pulsar_index, progress_callback)
    results['gamma_p'] = {'values': gamma_p_values, 'likelihoods': ll_gamma_p}
    
    # Save data and plot
    save_likelihood_data(output_dir, 'gamma_p', gamma_p_values, ll_gamma_p, args.pulsar_index)
    fig = plot_likelihood_curve(gamma_p_values, ll_gamma_p, 'γ_p', f'γ_p[{args.pulsar_index}] (s⁻¹)', 
                               baseline_value=gamma_p_baseline, pulsar_index=args.pulsar_index)
    fig.savefig(os.path.join(output_dir, f'likelihood_curve_gamma_p_pulsar{args.pulsar_index:02d}.png'), 
               dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 3. Pulsar red noise sigma_p
    print(f"3. Pulsar {args.pulsar_index} sigma_p")
    print(f"   Baseline σ_p[{args.pulsar_index}]: {baseline_params.σp[args.pulsar_index]:.2e}")
    # Use the same range as the prior: 1e-18 to 1e-12
    sigma_p_min = 1e-18
    sigma_p_max = 1e-12
    sigma_p_baseline = sigma_p_array[args.pulsar_index]
    sigma_p_values = jnp.logspace(jnp.log10(sigma_p_min), jnp.log10(sigma_p_max), args.n_points)
    ll_sigma_p = likelihood_curve_sigma_p(KF, baseline_params, sigma_p_values, args.pulsar_index, progress_callback)
    results['sigma_p'] = {'values': sigma_p_values, 'likelihoods': ll_sigma_p}
    
    # Save data and plot
    save_likelihood_data(output_dir, 'sigma_p', sigma_p_values, ll_sigma_p, args.pulsar_index)
    fig = plot_likelihood_curve(sigma_p_values, ll_sigma_p, 'σ_p', f'σ_p[{args.pulsar_index}]', 
                               baseline_value=sigma_p_baseline, pulsar_index=args.pulsar_index)
    fig.savefig(os.path.join(output_dir, f'likelihood_curve_sigma_p_pulsar{args.pulsar_index:02d}.png'), 
               dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 4. EFAC
    print(f"4. Pulsar {args.pulsar_index} EFAC")
    print(f"   Baseline EFAC[{args.pulsar_index}]: {baseline_params.EFAC[args.pulsar_index]:.3f}")
    efac_baseline = efac_array[args.pulsar_index]
    efac_min = max(0.1, efac_baseline * 0.1)
    efac_max = efac_baseline * 5.0
    efac_values = jnp.linspace(efac_min, efac_max, args.n_points)
    ll_efac = likelihood_curve_efac(KF, baseline_params, efac_values, args.pulsar_index, progress_callback)
    results['efac'] = {'values': efac_values, 'likelihoods': ll_efac}
    
    # Save data and plot
    save_likelihood_data(output_dir, 'efac', efac_values, ll_efac, args.pulsar_index)
    fig = plot_likelihood_curve(efac_values, ll_efac, 'EFAC', f'EFAC[{args.pulsar_index}]', 
                               baseline_value=efac_baseline, use_logscale=False, pulsar_index=args.pulsar_index)
    fig.savefig(os.path.join(output_dir, f'likelihood_curve_efac_pulsar{args.pulsar_index:02d}.png'), 
               dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 5. EQUAD
    print(f"5. Pulsar {args.pulsar_index} EQUAD")
    print(f"   Baseline EQUAD[{args.pulsar_index}]: {baseline_params.EQUAD[args.pulsar_index]:.2e}")
    equad_baseline = equad_array[args.pulsar_index]
    equad_min = equad_baseline * 0.01
    equad_max = equad_baseline * 100
    equad_values = jnp.logspace(jnp.log10(equad_min), jnp.log10(equad_max), args.n_points)
    ll_equad = likelihood_curve_equad(KF, baseline_params, equad_values, args.pulsar_index, progress_callback)
    results['equad'] = {'values': equad_values, 'likelihoods': ll_equad}
    
    # Save data and plot
    save_likelihood_data(output_dir, 'equad', equad_values, ll_equad, args.pulsar_index)
    fig = plot_likelihood_curve(equad_values, ll_equad, 'EQUAD', f'EQUAD[{args.pulsar_index}]', 
                               baseline_value=equad_baseline, pulsar_index=args.pulsar_index)
    fig.savefig(os.path.join(output_dir, f'likelihood_curve_equad_pulsar{args.pulsar_index:02d}.png'), 
               dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 6. Create summary plot
    print("6. Creating summary plot...")
    fig = create_summary_plot(results, args.pulsar_index, baseline_params)
    fig.savefig(os.path.join(output_dir, 'likelihood_curves_summary.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 7. Generate analysis report
    print("7. Generating analysis report...")
    report_file = os.path.join(output_dir, 'analysis_report.txt')
    generate_analysis_report(results, args.pulsar_index, baseline_params, report_file)
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Results saved to: {output_dir}")
    print(f"Generated files:")
    print(f"  - Individual likelihood curve plots (PNG)")
    print(f"  - Summary comparison plot (PNG)")
    print(f"  - Numerical data files (CSV)")
    print(f"  - Analysis report (TXT)")


if __name__ == "__main__":
    print("Starting likelihood curve generation...")
    main()