#!/usr/bin/env python3
"""
Debug version of the diffuse Kalman filter test with verbose output.
"""

import os
import sys
import pickle
import jax.numpy as jnp
import jax

# Add the python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../python'))

from argus.jax_kalman_filter import JaxKalmanFilter
from argus.io_manager import setup_single_logger
from argus import bayesian_inference
from argus.utils import get_efac_equad_injections, get_psr_noise_injections

# Enable JAX debug output
jax.config.update("jax_disable_jit", False)  # Keep JIT for performance but enable debug

# Create a minimal config for logging
class MockConfig:
    def get(self, section, key, fallback=None):
        return fallback

config = MockConfig()
setup_single_logger(config, enable_file_logging=False)

def load_test_data():
    """Load existing test data like test_likelihood_value.py does."""

    # Get the preprocessed data file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessed_data_path = os.path.join(script_dir, "data/processed_pulsar_data.pkl")

    # Check if preprocessed data exists
    if not os.path.exists(preprocessed_data_path):
        print(f"Preprocessed pulsar data not found: {preprocessed_data_path}")
        print("Run 'python test/get_pulsar_data_for_testing.py' to generate it.")
        return None, None

    # Load the preprocessed pulsar data
    with open(preprocessed_data_path, 'rb') as f:
        pulsar_data = pickle.load(f)

    # Get noise parameters from test data directory
    noise_params_path = os.path.join(script_dir, "data/noise_parameters.json")
    spin_injections_path = os.path.join(script_dir, "data/spin_injections.pkl")

    # Check if noise parameter files exist
    if not os.path.exists(noise_params_path) or not os.path.exists(spin_injections_path):
        print("Noise parameter files not found")
        return None, None

    # Get efac and equad
    efac_array, equad_array = get_efac_equad_injections(noise_params_path, excluded_psrs=['J1640+2224'])

    # Get psr noise
    sigma_p_injected, gamma_p_injected = get_psr_noise_injections(spin_injections_path, excluded_psrs=['J1640+2224'])

    # Create parameters like in test_likelihood_value.py
    γa = 1e-9
    ha = 1e-15

    params = bayesian_inference.Parameters(
        # GW parameters
        γa=γa,
        ha=ha,
        log10_gamma_a=jnp.log10(γa),  # Add required log10 parameter

        # Spin parameters
        γp=gamma_p_injected,
        σp=sigma_p_injected,

        # Measurement noise parameters
        EFAC=efac_array,
        EQUAD=equad_array
    )

    return pulsar_data, params

def debug_diffuse_filter():
    """Debug the diffuse filter to identify NaN sources."""

    print("Loading test data...")
    data, params = load_test_data()

    if data is None or params is None:
        print("❌ Could not load test data. Skipping test.")
        return False

    # Print some basic info about the data
    print(f"Number of observations: {data['processed_residuals']['residuals'].shape[0]}")
    print(f"Number of pulsars: {data['processed_residuals']['residuals'].shape[1]}")
    print(f"Residuals shape: {data['processed_residuals']['residuals'].shape}")
    print(f"Errors shape: {data['processed_residuals']['errors'].shape}")

    # Check for any NaNs or infs in input data
    residuals = data['processed_residuals']['residuals']
    errors = data['processed_residuals']['errors']
    print(f"Input residuals finite: {jnp.all(jnp.isfinite(residuals))}")
    print(f"Input errors finite: {jnp.all(jnp.isfinite(errors))}")
    print(f"Input errors positive: {jnp.all(errors > 0)}")

    print("\nTesting diffuse Kalman filter with debugging...")
    try:
        kf_diffuse = JaxKalmanFilter(data, use_gw=True, use_diffuse=True)

        # Print filter dimensions
        print(f"Diffuse filter state dimension: {kf_diffuse.nx}")
        print(f"Number of timing parameters: {kf_diffuse.n_timing_params}")
        print(f"H matrix shape: {kf_diffuse.jax_H_matrices.shape}")
        print(f"M matrix shape: {kf_diffuse.jax_M_matrices.shape}")

        ll_diffuse = kf_diffuse.get_likelihood(params)
        print(f"Diffuse filter log-likelihood: {ll_diffuse}")
        print(f"Likelihood is finite: {jnp.isfinite(ll_diffuse)}")

        return jnp.isfinite(ll_diffuse)

    except Exception as e:
        print(f"Diffuse filter failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_diffuse_filter()
    if success:
        print("\n✅ Debug test passed!")
    else:
        print("\n❌ Debug test failed!")