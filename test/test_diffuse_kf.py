#!/usr/bin/env python3
"""
Test script to verify the diffuse Kalman filter implementation using existing test data.
"""

import os
import sys
import pickle
import jax.numpy as jnp

# Add the python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../python'))

from argus.jax_kalman_filter import JaxKalmanFilter
from argus.io_manager import setup_single_logger
from argus import bayesian_inference
from argus.utils import get_efac_equad_injections, get_psr_noise_injections

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

def test_diffuse_vs_standard():
    """Test that both standard and diffuse filters can be initialized and run."""

    print("Loading test data...")
    data, params = load_test_data()

    if data is None or params is None:
        print("❌ Could not load test data. Skipping test.")
        return False

    print("Testing standard Kalman filter...")
    try:
        kf_standard = JaxKalmanFilter(data, use_gw=True, use_diffuse=False)
        ll_standard = kf_standard.get_likelihood(params)
        print(f"Standard filter log-likelihood: {ll_standard}")
    except Exception as e:
        print(f"Standard filter failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("Testing diffuse Kalman filter...")
    try:
        kf_diffuse = JaxKalmanFilter(data, use_gw=True, use_diffuse=True)
        ll_diffuse = kf_diffuse.get_likelihood(params)
        print(f"Diffuse filter log-likelihood: {ll_diffuse}")
    except Exception as e:
        print(f"Diffuse filter failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("Both filters completed successfully!")
    

    # Check that the likelihoods are reasonable (finite numbers)
    if jnp.isfinite(ll_standard) and jnp.isfinite(ll_diffuse):
        print(f"Likelihood difference: {ll_diffuse - ll_standard}")
        return True
    else:
        print("Warning: Non-finite likelihoods detected")
        return False

if __name__ == "__main__":
    success = test_diffuse_vs_standard()
    if success:
        print("\n✅ Test passed! Diffuse Kalman filter implementation appears to be working.")
    else:
        print("\n❌ Test failed! There may be issues with the implementation.")