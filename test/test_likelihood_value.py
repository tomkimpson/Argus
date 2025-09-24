"""Unit tests for likelihood evaluation in the Argus package."""

import os
import pytest
import jax.numpy as jnp

from argus import data_loader
from argus import jax_kalman_filter as jk
from argus import bayesian_inference
from argus import io_manager
from argus.utils import get_efac_equad_injections, get_psr_noise_injections


def test_likelihood_value():
    """
    Tests the likelihood evaluation is correct. The "correct" value is inserted by hand to ensure consistency between edits
    """
    # Initialize logger for testing (without file logging)
    class MockConfig:
        def get(self, section, key, fallback=None):
            return fallback

    config = MockConfig()
    io_manager.setup_single_logger(config, enable_file_logging=False)

    # Get the data directory path relative to test file location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(script_dir, "../data/IPTA_MockDataChallenge2/dataset_2b/")

    # Skip test if data directory doesn't exist
    if not os.path.exists(directory):
        pytest.skip(f"Data directory not found: {directory}")

    # Get the data
    pulsar_data = data_loader.LoadWidebandPulsarData.get_processed_residuals(
        directory, excluded_psrs=['J1640+2224']
    )

    # Get noise parameters from test data directory
    noise_params_path = os.path.join(script_dir, "data/noise_parameters.json")
    spin_injections_path = os.path.join(script_dir, "data/spin_injections.pkl")

    # Skip if noise parameter files don't exist
    if not os.path.exists(noise_params_path) or not os.path.exists(spin_injections_path):
        pytest.skip(f"Noise parameter files not found")

    # Get efac and equad
    efac_array, equad_array = get_efac_equad_injections(noise_params_path, excluded_psrs=['J1640+2224'])

    # Get psr noise
    sigma_p_injected, gamma_p_injected = get_psr_noise_injections(spin_injections_path, excluded_psrs=['J1640+2224'])

    # Initialize the Kalman filter
    KF = jk.JaxKalmanFilter(data=pulsar_data, use_gw=True)

    # Set GW parameters
    γa = 1e-9
    ha = 1e-15

    # Set the parameters
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

    # Calculate likelihood
    log_likelihood = KF.get_likelihood(params)

    print("the computed log likelihood is:", log_likelihood)

    # Assert expected value - using a placeholder value that should be updated
    # based on actual test runs with the specific dataset
    expected_likelihood = 55963.86  # Approximate expected value

    # Use relative tolerance for floating point comparison
    assert abs(log_likelihood - expected_likelihood) < 1.0, f"Expected ~{expected_likelihood}, got {log_likelihood}"