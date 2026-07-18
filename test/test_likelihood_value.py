"""Unit tests for likelihood evaluation in the Argus package."""

import os
import pytest
import pickle
import jax.numpy as jnp

from argus import data_loader
from argus import jax_kalman_filter as jk
from argus import bayesian_inference
from argus import io_manager
from argus.utils import get_efac_equad_injections, get_psr_noise_injections


def test_likelihood_value():
    """
    Tests the likelihood evaluation is correct. The "correct" value is inserted by hand to ensure consistency between edits during development.
    """

    # Initialize logger for testing (without file logging)
    class MockConfig:
        def get(self, section, key, fallback=None):
            return fallback

    config = MockConfig()
    io_manager.setup_single_logger(config, enable_file_logging=False)

    # Get the preprocessed data file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessed_data_path = os.path.join(script_dir, "data/processed_pulsar_data.pkl")

    # Skip test if preprocessed data doesn't exist
    if not os.path.exists(preprocessed_data_path):
        pytest.skip(
            f"Preprocessed pulsar data not found: {preprocessed_data_path}. "
            f"Run 'python test/get_pulsar_data_for_testing.py' to generate it."
        )

    # Load the preprocessed pulsar data
    with open(preprocessed_data_path, "rb") as f:
        pulsar_data = pickle.load(f)

    # Get noise parameters from test data directory
    noise_params_path = os.path.join(script_dir, "data/noise_parameters.json")
    spin_injections_path = os.path.join(script_dir, "data/spin_injections.pkl")

    # Skip if noise parameter files don't exist
    if not os.path.exists(noise_params_path) or not os.path.exists(
        spin_injections_path
    ):
        pytest.skip(f"Noise parameter files not found")

    # Get efac and equad
    efac_array, equad_array = get_efac_equad_injections(
        noise_params_path, excluded_psrs=["J1640+2224"]
    )

    # Get psr noise
    sigma_p_injected, gamma_p_injected = get_psr_noise_injections(
        spin_injections_path, excluded_psrs=["J1640+2224"]
    )

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
        EQUAD=equad_array,
    )

    # Assert expected value - a golden value recorded from an actual run on this
    # dataset (γa=1e-9, ha=1e-15, 32 MDC2 pulsars). Updated from 55963.86 after the
    # get_Q_block q11 fix (γ**3 -> γ**2): correcting the integrated-OU position-noise
    # normalization shifts the log-likelihood by ~7655 nats.
    expected_likelihood = 63618.93  # Golden value (post q11 fix)

    # Both the default sequential filter and the marginalized (Rao-Blackwellized)
    # timing-model filter must reproduce the golden value: they are mathematically
    # equivalent, differing only in whether the timing parameters are carried as state
    # or integrated out analytically.
    for use_marginal in (False, True):
        KF = jk.JaxKalmanFilter(
            data=pulsar_data, use_gw=True, use_marginal=use_marginal
        )
        log_likelihood = KF.get_likelihood(params)
        backend = "marginal" if use_marginal else "sequential"
        print(f"the computed log likelihood ({backend}) is:", log_likelihood)

        # Use relative tolerance for floating point comparison
        assert (
            abs(log_likelihood - expected_likelihood) < 1.0
        ), f"[{backend}] Expected ~{expected_likelihood}, got {log_likelihood}"

    # Diffuse (flat/improper) timing-model prior on the marginal filter. This is a
    # different likelihood from the informative-prior golden above (P_eps⁻¹ → 0 fully
    # projects out the timing-model subspace and drops a parameter-independent additive
    # constant), so it has its own recorded reference. The value is independently
    # validated in test_jax_kalman_filter.py::TestDiffuseFilter against a batch GLS /
    # G-matrix oracle and the α → ∞ limit of the informative filter.
    expected_diffuse_likelihood = 59420.06
    KF_diffuse = jk.JaxKalmanFilter(
        data=pulsar_data, use_gw=True, use_marginal=True, timing_prior="diffuse"
    )
    log_likelihood_diffuse = KF_diffuse.get_likelihood(params)
    print("the computed log likelihood (diffuse) is:", log_likelihood_diffuse)
    assert (
        abs(log_likelihood_diffuse - expected_diffuse_likelihood) < 1.0
    ), f"[diffuse] Expected ~{expected_diffuse_likelihood}, got {log_likelihood_diffuse}"
