"""Shared pytest fixtures for argus tests."""

import pytest
import numpy as np
import pandas as pd
import jax.numpy as jnp
from unittest.mock import Mock
import configparser


@pytest.fixture
def mock_config():
    """Create a mock configuration object for testing."""
    config = configparser.ConfigParser()

    # Data section
    config.add_section("Data")
    config.set("Data", "data_path", "/path/to/data")
    config.set("Data", "excluded_psrs", "J1640+2224")

    # PriorModel section
    config.add_section("PriorModel")
    config.set("PriorModel", "log10_ha_fixed", "False")
    config.set("PriorModel", "log10_ha_min", "-16.0")
    config.set("PriorModel", "log10_ha_max", "-14.0")
    config.set("PriorModel", "log10_gamma_a_fixed", "False")
    config.set("PriorModel", "log10_gamma_a_min", "-10.0")
    config.set("PriorModel", "log10_gamma_a_max", "-8.0")
    config.set("PriorModel", "efac_min", "0.5")
    config.set("PriorModel", "efac_max", "2.0")
    config.set("PriorModel", "log10_equad_min", "-8.0")
    config.set("PriorModel", "log10_equad_max", "-6.0")
    config.set("PriorModel", "log10_gamma_p_mean_min", "-10.0")
    config.set("PriorModel", "log10_gamma_p_mean_max", "-7.0")
    config.set("PriorModel", "log10_gamma_p_std_min", "0.1")
    config.set("PriorModel", "log10_gamma_p_std_max", "2.0")
    config.set("PriorModel", "log10_ratio_mean_min", "-8.0")
    config.set("PriorModel", "log10_ratio_mean_max", "-6.0")
    config.set("PriorModel", "log10_ratio_std_min", "0.1")
    config.set("PriorModel", "log10_ratio_std_max", "2.0")
    config.set("PriorModel", "noise_params_path", "")
    config.set("PriorModel", "spin_injections_path", "")

    # NUTS section
    config.add_section("NUTS")
    config.set("NUTS", "num_samples", "100")
    config.set("NUTS", "num_warmup", "100")
    config.set("NUTS", "num_chains", "2")
    config.set("NUTS", "target_accept_prob", "0.95")
    config.set("NUTS", "max_tree_depth", "10")
    config.set("NUTS", "dense_mass", "False")

    # Output section
    config.add_section("Output")
    config.set("Output", "output_id", "test_run")
    config.set("Output", "base_dir", "test_output_{output_id}")

    # Logging section
    config.add_section("Logging")
    config.set("Logging", "level", "INFO")
    config.set("Logging", "enable_file_logging", "False")

    return config


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    return logger


@pytest.fixture
def sample_pulsar_metadata():
    """Create sample pulsar metadata for testing."""
    return pd.DataFrame(
        {
            "name": ["PSR_J0030+0451", "PSR_J0613-0200"],
            "dim_M": [5, 6],
            "RA": [0.5, 1.2],
            "DEC": [0.3, -0.1],
            "F0": [200.0, 150.0],
            "par_file": ["/path/to/J0030.par", "/path/to/J0613.par"],
            "tim_file": ["/path/to/J0030.tim", "/path/to/J0613.tim"],
        }
    )


@pytest.fixture
def sample_pulsar_residuals():
    """Create sample pulsar residual data for testing."""
    n_epochs = 10
    n_pulsars = 2

    return {
        "toas": np.linspace(0, 1000, n_epochs),
        "residuals": np.random.randn(n_epochs, n_pulsars) * 1e-6,
        "errors": np.ones((n_epochs, n_pulsars)) * 1e-7,
    }


@pytest.fixture
def sample_design_matrices():
    """Create sample design matrices for testing."""
    n_epochs = 10
    n_pulsars = 2
    dims = [5, 6]  # dimension for each pulsar

    matrices = []
    for i in range(n_pulsars):
        matrices.append(np.random.randn(n_epochs, dims[i]))

    return matrices


@pytest.fixture
def sample_covariance_matrices():
    """Create sample parameter covariance matrices for testing."""
    n_pulsars = 2
    dims = [5, 6]

    matrices = []
    for i in range(n_pulsars):
        # Create positive definite matrix
        A = np.random.randn(dims[i], dims[i])
        matrices.append(A @ A.T + np.eye(dims[i]) * 0.1)

    return matrices


@pytest.fixture
def sample_hd_correlation():
    """Create sample Hellings-Downs correlation matrix."""
    n_pulsars = 2
    # Simple correlation matrix
    return np.array([[1.0, 0.5], [0.5, 1.0]])


@pytest.fixture
def sample_pulsar_data(
    sample_pulsar_residuals,
    sample_pulsar_metadata,
    sample_design_matrices,
    sample_covariance_matrices,
    sample_hd_correlation,
):
    """Create complete sample pulsar data dictionary."""
    return {
        "processed_residuals": sample_pulsar_residuals,
        "metadata": sample_pulsar_metadata,
        "design_matrices": sample_design_matrices,
        "parameter_covariances": sample_covariance_matrices,
        "hd_correlation": sample_hd_correlation,
    }


@pytest.fixture
def sample_noise_parameters():
    """Create sample noise parameters for testing."""
    n_pulsars = 2
    return {
        "efac": jnp.ones(n_pulsars),
        "equad": jnp.full(n_pulsars, 1e-7),
        "gamma_p": jnp.full(n_pulsars, 1e-8),
        "sigma_p": jnp.full(n_pulsars, 1e-15),
    }


@pytest.fixture
def mock_enterprise_pulsar():
    """Create a mock Enterprise Pulsar object."""
    pulsar = Mock()
    pulsar.name = "PSR_J0030+0451"
    pulsar.toas = np.linspace(0, 1000, 10)
    pulsar.toaerrs = np.ones(10) * 1e-7
    pulsar.residuals = np.random.randn(10) * 1e-6
    pulsar.fitpars = ["F0", "F1", "RAJ", "DECJ", "DM"]
    pulsar.Mmat = np.random.randn(10, 5)
    pulsar._raj = 0.5
    pulsar._decj = 0.3
    return pulsar
