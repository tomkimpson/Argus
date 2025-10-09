"""Unit tests for utils module."""

import pytest
import os
import tempfile
import json
import pickle
import jax.numpy as jnp
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from argus import utils


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_valid_config(self, tmp_path):
        """Test loading a valid configuration file."""
        config_file = tmp_path / "test_config.ini"
        config_file.write_text("""[Data]
data_path = /path/to/data
excluded_psrs = J1640+2224

[Output]
output_id = test
""")

        config = utils.load_config(str(config_file))
        assert config.get("Data", "data_path") == "/path/to/data"
        assert config.get("Output", "output_id") == "test"

    def test_load_nonexistent_config(self):
        """Test that loading non-existent config raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            utils.load_config("/nonexistent/config.ini")


class TestResolveConfigPaths:
    """Tests for resolve_config_paths function."""

    def test_resolve_relative_paths(self, tmp_path):
        """Test resolving relative paths to absolute paths."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "config.ini"

        config_file.write_text("""[Data]
data_path = ../data

[PriorModel]
noise_params_path = params/noise.json
""")

        config = utils.load_config(str(config_file))
        resolved_config = utils.resolve_config_paths(config, str(config_file))

        # Paths should now be absolute
        data_path = resolved_config.get("Data", "data_path")
        assert os.path.isabs(data_path)

    def test_absolute_paths_unchanged(self, tmp_path):
        """Test that absolute paths remain unchanged."""
        config_file = tmp_path / "config.ini"
        abs_path = "/absolute/path/to/data"

        config_file.write_text(f"""[Data]
data_path = {abs_path}
""")

        config = utils.load_config(str(config_file))
        resolved_config = utils.resolve_config_paths(config, str(config_file))

        assert resolved_config.get("Data", "data_path") == abs_path


class TestGetNoiseParameters:
    """Tests for get_noise_parameters function."""

    def test_with_noise_params(self, mock_config, tmp_path):
        """Test loading noise parameters from files."""
        # Create mock noise params file
        noise_file = tmp_path / "noise_params.json"
        noise_data = {
            "PSR_J0030": {"efac": 1.0, "equad": -7.0},
            "PSR_J0613": {"efac": 1.2, "equad": -6.5}
        }
        noise_file.write_text(json.dumps(noise_data))

        # Create mock spin injections file
        spin_file = tmp_path / "spin_injections.pkl"
        spin_df = pd.DataFrame({
            'psr': ['PSR_J0030', 'PSR_J0613'],
            'optimal_sigma': [1e-15, 2e-15],
            'optimal_gamma': [1e-8, 2e-8]
        })
        with open(spin_file, 'wb') as f:
            pickle.dump(spin_df, f)

        mock_config.set("PriorModel", "noise_params_path", str(noise_file))
        mock_config.set("PriorModel", "spin_injections_path", str(spin_file))
        mock_config.set("Data", "excluded_psrs", "")

        efac, equad, sigma_p, gamma_p = utils.get_noise_parameters(mock_config)

        assert efac is not None
        assert equad is not None
        assert sigma_p is not None
        assert gamma_p is not None

    def test_without_noise_params(self, mock_config):
        """Test when no noise parameter files are provided."""
        mock_config.set("PriorModel", "noise_params_path", "")
        mock_config.set("PriorModel", "spin_injections_path", "")

        efac, equad, sigma_p, gamma_p = utils.get_noise_parameters(mock_config)

        assert efac is None
        assert equad is None
        assert sigma_p is None
        assert gamma_p is None


class TestGetEfacEquadInjections:
    """Tests for get_efac_equad_injections function."""

    def test_load_efac_equad(self, tmp_path):
        """Test loading EFAC and EQUAD values."""
        noise_file = tmp_path / "noise_params.json"
        noise_data = {
            "PSR_J0030": {"efac": 1.0, "equad": -7.0},
            "PSR_J0613": {"efac": 1.2, "equad": -6.8},
            "J1640+2224": {"efac": 0.9, "equad": -7.2}  # Will be excluded
        }
        noise_file.write_text(json.dumps(noise_data))

        efac, equad = utils.get_efac_equad_injections(
            str(noise_file),
            excluded_psrs=["J1640+2224"]
        )

        # Should have 2 pulsars (third excluded)
        assert len(efac) == 2
        assert len(equad) == 2

        # Check EFAC values
        assert jnp.allclose(efac, jnp.array([1.0, 1.2]))

        # Check EQUAD values (10^equad_log10)
        expected_equad = jnp.array([10**(-7.0), 10**(-6.8)])
        assert jnp.allclose(equad, expected_equad)

    def test_empty_exclusions(self, tmp_path):
        """Test with no pulsars excluded."""
        noise_file = tmp_path / "noise_params.json"
        noise_data = {
            "PSR_J0030": {"efac": 1.0, "equad": -7.0},
            "PSR_J0613": {"efac": 1.2, "equad": -6.8}
        }
        noise_file.write_text(json.dumps(noise_data))

        efac, equad = utils.get_efac_equad_injections(str(noise_file), excluded_psrs=[])

        assert len(efac) == 2
        assert len(equad) == 2


class TestGetPsrNoiseInjections:
    """Tests for get_psr_noise_injections function."""

    def test_load_pulsar_noise(self, tmp_path):
        """Test loading pulsar noise parameters."""
        spin_file = tmp_path / "spin_injections.pkl"
        spin_df = pd.DataFrame({
            'psr': ['PSR_J0030', 'PSR_J0613', 'J1640+2224'],
            'optimal_sigma': [1e-15, 2e-15, 3e-15],
            'optimal_gamma': [1e-8, 2e-8, 3e-8]
        })
        with open(spin_file, 'wb') as f:
            pickle.dump(spin_df, f)

        sigma_p, gamma_p = utils.get_psr_noise_injections(
            str(spin_file),
            excluded_psrs=["J1640+2224"]
        )

        # Should have 2 pulsars (third excluded)
        assert len(sigma_p) == 2
        assert len(gamma_p) == 2

        expected_sigma = jnp.array([1e-15, 2e-15])
        expected_gamma = jnp.array([1e-8, 2e-8])

        assert jnp.allclose(sigma_p, expected_sigma)
        assert jnp.allclose(gamma_p, expected_gamma)

    def test_multiple_exclusions(self, tmp_path):
        """Test excluding multiple pulsars."""
        spin_file = tmp_path / "spin_injections.pkl"
        spin_df = pd.DataFrame({
            'psr': ['PSR_J0030', 'J1640+2224', 'PSR_J0613', 'PSR_J1234'],
            'optimal_sigma': [1e-15, 2e-15, 3e-15, 4e-15],
            'optimal_gamma': [1e-8, 2e-8, 3e-8, 4e-8]
        })
        with open(spin_file, 'wb') as f:
            pickle.dump(spin_df, f)

        sigma_p, gamma_p = utils.get_psr_noise_injections(
            str(spin_file),
            excluded_psrs=["J1640+2224", "PSR_J1234"]
        )

        # Should have 2 pulsars remaining
        assert len(sigma_p) == 2
        assert len(gamma_p) == 2


class TestCheckGpuAvailability:
    """Tests for check_gpu_availability function."""

    @patch('jax.devices')
    def test_gpu_available(self, mock_devices):
        """Test when GPU is available."""
        mock_devices.return_value = [Mock(device_kind='gpu')]

        result = utils.check_gpu_availability()

        assert result is True
        mock_devices.assert_called_once_with("gpu")

    @patch('jax.devices')
    def test_gpu_not_available(self, mock_devices):
        """Test when GPU is not available."""
        mock_devices.return_value = []

        result = utils.check_gpu_availability()

        assert result is False

    @patch('jax.devices')
    def test_gpu_check_error(self, mock_devices):
        """Test when GPU check raises an error."""
        mock_devices.side_effect = Exception("GPU error")

        result = utils.check_gpu_availability()

        assert result is False


class TestPrintParameterRanges:
    """Tests for _print_parameter_ranges function."""

    def test_print_ranges_no_config(self, capsys):
        """Test printing parameter ranges without config."""
        samples = jnp.array([[1.0, 2.0], [1.5, 2.5], [1.2, 2.2]])
        labels = ["param1", "param2"]

        utils._print_parameter_ranges(samples, labels, config=None)

        captured = capsys.readouterr()
        assert "Parameter ranges" in captured.out
        assert "param1" in captured.out
        assert "param2" in captured.out

    def test_print_ranges_with_config(self, mock_config, capsys):
        """Test printing parameter ranges with config."""
        samples = jnp.array([[-15.0], [-14.5], [-15.5]])
        labels = [r"$\log_{10} h_a$"]

        utils._print_parameter_ranges(samples, labels, config=mock_config)

        captured = capsys.readouterr()
        assert "Parameter ranges" in captured.out
        assert "Prior ranges" in captured.out


class TestGetLog10HaPriorPdf:
    """Tests for _get_log10_ha_prior_pdf function."""

    def test_uniform_prior(self, mock_config):
        """Test getting uniform prior PDF."""
        x = jnp.linspace(-16, -14, 100)
        pdf = utils._get_log10_ha_prior_pdf(mock_config, x)

        # Should return uniform distribution
        assert pdf is not None
        # PDF should be constant in the range
        assert jnp.allclose(pdf[10], pdf[50], rtol=0.1)

    def test_fixed_prior(self, mock_config):
        """Test when prior is fixed."""
        mock_config.set("PriorModel", "log10_ha_fixed", "True")
        x = jnp.linspace(-16, -14, 100)
        pdf = utils._get_log10_ha_prior_pdf(mock_config, x)

        # Should return None for fixed parameter
        assert pdf is None

    def test_no_config(self):
        """Test with no config."""
        x = jnp.linspace(-16, -14, 100)
        pdf = utils._get_log10_ha_prior_pdf(None, x)

        assert pdf is None


class TestGetLog10GammaAPriorPdf:
    """Tests for _get_log10_gamma_a_prior_pdf function."""

    def test_uniform_prior(self, mock_config):
        """Test getting uniform prior PDF for gamma_a."""
        x = jnp.linspace(-10, -8, 100)
        pdf = utils._get_log10_gamma_a_prior_pdf(mock_config, x)

        assert pdf is not None
        # Should be uniform
        assert jnp.allclose(pdf[10], pdf[50], rtol=0.1)

    def test_fixed_prior(self, mock_config):
        """Test when gamma_a prior is fixed."""
        mock_config.set("PriorModel", "log10_gamma_a_fixed", "True")
        x = jnp.linspace(-10, -8, 100)
        pdf = utils._get_log10_gamma_a_prior_pdf(mock_config, x)

        assert pdf is None
