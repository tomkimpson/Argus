"""Unit tests for parameter_sampling module."""

import pytest
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from unittest.mock import Mock, patch, MagicMock
from argus import parameter_sampling

tfpd = tfp.distributions


class TestSampleGwParameters:
    """Tests for sample_gw_parameters function."""

    @patch("numpyro.sample")
    @patch("numpyro.deterministic")
    def test_reparameterized_ha_sampling(self, mock_deterministic, mock_sample):
        """Test GW parameter sampling with reparameterization."""
        prior_specs = {
            "log10_ha_spec": tfpd.Normal(0.0, 1.0),
            "log10_ha_transform_params": {
                "mean": -15.0,
                "std": 0.33,
                "min": -16.0,
                "max": -14.0,
            },
            "log10_gamma_a_spec": tfpd.Uniform(-10.0, -8.0),
        }

        mock_sample.return_value = 0.5  # Mock sampled value
        mock_deterministic.return_value = -14.835  # Mock transformed value

        parameter_sampling.sample_gw_parameters(prior_specs)

        # Check that sample was called for reparameterized parameter
        assert mock_sample.called
        # Check that deterministic was called for transformed values
        assert mock_deterministic.called

    @patch("numpyro.deterministic")
    def test_fixed_ha_sampling(self, mock_deterministic):
        """Test with fixed log10_ha value."""
        prior_specs = {
            "log10_ha_spec": -15.0,
            "log10_ha_transform_params": None,
            "log10_gamma_a_spec": -9.0,
        }

        parameter_sampling.sample_gw_parameters(prior_specs)

        # Should call deterministic for fixed values
        assert mock_deterministic.called

    @patch("numpyro.sample")
    @patch("numpyro.deterministic")
    def test_gamma_a_distribution(self, mock_deterministic, mock_sample):
        """Test log10_gamma_a as distribution."""
        prior_specs = {
            "log10_ha_spec": -15.0,
            "log10_ha_transform_params": None,
            "log10_gamma_a_spec": tfpd.Uniform(-10.0, -8.0),
        }

        mock_sample.return_value = 0.0
        mock_deterministic.return_value = -9.0

        parameter_sampling.sample_gw_parameters(prior_specs)

        # Should sample gamma_a_prime
        assert mock_sample.called


class TestSampleHierarchicalGammaParameters:
    """Tests for sample_hierarchical_gamma_parameters function."""

    @patch("numpyro.sample")
    @patch("numpyro.deterministic")
    def test_hierarchical_sampling(self, mock_deterministic, mock_sample):
        """Test hierarchical gamma parameter sampling."""
        hierarchical_specs = {
            "log10_gamma_p_mean_spec": tfpd.Uniform(-10.0, -7.0),
            "log10_gamma_p_std_spec": tfpd.Uniform(0.1, 2.0),
        }

        n_pulsars = 3
        mock_sample.side_effect = [0.0, 0.0, jnp.zeros(n_pulsars)]
        mock_deterministic.return_value = jnp.full(n_pulsars, -8.5)

        result = parameter_sampling.sample_hierarchical_gamma_parameters(
            hierarchical_specs, n_pulsars
        )

        # Should sample mean, std, and individual parameters
        assert mock_sample.call_count >= 2

    @patch("numpyro.sample")
    @patch("numpyro.deterministic")
    def test_gradient_balancing(self, mock_deterministic, mock_sample):
        """Test that gradient balancing is applied."""
        hierarchical_specs = {
            "log10_gamma_p_mean_spec": tfpd.Uniform(-10.0, -7.0),
            "log10_gamma_p_std_spec": tfpd.Uniform(0.1, 2.0),
        }

        n_pulsars = 2
        mock_sample.side_effect = [0.0, 0.0, jnp.zeros(n_pulsars)]
        mock_deterministic.return_value = jnp.array([-8.5, -8.5])

        parameter_sampling.sample_hierarchical_gamma_parameters(
            hierarchical_specs, n_pulsars
        )

        # Check that deterministic is called for transformed values
        assert mock_deterministic.call_count >= 2


class TestSampleReparameterizedParameters:
    """Tests for sample_reparameterized_parameters function."""

    @patch("numpyro.sample")
    @patch("numpyro.deterministic")
    def test_reparameterization(self, mock_deterministic, mock_sample):
        """Test parameter reparameterization."""
        prior_spec = tfpd.Uniform(jnp.array([-10.0, -10.0]), jnp.array([-8.0, -8.0]))
        n_pulsars = 2

        mock_sample.return_value = jnp.zeros(n_pulsars)
        mock_deterministic.return_value = jnp.array([-9.0, -9.0])

        result = parameter_sampling.sample_reparameterized_parameters(
            prior_spec, "test_param", n_pulsars
        )

        # Should sample standardized parameters
        mock_sample.assert_called_once()
        # Should create deterministic transformed parameters
        mock_deterministic.assert_called_once()


class TestSampleLogRatioParameters:
    """Tests for sample_log_ratio_parameters function."""

    @patch("numpyro.sample")
    @patch("numpyro.deterministic")
    def test_log_ratio_derivation(self, mock_deterministic, mock_sample):
        """Test log-ratio parameterization for sigma_p."""
        hierarchical_specs = {
            "log10_ratio_mean_spec": tfpd.Uniform(-8.0, -6.0),
            "log10_ratio_std_spec": tfpd.Uniform(0.1, 2.0),
        }

        log10_γp = jnp.array([-8.5, -8.0])
        n_pulsars = 2

        mock_sample.side_effect = [0.0, 0.0, jnp.zeros(n_pulsars)]
        mock_deterministic.side_effect = [
            -7.0,  # log10_ratio_mean
            0.5,  # log10_ratio_std
            jnp.array([-7.0, -7.0]),  # log10_ratio
            jnp.array([-15.5, -15.0]),  # log10_σp
        ]

        result = parameter_sampling.sample_log_ratio_parameters(
            hierarchical_specs, log10_γp, n_pulsars
        )

        # Should sample ratio parameters and derive sigma_p
        assert mock_sample.call_count >= 2
        assert mock_deterministic.call_count >= 3


class TestSamplePulsarNoiseParameters:
    """Tests for sample_pulsar_noise_parameters function."""

    @patch("numpyro.deterministic")
    def test_with_fixed_values(self, mock_deterministic):
        """Test with fixed pulsar noise values."""
        prior_specs = {
            "log10_gamma_p_spec": jnp.array([-8.5, -8.0]),
            "log10_sigma_p_spec": jnp.array([-15.5, -15.0]),
            "hierarchical_specs": {},
        }

        n_pulsars = 2

        parameter_sampling.sample_pulsar_noise_parameters(prior_specs, n_pulsars)

        # Should create deterministic variables for fixed values
        assert mock_deterministic.call_count == 2

    @patch("argus.parameter_sampling.sample_hierarchical_gamma_parameters")
    @patch("argus.parameter_sampling.sample_log_ratio_parameters")
    def test_with_hierarchical_modeling(self, mock_log_ratio, mock_hierarchical):
        """Test with hierarchical modeling."""
        prior_specs = {
            "log10_gamma_p_spec": None,
            "log10_sigma_p_spec": None,
            "hierarchical_specs": {
                "hierarchical_noise": True,
                "log_ratio_parameterization": True,
            },
        }

        n_pulsars = 2
        mock_hierarchical.return_value = jnp.array([-8.5, -8.0])
        mock_log_ratio.return_value = jnp.array([-15.5, -15.0])

        result = parameter_sampling.sample_pulsar_noise_parameters(
            prior_specs, n_pulsars
        )

        # Should call hierarchical sampling
        mock_hierarchical.assert_called_once()
        mock_log_ratio.assert_called_once()


class TestSampleMeasurementNoiseParameters:
    """Tests for sample_measurement_noise_parameters function."""

    @patch("numpyro.sample")
    @patch("numpyro.deterministic")
    def test_efac_distribution(self, mock_deterministic, mock_sample):
        """Test EFAC sampling from distribution."""
        n_pulsars = 2
        prior_specs = {
            "efac_spec": tfpd.Uniform(
                jnp.full(n_pulsars, 0.5), jnp.full(n_pulsars, 2.0)
            ),
            "equad_spec": jnp.full(n_pulsars, 1e-7),
        }

        mock_sample.return_value = jnp.ones(n_pulsars)
        mock_deterministic.side_effect = [
            jnp.ones(n_pulsars),
            jnp.full(n_pulsars, 1e-7),
        ]

        parameter_sampling.sample_measurement_noise_parameters(prior_specs, n_pulsars)

        # Should sample EFAC
        assert mock_sample.called

    @patch("numpyro.deterministic")
    def test_fixed_equad(self, mock_deterministic):
        """Test with fixed EQUAD values."""
        n_pulsars = 2
        prior_specs = {
            "efac_spec": jnp.ones(n_pulsars),
            "equad_spec": jnp.full(n_pulsars, 1e-7),
        }

        parameter_sampling.sample_measurement_noise_parameters(prior_specs, n_pulsars)

        # Should create deterministic for fixed values
        assert mock_deterministic.call_count == 2

    @patch("numpyro.sample")
    @patch("numpyro.deterministic")
    def test_log10_equad_parameterization(self, mock_deterministic, mock_sample):
        """Test log10(EQUAD) parameterization."""
        n_pulsars = 2
        prior_specs = {
            "efac_spec": jnp.ones(n_pulsars),
            "equad_spec": {
                "use_log10": True,
                "log10_equad_spec": tfpd.Uniform(
                    jnp.full(n_pulsars, -8.0), jnp.full(n_pulsars, -6.0)
                ),
            },
        }

        mock_sample.return_value = jnp.full(n_pulsars, -7.0)
        mock_deterministic.side_effect = [
            jnp.ones(n_pulsars),
            jnp.full(n_pulsars, -7.0),
            jnp.full(n_pulsars, 1e-7),
        ]

        parameter_sampling.sample_measurement_noise_parameters(prior_specs, n_pulsars)

        # Should sample log10_equad
        assert mock_sample.called


class TestCountFreeParameters:
    """Tests for count_free_parameters function."""

    def test_count_with_reparameterized_ha(self):
        """Test parameter counting with reparameterized ha."""
        prior_specs = {
            "log10_ha_transform_params": {"mean": -15.0, "std": 0.33},
            "log10_gamma_a_spec": tfpd.Uniform(-10.0, -8.0),
            "log10_gamma_p_spec": None,
            "log10_sigma_p_spec": None,
            "efac_spec": tfpd.Uniform(jnp.zeros(2), jnp.ones(2)),
            "equad_spec": {
                "use_log10": True,
                "log10_equad_spec": tfpd.Uniform(jnp.zeros(2), jnp.ones(2)),
            },
        }

        n_pulsars = 2
        count = parameter_sampling.count_free_parameters(prior_specs, n_pulsars)

        # ha (1) + gamma_a (1) + hierarchical gamma_p (2+2) + hierarchical sigma_p (2+2) + efac (2) + equad (2) = 14
        assert count == 14

    def test_count_with_fixed_parameters(self):
        """Test counting with some fixed parameters."""
        prior_specs = {
            "log10_ha_transform_params": None,  # Fixed
            "log10_gamma_a_spec": -9.0,  # Fixed
            "log10_gamma_p_spec": jnp.array([-8.5, -8.0]),  # Fixed
            "log10_sigma_p_spec": jnp.array([-15.5, -15.0]),  # Fixed
            "efac_spec": tfpd.Uniform(jnp.zeros(2), jnp.ones(2)),  # Free (2)
            "equad_spec": jnp.array([1e-7, 1e-7]),  # Fixed
        }

        n_pulsars = 2
        count = parameter_sampling.count_free_parameters(prior_specs, n_pulsars)

        # Only EFAC is free (2 parameters)
        assert count == 2

    def test_count_all_free(self):
        """Test counting with all parameters free."""
        prior_specs = {
            "log10_ha_transform_params": {"mean": -15.0, "std": 0.33},
            "log10_gamma_a_spec": tfpd.Uniform(-10.0, -8.0),
            "log10_gamma_p_spec": tfpd.Uniform(jnp.zeros(3), jnp.ones(3)),
            "log10_sigma_p_spec": tfpd.Uniform(jnp.zeros(3), jnp.ones(3)),
            "efac_spec": tfpd.Uniform(jnp.zeros(3), jnp.ones(3)),
            "equad_spec": {
                "use_log10": True,
                "log10_equad_spec": tfpd.Uniform(jnp.zeros(3), jnp.ones(3)),
            },
        }

        n_pulsars = 3
        count = parameter_sampling.count_free_parameters(prior_specs, n_pulsars)

        # ha (1) + gamma_a (1) + gamma_p (3) + sigma_p (3) + efac (3) + equad (3) = 14
        assert count == 14
