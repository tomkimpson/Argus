"""Unit tests for bayesian_inference module."""

import pytest
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from unittest.mock import Mock, patch, MagicMock
from argus import bayesian_inference

tfpd = tfp.distributions


class TestParameters:
    """Tests for Parameters dataclass."""

    def test_parameters_creation(self):
        """Test creating Parameters instance."""
        params = bayesian_inference.Parameters(
            log10_gamma_a=-9.0,
            γa=1e-9,
            ha=1e-15,
            γp=jnp.array([1e-8, 2e-8]),
            σp=jnp.array([1e-15, 2e-15]),
            EFAC=jnp.array([1.0, 1.2]),
            EQUAD=jnp.array([1e-7, 1.5e-7])
        )

        assert params.log10_gamma_a == -9.0
        assert params.γa == 1e-9
        assert params.ha == 1e-15
        assert len(params.γp) == 2
        assert len(params.σp) == 2

    def test_parameters_with_arrays(self):
        """Test Parameters with JAX arrays."""
        γp = jnp.array([1e-8, 2e-8, 3e-8])
        σp = jnp.array([1e-15, 2e-15, 3e-15])

        params = bayesian_inference.Parameters(
            log10_gamma_a=-9.0,
            γa=1e-9,
            ha=1e-15,
            γp=γp,
            σp=σp,
            EFAC=jnp.ones(3),
            EQUAD=jnp.full(3, 1e-7)
        )

        assert jnp.allclose(params.γp, γp)
        assert jnp.allclose(params.σp, σp)


class TestDisplayPriorSummary:
    """Tests for display_prior_summary function."""

    def test_basic_display(self, mock_logger, capsys):
        """Test basic prior summary display."""
        prior_specs = {
            "log10_ha_spec": tfpd.Uniform(-16.0, -14.0),
            "log10_ha_transform_params": None,
            "log10_gamma_a_spec": tfpd.Uniform(-10.0, -8.0),
            "log10_gamma_p_spec": None,
            "log10_sigma_p_spec": None,
            "efac_spec": jnp.ones(2),
            "equad_spec": jnp.full(2, 1e-7),
            "hierarchical_specs": {
                "hierarchical_noise": False,
                "log_ratio_parameterization": False
            }
        }

        bayesian_inference.display_prior_summary(prior_specs, n_pulsars=2, logger=mock_logger)

        # Should have logged information
        assert mock_logger.info.called

    def test_reparameterized_display(self, mock_logger):
        """Test display with reparameterized priors."""
        prior_specs = {
            "log10_ha_spec": tfpd.Normal(0.0, 1.0),
            "log10_ha_transform_params": {
                "mean": -15.0,
                "std": 0.33,
                "min": -16.0,
                "max": -14.0
            },
            "log10_gamma_a_spec": -9.0,
            "log10_gamma_p_spec": None,
            "log10_sigma_p_spec": None,
            "efac_spec": jnp.ones(2),
            "equad_spec": jnp.full(2, 1e-7),
            "hierarchical_specs": {
                "hierarchical_noise": True,
                "log_ratio_parameterization": True,
                "log10_gamma_p_mean_spec": tfpd.Uniform(-10.0, -7.0),
                "log10_gamma_p_std_spec": tfpd.Uniform(0.1, 2.0),
                "log10_ratio_mean_spec": tfpd.Uniform(-8.0, -6.0),
                "log10_ratio_std_spec": tfpd.Uniform(0.1, 2.0)
            }
        }

        bayesian_inference.display_prior_summary(prior_specs, n_pulsars=2, logger=mock_logger)

        assert mock_logger.info.called


class TestLogLikelihoodFn:
    """Tests for log_likelihood_fn function."""

    def test_basic_likelihood(self):
        """Test basic log likelihood calculation."""
        # Mock Kalman filter
        mock_kf = Mock()
        mock_kf.get_likelihood = Mock(return_value=100.0)

        ll = bayesian_inference.log_likelihood_fn(
            mock_kf,
            log10_ha=-15.0,
            log10_gamma_a=-9.0,
            log10_γp=jnp.array([-8.0, -8.5]),
            log10_σp=jnp.array([-15.0, -15.5]),
            efac=jnp.array([1.0, 1.2]),
            equad=jnp.array([1e-7, 1.5e-7])
        )

        # Should call Kalman filter
        mock_kf.get_likelihood.assert_called_once()
        assert ll == 100.0

    def test_parameter_transformation(self):
        """Test that log10 parameters are transformed correctly."""
        mock_kf = Mock()
        mock_kf.get_likelihood = Mock(return_value=100.0)

        bayesian_inference.log_likelihood_fn(
            mock_kf,
            log10_ha=-15.0,
            log10_gamma_a=-9.0,
            log10_γp=jnp.array([-8.0]),
            log10_σp=jnp.array([-15.0]),
            efac=jnp.array([1.0]),
            equad=jnp.array([1e-7])
        )

        # Get the Parameters object that was passed
        call_args = mock_kf.get_likelihood.call_args[0][0]

        # Check transformations
        assert jnp.isclose(call_args.ha, 10**(-15.0))
        assert jnp.isclose(call_args.γa, 10**(-9.0))
        assert jnp.allclose(call_args.γp, 10**jnp.array([-8.0]))


class TestSetupNutsKernel:
    """Tests for setup_nuts_kernel function."""

    @patch('argus.parameter_sampling.count_free_parameters')
    def test_kernel_setup(self, mock_count, mock_config):
        """Test NUTS kernel setup."""
        mock_count.return_value = 10

        prior_specs = {
            "log10_ha_transform_params": {"mean": -15.0, "std": 0.33},
            "log10_gamma_a_spec": tfpd.Uniform(-10.0, -8.0),
            "log10_gamma_p_spec": None,
            "log10_sigma_p_spec": None,
            "efac_spec": jnp.ones(2),
            "equad_spec": jnp.full(2, 1e-7)
        }

        kernel, nuts_info = bayesian_inference.setup_nuts_kernel(
            prior_specs, n_pulsars=2, config=mock_config
        )

        assert nuts_info["total_params"] == 10
        assert nuts_info["target_accept_prob"] == 0.95

    @patch('argus.parameter_sampling.count_free_parameters')
    def test_custom_nuts_params(self, mock_count, mock_config):
        """Test NUTS with custom parameters."""
        mock_count.return_value = 5
        mock_config.set("NUTS", "target_accept_prob", "0.90")
        mock_config.set("NUTS", "max_tree_depth", "15")

        prior_specs = {
            "log10_ha_transform_params": None,
            "log10_gamma_a_spec": -9.0,
            "log10_gamma_p_spec": None,
            "log10_sigma_p_spec": None,
            "efac_spec": jnp.ones(2),
            "equad_spec": jnp.full(2, 1e-7)
        }

        kernel, nuts_info = bayesian_inference.setup_nuts_kernel(
            prior_specs, n_pulsars=2, config=mock_config
        )

        assert nuts_info["target_accept_prob"] == 0.90
        assert nuts_info["max_tree_depth"] == 15


class TestPrintNutsDiagnostics:
    """Tests for print_nuts_diagnostics function."""

    def test_basic_diagnostics(self, mock_config, capsys):
        """Test printing NUTS diagnostics."""
        prior_specs = {
            "hierarchical_specs": {
                "hierarchical_noise": True,
                "log_ratio_parameterization": True
            }
        }

        nuts_info = {
            "total_params": 10,
            "target_accept_prob": 0.95,
            "max_tree_depth": 10,
            "dense_mass": False
        }

        bayesian_inference.print_nuts_diagnostics(prior_specs, nuts_info, mock_config)

        captured = capsys.readouterr()
        assert "NumPyro NUTS inference" in captured.out
        assert "10" in captured.out  # Total params


class TestTestLikelihoodPerformance:
    """Tests for test_likelihood_performance function."""

    @patch('argus.utils.get_noise_parameters')
    def test_performance_test(self, mock_get_noise, mock_config, mock_logger):
        """Test likelihood performance testing."""
        # Create a mock that returns a JAX array-like value
        mock_result = Mock()
        mock_result.block_until_ready = Mock(return_value=None)
        # Make the mock convertible to float
        mock_ll_value = jnp.array(-100.5)

        mock_kf = Mock()
        mock_kf.get_likelihood = Mock(return_value=mock_ll_value)

        mock_get_noise.return_value = (
            jnp.ones(2),
            jnp.full(2, 1e-7),
            jnp.full(2, 1e-15),
            jnp.full(2, 1e-8)
        )

        bayesian_inference.test_likelihood_performance(
            mock_kf, mock_config, n_pulsars=2, logger=mock_logger
        )

        # Should have called get_likelihood
        assert mock_kf.get_likelihood.called
        # Should have logged results
        assert mock_logger.info.called

    @patch('argus.utils.get_noise_parameters')
    def test_with_none_noise_params(self, mock_get_noise, mock_config, mock_logger):
        """Test with None noise parameters (creates defaults)."""
        # Make the mock convertible to float
        mock_ll_value = jnp.array(-100.5)

        mock_kf = Mock()
        mock_kf.get_likelihood = Mock(return_value=mock_ll_value)

        # Return None for all noise params
        mock_get_noise.return_value = (None, None, None, None)

        bayesian_inference.test_likelihood_performance(
            mock_kf, mock_config, n_pulsars=2, logger=mock_logger
        )

        # Should still work with defaults
        assert mock_kf.get_likelihood.called


class TestNumpyroModel:
    """Tests for numpyro_model function."""

    @patch('argus.bayesian_inference.sample_gw_parameters')
    @patch('argus.bayesian_inference.sample_pulsar_noise_parameters')
    @patch('argus.bayesian_inference.sample_measurement_noise_parameters')
    def test_model_structure(self, mock_meas, mock_psr, mock_gw):
        """Test NumPyro model structure."""
        import numpyro
        import jax.random as random

        # Setup mocks
        mock_gw.return_value = (-15.0, -9.0, 1e-9)
        mock_psr.return_value = (jnp.array([-8.0]), jnp.array([-15.0]))
        mock_meas.return_value = (jnp.array([1.0]), jnp.array([1e-7]))

        mock_kf = Mock()
        mock_kf.get_likelihood = Mock(return_value=100.0)

        prior_specs = {
            "log10_ha_transform_params": None,
            "log10_ha_spec": -15.0,
            "log10_gamma_a_spec": -9.0,
            "log10_gamma_p_spec": None,
            "log10_sigma_p_spec": None,
            "efac_spec": jnp.ones(1),
            "equad_spec": jnp.full(1, 1e-7)
        }

        # Run model in NumPyro context with seed
        rng_key = random.PRNGKey(0)
        with numpyro.handlers.seed(rng_seed=0):
            with numpyro.handlers.trace() as trace:
                bayesian_inference.numpyro_model(mock_kf, prior_specs, n_pulsars=1)

        # Should sample all parameter groups
        mock_gw.assert_called_once()
        mock_psr.assert_called_once()
        mock_meas.assert_called_once()

        # Should have likelihood in trace
        assert "likelihood" in trace
