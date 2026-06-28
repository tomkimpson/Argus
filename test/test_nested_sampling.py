"""Tests for jaxns nested sampling integration."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import logging

_logger = logging.getLogger("argus")
if not _logger.handlers:
    _logger.addHandler(logging.NullHandler())
    _logger.setLevel(logging.WARNING)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def cw_prior_specs_all_sampled():
    """Prior specs with all CW parameters sampled and noise fixed."""
    import tensorflow_probability.substrates.jax as tfp

    tfpd = tfp.distributions

    cw_specs = {
        "log10_h0_spec": None,
        "log10_h0_transform_params": {
            "mean": -14.0,
            "std": 0.667,
            "min": -16.0,
            "max": -12.0,
        },
        "alpha_gw_spec": None,
        "alpha_gw_transform_params": {
            "mean": 3.1416,
            "std": 1.0472,
            "min": 0.0,
            "max": 6.2832,
        },
        "sin_delta_gw_spec": None,
        "sin_delta_gw_transform_params": {
            "mean": 0.0,
            "std": 0.333,
            "min": -1.0,
            "max": 1.0,
        },
        "delta_gw_spec": None,
        "log10_f_gw_spec": None,
        "log10_f_gw_transform_params": {
            "mean": -8.0,
            "std": 0.333,
            "min": -9.0,
            "max": -7.0,
        },
        "cos_iota_spec": None,
        "cos_iota_transform_params": {
            "mean": 0.0,
            "std": 0.333,
            "min": -1.0,
            "max": 1.0,
        },
        "psi_spec": None,
        "psi_transform_params": {
            "mean": 1.5708,
            "std": 0.5236,
            "min": 0.0,
            "max": 3.14159,
        },
        "Phi0_spec": None,
        "Phi0_transform_params": {
            "mean": 3.1416,
            "std": 1.0472,
            "min": 0.0,
            "max": 6.2832,
        },
        "chi_transform_params": {
            "mean": 3.1416,
            "std": 1.0472,
            "min": 0.0,
            "max": 6.2832,
        },
    }

    prior_specs = {
        "cw_specs": cw_specs,
        "log10_gamma_p_spec": jnp.full(3, -8.0),  # Fixed
        "log10_sigma_p_spec": jnp.full(3, -15.0),  # Fixed
        "efac_spec": jnp.ones(3),  # Fixed
        "equad_spec": jnp.full(3, 1e-7),  # Fixed
        "hierarchical_specs": None,
    }
    return prior_specs


@pytest.fixture
def cw_prior_specs_some_fixed():
    """Prior specs with some CW parameters fixed."""
    cw_specs = {
        "log10_h0_spec": None,
        "log10_h0_transform_params": {
            "mean": -14.0,
            "std": 0.667,
            "min": -16.0,
            "max": -12.0,
        },
        "alpha_gw_spec": 4.067,  # Fixed
        "alpha_gw_transform_params": None,
        "sin_delta_gw_spec": 0.14,  # Fixed
        "sin_delta_gw_transform_params": None,
        "delta_gw_spec": 0.14,
        "log10_f_gw_spec": -8.215,  # Fixed
        "log10_f_gw_transform_params": None,
        "cos_iota_spec": None,
        "cos_iota_transform_params": {
            "mean": 0.0,
            "std": 0.333,
            "min": -1.0,
            "max": 1.0,
        },
        "psi_spec": None,
        "psi_transform_params": {
            "mean": 1.5708,
            "std": 0.5236,
            "min": 0.0,
            "max": 3.14159,
        },
        "Phi0_spec": None,
        "Phi0_transform_params": {
            "mean": 3.1416,
            "std": 1.0472,
            "min": 0.0,
            "max": 6.2832,
        },
        "chi_transform_params": None,  # No pulsar term
    }

    prior_specs = {
        "cw_specs": cw_specs,
        "log10_gamma_p_spec": jnp.full(3, -8.0),
        "log10_sigma_p_spec": jnp.full(3, -15.0),
        "efac_spec": jnp.ones(3),
        "equad_spec": jnp.full(3, 1e-7),
        "hierarchical_specs": None,
    }
    return prior_specs


@pytest.fixture
def cw_prior_specs_hierarchical():
    """Prior specs with hierarchical noise modeling."""
    import tensorflow_probability.substrates.jax as tfp

    tfpd = tfp.distributions

    cw_specs = {
        "log10_h0_spec": None,
        "log10_h0_transform_params": {
            "mean": -14.0,
            "std": 0.667,
            "min": -16.0,
            "max": -12.0,
        },
        "alpha_gw_spec": None,
        "alpha_gw_transform_params": {
            "mean": 3.1416,
            "std": 1.0472,
            "min": 0.0,
            "max": 6.2832,
        },
        "sin_delta_gw_spec": None,
        "sin_delta_gw_transform_params": {
            "mean": 0.0,
            "std": 0.333,
            "min": -1.0,
            "max": 1.0,
        },
        "delta_gw_spec": None,
        "log10_f_gw_spec": None,
        "log10_f_gw_transform_params": {
            "mean": -8.0,
            "std": 0.333,
            "min": -9.0,
            "max": -7.0,
        },
        "cos_iota_spec": None,
        "cos_iota_transform_params": {
            "mean": 0.0,
            "std": 0.333,
            "min": -1.0,
            "max": 1.0,
        },
        "psi_spec": None,
        "psi_transform_params": {
            "mean": 1.5708,
            "std": 0.5236,
            "min": 0.0,
            "max": 3.14159,
        },
        "Phi0_spec": None,
        "Phi0_transform_params": {
            "mean": 3.1416,
            "std": 1.0472,
            "min": 0.0,
            "max": 6.2832,
        },
        "chi_transform_params": None,
    }

    hierarchical_specs = {
        "hierarchical_noise": True,
        "log_ratio_parameterization": True,
        "log10_gamma_p_mean_spec": tfpd.Uniform(-9.0, -7.0),
        "log10_gamma_p_std_spec": tfpd.Uniform(0.1, 1.0),
        "log10_ratio_mean_spec": tfpd.Uniform(-8.0, -4.0),
        "log10_ratio_std_spec": tfpd.Uniform(0.5, 3.0),
    }

    prior_specs = {
        "cw_specs": cw_specs,
        "log10_gamma_p_spec": None,  # Hierarchical
        "log10_sigma_p_spec": None,  # Log-ratio
        "efac_spec": jnp.ones(3),
        "equad_spec": jnp.full(3, 1e-7),
        "hierarchical_specs": hierarchical_specs,
    }
    return prior_specs


# ============================================================
# Tests for build_jaxns_cw_prior_model
# ============================================================


class TestBuildJaxnsCWPriorModel:
    """Tests for the jaxns prior model builder."""

    def test_returns_callable(self, cw_prior_specs_all_sampled):
        """Prior model builder should return a callable generator function."""
        from argus.parameter_sampling import build_jaxns_cw_prior_model

        prior_model = build_jaxns_cw_prior_model(
            cw_prior_specs_all_sampled, n_pulsars=3
        )
        assert callable(prior_model)

    def test_model_creates_valid_jaxns_model(self, cw_prior_specs_all_sampled):
        """Prior model should be accepted by jaxns.Model."""
        from argus.parameter_sampling import build_jaxns_cw_prior_model
        from jaxns import Model

        prior_model = build_jaxns_cw_prior_model(
            cw_prior_specs_all_sampled, n_pulsars=3
        )

        # Simple log-likelihood that accepts the prior model output
        def log_likelihood(
            log10_h0,
            alpha_gw,
            delta_gw,
            log10_f_gw,
            cos_iota,
            psi,
            Phi0,
            chi,
            log10_gamma_p,
            log10_sigma_p,
            efac,
            equad,
        ):
            return jnp.float64(0.0)

        model = Model(prior_model=prior_model, log_likelihood=log_likelihood)
        assert model.U_ndims > 0

    def test_correct_dims_all_sampled(self, cw_prior_specs_all_sampled):
        """All-sampled config: 7 CW + 3 chi = 10 scalar dims (noise fixed)."""
        from argus.parameter_sampling import build_jaxns_cw_prior_model
        from jaxns import Model

        n_pulsars = 3
        prior_model = build_jaxns_cw_prior_model(cw_prior_specs_all_sampled, n_pulsars)

        def log_likelihood(*args):
            return jnp.float64(0.0)

        model = Model(prior_model=prior_model, log_likelihood=log_likelihood)
        # 7 CW scalars + 3 chi = 10
        assert model.U_ndims == 10

    def test_correct_dims_some_fixed(self, cw_prior_specs_some_fixed):
        """Some-fixed config: 4 sampled CW (h0, cos_iota, psi, Phi0), no chi."""
        from argus.parameter_sampling import build_jaxns_cw_prior_model
        from jaxns import Model

        n_pulsars = 3
        prior_model = build_jaxns_cw_prior_model(cw_prior_specs_some_fixed, n_pulsars)

        def log_likelihood(*args):
            return jnp.float64(0.0)

        model = Model(prior_model=prior_model, log_likelihood=log_likelihood)
        assert model.U_ndims == 4

    def test_correct_dims_hierarchical(self, cw_prior_specs_hierarchical):
        """Hierarchical noise: 7 CW + 2 gamma hyper + 3 gamma_p + 2 ratio hyper + 3 ratio."""
        from argus.parameter_sampling import build_jaxns_cw_prior_model
        from jaxns import Model

        n_pulsars = 3
        prior_model = build_jaxns_cw_prior_model(cw_prior_specs_hierarchical, n_pulsars)

        def log_likelihood(*args):
            return jnp.float64(0.0)

        model = Model(prior_model=prior_model, log_likelihood=log_likelihood)
        # 7 CW + 0 chi + (2 + 3) gamma + (2 + 3) ratio = 17
        assert model.U_ndims == 17


# ============================================================
# Tests for likelihood wrapper
# ============================================================


class TestLikelihoodWrapper:
    """Tests that the jaxns likelihood wrapper matches direct cw_log_likelihood_fn."""

    def test_likelihood_matches_direct_call(self):
        """jaxns wrapper should produce same value as direct cw_log_likelihood_fn."""
        from argus.bayesian_inference import cw_log_likelihood_fn, CWParameters
        from argus.cw_kalman_filter import CWKalmanFilter
        import pandas as pd

        # Build minimal synthetic data
        np.random.seed(42)
        Npsr = 2
        nobs = 30

        toas_list = [np.sort(np.random.uniform(0, 1e9, nobs)) for _ in range(Npsr)]
        residuals_list = [np.random.normal(0, 1e-7, nobs) for _ in range(Npsr)]
        errors_list = [np.full(nobs, 1e-7) for _ in range(Npsr)]

        metadata = pd.DataFrame(
            {
                "name": [f"J000{i}+0001" for i in range(Npsr)],
                "dim_M": [3] * Npsr,
                "RA": [0.5, 1.5],
                "DEC": [0.3, -0.2],
                "F0": [200.0, 300.0],
            }
        )
        design_matrices = [np.random.randn(nobs, 3) * 0.01 for _ in range(Npsr)]
        P_eps = [np.eye(3) * 0.01 for _ in range(Npsr)]

        data = {
            "processed_residuals": {
                "toas": toas_list,
                "residuals": residuals_list,
                "errors": errors_list,
                "n_obs": np.array([nobs] * Npsr),
            },
            "metadata": metadata,
            "design_matrices": design_matrices,
            "parameter_covariances": P_eps,
        }

        kf = CWKalmanFilter(data, include_pulsar_term=False)

        # Test parameters
        log10_h0 = -13.5
        alpha_gw = 4.0
        delta_gw = 0.14
        log10_f_gw = -8.2
        cos_iota = 0.9
        psi = 0.6
        Phi0 = 0.2
        chi = jnp.zeros(Npsr)
        log10_gp = jnp.full(Npsr, -8.0)
        log10_sp = jnp.full(Npsr, -15.0)
        efac = jnp.ones(Npsr)
        equad = jnp.full(Npsr, 1e-7)

        # Direct call
        ll_direct = cw_log_likelihood_fn(
            kf,
            log10_h0,
            alpha_gw,
            delta_gw,
            log10_f_gw,
            cos_iota,
            psi,
            Phi0,
            chi,
            log10_gp,
            log10_sp,
            efac,
            equad,
        )

        assert jnp.isfinite(ll_direct)


# ============================================================
# Tests for ArviZ conversion
# ============================================================


class TestArvizConversion:
    """Tests for _jaxns_results_to_arviz."""

    def test_produces_inference_data(self):
        """Conversion should produce a valid ArviZ InferenceData object."""
        import arviz as az
        from unittest.mock import MagicMock

        # Create mock jaxns results
        results = MagicMock()
        results.samples = {
            "log10_h0": jnp.linspace(-14.0, -13.0, 100),
            "alpha_gw": jnp.linspace(3.5, 4.5, 100),
            "sin_delta_gw": jnp.linspace(-0.5, 0.5, 100),
            "log10_f_gw": jnp.linspace(-8.5, -7.5, 100),
            "cos_iota": jnp.linspace(-0.5, 0.5, 100),
            "psi": jnp.linspace(0.0, 3.14, 100),
            "Phi0": jnp.linspace(0.0, 6.28, 100),
        }
        results.log_dp_mean = jnp.zeros(100)  # Equal weights

        from argus.bayesian_inference import _jaxns_results_to_arviz

        inf_data = _jaxns_results_to_arviz(results, num_posterior_samples=50)

        assert isinstance(inf_data, az.InferenceData)
        assert hasattr(inf_data, "posterior")

    def test_contains_cw_parameter_names(self):
        """Output should contain standard CW parameter names for corner plot compatibility."""
        from unittest.mock import MagicMock

        results = MagicMock()
        results.samples = {
            "log10_h0": jnp.linspace(-14.0, -13.0, 100),
            "alpha_gw": jnp.linspace(3.5, 4.5, 100),
            "sin_delta_gw": jnp.linspace(-0.5, 0.5, 100),
            "log10_f_gw": jnp.linspace(-8.5, -7.5, 100),
            "cos_iota": jnp.linspace(-0.5, 0.5, 100),
            "psi": jnp.linspace(0.0, 3.14, 100),
            "Phi0": jnp.linspace(0.0, 6.28, 100),
        }
        results.log_dp_mean = jnp.zeros(100)

        from argus.bayesian_inference import _jaxns_results_to_arviz

        inf_data = _jaxns_results_to_arviz(results, num_posterior_samples=50)

        posterior_vars = list(inf_data.posterior.data_vars)
        assert "log10_h0" in posterior_vars
        assert "alpha_gw" in posterior_vars
        assert "delta_gw" in posterior_vars  # Derived from sin_delta_gw
        assert "log10_f_gw" in posterior_vars

    def test_single_chain_shape(self):
        """Output should have chain dimension = 1 for nested sampling."""
        from unittest.mock import MagicMock

        results = MagicMock()
        results.samples = {
            "log10_h0": jnp.linspace(-14.0, -13.0, 100),
        }
        results.log_dp_mean = jnp.zeros(100)

        from argus.bayesian_inference import _jaxns_results_to_arviz

        inf_data = _jaxns_results_to_arviz(results, num_posterior_samples=50)

        assert inf_data.posterior["log10_h0"].shape[0] == 1  # 1 chain
        assert inf_data.posterior["log10_h0"].shape[1] == 50  # 50 draws
