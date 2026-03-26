"""Tests for blackjax tempered SMC integration."""

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
# Fixtures (reuse patterns from test_nested_sampling.py)
# ============================================================


@pytest.fixture
def cw_prior_specs_all_sampled():
    """Prior specs with all CW parameters sampled and noise fixed."""
    cw_specs = {
        "log10_h0_spec": None,
        "log10_h0_transform_params": {"mean": -14.0, "std": 0.667, "min": -16.0, "max": -12.0},
        "alpha_gw_spec": None,
        "alpha_gw_transform_params": {"mean": 3.1416, "std": 1.0472, "min": 0.0, "max": 6.2832},
        "sin_delta_gw_spec": None,
        "sin_delta_gw_transform_params": {"mean": 0.0, "std": 0.333, "min": -1.0, "max": 1.0},
        "delta_gw_spec": None,
        "log10_f_gw_spec": None,
        "log10_f_gw_transform_params": {"mean": -8.0, "std": 0.333, "min": -9.0, "max": -7.0},
        "cos_iota_spec": None,
        "cos_iota_transform_params": {"mean": 0.0, "std": 0.333, "min": -1.0, "max": 1.0},
        "psi_spec": None,
        "psi_transform_params": {"mean": 1.5708, "std": 0.5236, "min": 0.0, "max": 3.14159},
        "Phi0_spec": None,
        "Phi0_transform_params": {"mean": 3.1416, "std": 1.0472, "min": 0.0, "max": 6.2832},
        "chi_transform_params": {"mean": 3.1416, "std": 1.0472, "min": 0.0, "max": 6.2832},
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
        "log10_h0_transform_params": {"mean": -14.0, "std": 0.667, "min": -16.0, "max": -12.0},
        "alpha_gw_spec": 4.067,  # Fixed
        "alpha_gw_transform_params": None,
        "sin_delta_gw_spec": 0.14,  # Fixed
        "sin_delta_gw_transform_params": None,
        "delta_gw_spec": 0.14,
        "log10_f_gw_spec": -8.215,  # Fixed
        "log10_f_gw_transform_params": None,
        "cos_iota_spec": None,
        "cos_iota_transform_params": {"mean": 0.0, "std": 0.333, "min": -1.0, "max": 1.0},
        "psi_spec": None,
        "psi_transform_params": {"mean": 1.5708, "std": 0.5236, "min": 0.0, "max": 3.14159},
        "Phi0_spec": None,
        "Phi0_transform_params": {"mean": 3.1416, "std": 1.0472, "min": 0.0, "max": 6.2832},
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
        "log10_h0_transform_params": {"mean": -14.0, "std": 0.667, "min": -16.0, "max": -12.0},
        "alpha_gw_spec": None,
        "alpha_gw_transform_params": {"mean": 3.1416, "std": 1.0472, "min": 0.0, "max": 6.2832},
        "sin_delta_gw_spec": None,
        "sin_delta_gw_transform_params": {"mean": 0.0, "std": 0.333, "min": -1.0, "max": 1.0},
        "delta_gw_spec": None,
        "log10_f_gw_spec": None,
        "log10_f_gw_transform_params": {"mean": -8.0, "std": 0.333, "min": -9.0, "max": -7.0},
        "cos_iota_spec": None,
        "cos_iota_transform_params": {"mean": 0.0, "std": 0.333, "min": -1.0, "max": 1.0},
        "psi_spec": None,
        "psi_transform_params": {"mean": 1.5708, "std": 0.5236, "min": 0.0, "max": 3.14159},
        "Phi0_spec": None,
        "Phi0_transform_params": {"mean": 3.1416, "std": 1.0472, "min": 0.0, "max": 6.2832},
        "chi_transform_params": {"mean": 3.1416, "std": 1.0472, "min": 0.0, "max": 6.2832},
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
# Tests for parameter registry
# ============================================================


class TestParameterRegistry:
    """Tests for build_parameter_registry."""

    def test_ndim_all_sampled(self, cw_prior_specs_all_sampled):
        """All CW sampled + chi + noise fixed => 7 + 3 = 10 dims."""
        from argus.tempered_smc import build_parameter_registry

        registry = build_parameter_registry(cw_prior_specs_all_sampled, n_pulsars=3)
        assert registry.ndim == 10

    def test_ndim_some_fixed(self, cw_prior_specs_some_fixed):
        """4 CW sampled (h0, cos_iota, psi, Phi0), no chi, noise fixed => 4."""
        from argus.tempered_smc import build_parameter_registry

        registry = build_parameter_registry(cw_prior_specs_some_fixed, n_pulsars=3)
        assert registry.ndim == 4

    def test_ndim_hierarchical(self, cw_prior_specs_hierarchical):
        """7 CW + 3 chi + (2+3) gamma + (2+3) ratio = 20."""
        from argus.tempered_smc import build_parameter_registry

        registry = build_parameter_registry(cw_prior_specs_hierarchical, n_pulsars=3)
        # 7 CW + 3 chi + 2 gamma_hyper + 3 gamma_raw + 2 ratio_hyper + 3 ratio_raw = 20
        assert registry.ndim == 20

    def test_matches_count_free_parameters(self, cw_prior_specs_all_sampled):
        """Registry ndim should agree with count_free_parameters."""
        from argus.tempered_smc import build_parameter_registry
        from argus.parameter_sampling import count_free_parameters

        n_pulsars = 3
        registry = build_parameter_registry(cw_prior_specs_all_sampled, n_pulsars)
        expected = count_free_parameters(cw_prior_specs_all_sampled, n_pulsars)
        assert registry.ndim == expected

    def test_matches_count_free_parameters_hierarchical(self, cw_prior_specs_hierarchical):
        """Registry ndim should match count_free_parameters for hierarchical specs."""
        from argus.tempered_smc import build_parameter_registry
        from argus.parameter_sampling import count_free_parameters

        n_pulsars = 3
        registry = build_parameter_registry(cw_prior_specs_hierarchical, n_pulsars)
        expected = count_free_parameters(cw_prior_specs_hierarchical, n_pulsars)
        assert registry.ndim == expected

    def test_fixed_values_populated(self, cw_prior_specs_some_fixed):
        """Fixed parameters should be stored in fixed_values dict."""
        from argus.tempered_smc import build_parameter_registry

        registry = build_parameter_registry(cw_prior_specs_some_fixed, n_pulsars=3)
        assert "alpha_gw" in registry.fixed_values
        assert "log10_f_gw" in registry.fixed_values


# ============================================================
# Tests for unpack_to_physical
# ============================================================


class TestUnpackToPhysical:
    """Tests for unpacking flat unconstrained vector to physical parameters."""

    def test_round_trip_affine(self, cw_prior_specs_all_sampled):
        """Unpacking zeros should give transform means."""
        from argus.tempered_smc import build_parameter_registry, unpack_to_physical

        n_pulsars = 3
        registry = build_parameter_registry(cw_prior_specs_all_sampled, n_pulsars)

        x_flat = jnp.zeros(registry.ndim)
        params = unpack_to_physical(x_flat, registry)

        # At x=0, physical = mean for all simple affine params
        assert jnp.isclose(params["log10_h0"], -14.0, atol=1e-6)
        assert jnp.isclose(params["log10_f_gw"], -8.0, atol=1e-6)
        assert jnp.isclose(params["cos_iota"], 0.0, atol=1e-6)

    def test_delta_gw_derived(self, cw_prior_specs_all_sampled):
        """delta_gw should be arcsin(sin_delta_gw)."""
        from argus.tempered_smc import build_parameter_registry, unpack_to_physical

        n_pulsars = 3
        registry = build_parameter_registry(cw_prior_specs_all_sampled, n_pulsars)

        x_flat = jnp.zeros(registry.ndim)
        params = unpack_to_physical(x_flat, registry)

        assert "delta_gw" in params
        assert jnp.isclose(params["delta_gw"], jnp.arcsin(params["sin_delta_gw"]), atol=1e-10)

    def test_fixed_values_injected(self, cw_prior_specs_some_fixed):
        """Fixed parameters should appear in output dict."""
        from argus.tempered_smc import build_parameter_registry, unpack_to_physical

        n_pulsars = 3
        registry = build_parameter_registry(cw_prior_specs_some_fixed, n_pulsars)

        x_flat = jnp.zeros(registry.ndim)
        params = unpack_to_physical(x_flat, registry)

        assert "alpha_gw" in params
        assert jnp.isclose(params["alpha_gw"], 4.067, atol=1e-6)

    def test_hierarchical_coupling(self, cw_prior_specs_hierarchical):
        """Hierarchical gamma should be coupled: gp = mean + raw * std / sqrt(n)."""
        from argus.tempered_smc import build_parameter_registry, unpack_to_physical

        n_pulsars = 3
        registry = build_parameter_registry(cw_prior_specs_hierarchical, n_pulsars)

        x_flat = jnp.zeros(registry.ndim)
        params = unpack_to_physical(x_flat, registry)

        # At x=0, mean_phys = midpoint of [-9, -7] = -8.0
        # raw = 0 => gp = mean_phys + 0 = -8.0
        assert jnp.allclose(params["log10_γp"], -8.0, atol=1e-5)

    def test_sigma_derived_from_ratio(self, cw_prior_specs_hierarchical):
        """log10_sigma_p should be gamma_p + ratio."""
        from argus.tempered_smc import build_parameter_registry, unpack_to_physical

        n_pulsars = 3
        registry = build_parameter_registry(cw_prior_specs_hierarchical, n_pulsars)

        x_flat = jnp.zeros(registry.ndim)
        params = unpack_to_physical(x_flat, registry)

        # ratio_mean at x=0 = midpoint of [-8, -4] = -6.0
        # ratio_raw = 0 => ratio = -6.0
        # sigma = gamma + ratio = -8.0 + (-6.0) = -14.0
        assert jnp.allclose(params["log10_σp"], -14.0, atol=1e-5)


# ============================================================
# Tests for log-probability functions
# ============================================================


class TestLogProbFunctions:
    """Tests for log-prior and log-likelihood construction."""

    def test_logprior_at_zero(self, cw_prior_specs_all_sampled):
        """Log-prior at origin should be -0.5 * 0 = 0."""
        from argus.tempered_smc import build_parameter_registry, build_logprior_fn

        registry = build_parameter_registry(cw_prior_specs_all_sampled, n_pulsars=3)
        logprior_fn = build_logprior_fn(registry)

        x = jnp.zeros(registry.ndim)
        assert jnp.isclose(logprior_fn(x), 0.0, atol=1e-10)

    def test_logprior_negative_away_from_zero(self, cw_prior_specs_all_sampled):
        """Log-prior should decrease away from origin."""
        from argus.tempered_smc import build_parameter_registry, build_logprior_fn

        registry = build_parameter_registry(cw_prior_specs_all_sampled, n_pulsars=3)
        logprior_fn = build_logprior_fn(registry)

        x = jnp.ones(registry.ndim)
        assert logprior_fn(x) < 0.0

    def test_logprior_is_jittable(self, cw_prior_specs_all_sampled):
        """Log-prior should be JIT-compilable."""
        from argus.tempered_smc import build_parameter_registry, build_logprior_fn

        registry = build_parameter_registry(cw_prior_specs_all_sampled, n_pulsars=3)
        logprior_fn = build_logprior_fn(registry)

        jitted = jax.jit(logprior_fn)
        x = jnp.zeros(registry.ndim)
        assert jnp.isfinite(jitted(x))

    def test_loglikelihood_returns_finite(self):
        """Log-likelihood should return a finite value for reasonable parameters."""
        from argus.tempered_smc import (
            build_parameter_registry, build_loglikelihood_fn,
        )
        from argus.cw_kalman_filter import CWKalmanFilter
        import pandas as pd

        # Build minimal synthetic data
        np.random.seed(42)
        Npsr = 2
        nobs = 30

        toas_list = [np.sort(np.random.uniform(0, 1e9, nobs)) for _ in range(Npsr)]
        residuals_list = [np.random.normal(0, 1e-7, nobs) for _ in range(Npsr)]
        errors_list = [np.full(nobs, 1e-7) for _ in range(Npsr)]

        metadata = pd.DataFrame({
            "name": [f"J000{i}+0001" for i in range(Npsr)],
            "dim_M": [3] * Npsr,
            "RA": [0.5, 1.5],
            "DEC": [0.3, -0.2],
            "F0": [200.0, 300.0],
        })
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

        # Simple prior specs with noise fixed
        cw_specs = {
            "log10_h0_spec": None,
            "log10_h0_transform_params": {"mean": -14.0, "std": 0.667},
            "alpha_gw_spec": None,
            "alpha_gw_transform_params": {"mean": 3.1416, "std": 1.0472},
            "sin_delta_gw_spec": None,
            "sin_delta_gw_transform_params": {"mean": 0.0, "std": 0.333},
            "delta_gw_spec": None,
            "log10_f_gw_spec": None,
            "log10_f_gw_transform_params": {"mean": -8.0, "std": 0.333},
            "cos_iota_spec": None,
            "cos_iota_transform_params": {"mean": 0.0, "std": 0.333},
            "psi_spec": None,
            "psi_transform_params": {"mean": 1.5708, "std": 0.5236},
            "Phi0_spec": None,
            "Phi0_transform_params": {"mean": 3.1416, "std": 1.0472},
            "chi_transform_params": None,
        }
        prior_specs = {
            "cw_specs": cw_specs,
            "log10_gamma_p_spec": jnp.full(Npsr, -8.0),
            "log10_sigma_p_spec": jnp.full(Npsr, -15.0),
            "efac_spec": jnp.ones(Npsr),
            "equad_spec": jnp.full(Npsr, 1e-7),
            "hierarchical_specs": None,
        }

        registry = build_parameter_registry(prior_specs, Npsr)
        loglikelihood_fn = build_loglikelihood_fn(kf, registry, Npsr)

        x = jnp.zeros(registry.ndim)
        ll = loglikelihood_fn(x)
        assert jnp.isfinite(ll)


# ============================================================
# Tests for ArviZ conversion
# ============================================================


class TestSmcArvizConversion:
    """Tests for smc_results_to_arviz."""

    def test_produces_inference_data(self, cw_prior_specs_all_sampled):
        """Conversion should produce a valid ArviZ InferenceData object."""
        import arviz as az
        from argus.tempered_smc import (
            build_parameter_registry, smc_results_to_arviz,
        )

        n_pulsars = 3
        registry = build_parameter_registry(cw_prior_specs_all_sampled, n_pulsars)

        # Fake particles in unconstrained space
        particles = jnp.zeros((50, registry.ndim))

        inf_data = smc_results_to_arviz(particles, registry, n_pulsars)

        assert isinstance(inf_data, az.InferenceData)
        assert hasattr(inf_data, "posterior")

    def test_contains_cw_parameters(self, cw_prior_specs_all_sampled):
        """Output should contain CW parameter names."""
        from argus.tempered_smc import (
            build_parameter_registry, smc_results_to_arviz,
        )

        n_pulsars = 3
        registry = build_parameter_registry(cw_prior_specs_all_sampled, n_pulsars)
        particles = jnp.zeros((50, registry.ndim))

        inf_data = smc_results_to_arviz(particles, registry, n_pulsars)

        posterior_vars = list(inf_data.posterior.data_vars)
        assert "log10_h0" in posterior_vars
        assert "log10_f_gw" in posterior_vars
        assert "delta_gw" in posterior_vars

    def test_single_chain_shape(self, cw_prior_specs_all_sampled):
        """Output should have chain dimension = 1 for SMC."""
        from argus.tempered_smc import (
            build_parameter_registry, smc_results_to_arviz,
        )

        n_pulsars = 3
        registry = build_parameter_registry(cw_prior_specs_all_sampled, n_pulsars)
        particles = jnp.zeros((50, registry.ndim))

        inf_data = smc_results_to_arviz(particles, registry, n_pulsars)

        assert inf_data.posterior["log10_h0"].shape[0] == 1  # 1 chain
        assert inf_data.posterior["log10_h0"].shape[1] == 50  # 50 draws


# ============================================================
# Tests for temperature schedule
# ============================================================


class TestTemperatureSchedule:
    """Tests for build_temperature_schedule."""

    def test_geometric_endpoints(self):
        """Geometric schedule should end at 1.0."""
        from argus.tempered_smc import build_temperature_schedule

        schedule = build_temperature_schedule(10, "geometric")
        assert jnp.isclose(schedule[-1], 1.0, atol=1e-10)
        assert schedule[0] > 0.0

    def test_linear_endpoints(self):
        """Linear schedule should end at 1.0."""
        from argus.tempered_smc import build_temperature_schedule

        schedule = build_temperature_schedule(10, "linear")
        assert jnp.isclose(schedule[-1], 1.0, atol=1e-10)
        assert jnp.isclose(schedule[0], 0.1, atol=1e-10)

    def test_monotonically_increasing(self):
        """Schedule should be monotonically increasing."""
        from argus.tempered_smc import build_temperature_schedule

        for spacing in ["geometric", "linear"]:
            schedule = build_temperature_schedule(20, spacing)
            diffs = jnp.diff(schedule)
            assert jnp.all(diffs > 0)
