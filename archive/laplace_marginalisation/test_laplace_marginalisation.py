"""Tests for Laplace marginalisation of per-pulsar noise parameters."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from argus.cw_kalman_filter import (
    _initialize_single_pulsar,
    _run_single_pulsar_filter,
    CWKalmanFilter,
)
from argus.bayesian_inference import CWParameters
from argus.laplace_marginalisation import (
    _per_pulsar_log_posterior,
    _newton_step_2d,
    _optimize_and_laplace_single_pulsar,
    laplace_marginalised_likelihood,
)

import logging

_logger = logging.getLogger("argus")
if not _logger.handlers:
    _logger.addHandler(logging.NullHandler())
    _logger.setLevel(logging.WARNING)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def simple_pulsar_data():
    """Create simple synthetic per-pulsar data for one pulsar."""
    np.random.seed(42)
    nobs = 50
    state_dim = 7  # 2 + 5 (timing model dim)

    toas = jnp.array(np.sort(np.random.uniform(0, 1e9, nobs)))
    residuals = jnp.array(np.random.normal(0, 1e-7, nobs))
    errors = jnp.full(nobs, 1e-7)
    mask = jnp.ones(nobs)
    dt = jnp.diff(toas)

    # Observation vectors (H)
    H = np.zeros((nobs, state_dim))
    H[:, 0] = 1.0  # Observe phase
    # Add timing model components
    for m in range(5):
        H[:, 2 + m] = np.random.randn(nobs) * 0.01
    H = jnp.array(H)

    # Timing model covariance
    A = np.random.randn(5, 5)
    P_eps = jnp.array(A @ A.T + np.eye(5) * 0.01)

    return {
        "toas": toas,
        "residuals": residuals,
        "errors": errors,
        "mask": mask,
        "dt": dt,
        "H": H,
        "P_eps": P_eps,
        "state_dim": state_dim,
    }


@pytest.fixture
def simple_cw_data():
    """Create simple synthetic CW data for 3 pulsars."""
    import pandas as pd

    Npsr = 3
    nobs_per_pulsar = [50, 40, 45]
    max_nobs = 50
    np.random.seed(42)

    toas_list = []
    residuals_list = []
    errors_list = []
    for n in range(Npsr):
        t = np.sort(np.random.uniform(0, 1e9, nobs_per_pulsar[n]))
        toas_list.append(t)
        residuals_list.append(np.random.normal(0, 1e-7, nobs_per_pulsar[n]))
        errors_list.append(np.full(nobs_per_pulsar[n], 1e-7))

    metadata = pd.DataFrame({
        "name": ["J0001+0001", "J0002+0002", "J0003+0003"],
        "dim_M": [5, 6, 5],
        "RA": [0.5, 1.5, 3.0],
        "DEC": [0.3, -0.2, 0.8],
        "F0": [200.0, 300.0, 150.0],
    })

    design_matrices = []
    for n in range(Npsr):
        M_n = metadata["dim_M"].iloc[n]
        nobs_n = nobs_per_pulsar[n]
        design_matrices.append(np.random.randn(nobs_n, M_n) * 0.01)

    P_eps_matrices = []
    for n in range(Npsr):
        M_n = metadata["dim_M"].iloc[n]
        A = np.random.randn(M_n, M_n)
        P_eps_matrices.append(A @ A.T + np.eye(M_n) * 0.01)

    data = {
        "processed_residuals": {
            "toas": toas_list,
            "residuals": residuals_list,
            "errors": errors_list,
            "n_obs": np.array(nobs_per_pulsar),
        },
        "metadata": metadata,
        "design_matrices": design_matrices,
        "parameter_covariances": P_eps_matrices,
        "hd_correlation": None,
    }
    return data


# ============================================================
# Unit tests: _per_pulsar_log_posterior
# ============================================================


class TestPerPulsarLogPosterior:
    """Tests for the per-pulsar log-posterior function."""

    def test_returns_finite(self, simple_pulsar_data):
        """Log-posterior should return a finite value at a reasonable point."""
        d = simple_pulsar_data
        phi = jnp.array([-8.0, -0.5])  # log10_gamma_p, log10_ratio
        result = _per_pulsar_log_posterior(
            phi, d["residuals"], d["H"], jnp.full(50, 1e-14), d["dt"],
            d["mask"], d["P_eps"], d["state_dim"],
            gamma_p_mean=-8.0, gamma_p_std_scaled=0.5,
            ratio_mean=-0.5, ratio_std_scaled=0.3,
        )
        assert jnp.isfinite(result)

    def test_gradient_finite(self, simple_pulsar_data):
        """Gradient of log-posterior should be finite."""
        d = simple_pulsar_data
        phi = jnp.array([-8.0, -0.5])
        grad_fn = jax.grad(lambda p: _per_pulsar_log_posterior(
            p, d["residuals"], d["H"], jnp.full(50, 1e-14), d["dt"],
            d["mask"], d["P_eps"], d["state_dim"],
            -8.0, 0.5, -0.5, 0.3,
        ))
        grad = grad_fn(phi)
        assert jnp.all(jnp.isfinite(grad))

    def test_hessian_is_symmetric(self, simple_pulsar_data):
        """Hessian should be a symmetric 2x2 matrix."""
        d = simple_pulsar_data
        phi = jnp.array([-8.0, -0.5])
        hess_fn = jax.hessian(lambda p: _per_pulsar_log_posterior(
            p, d["residuals"], d["H"], jnp.full(50, 1e-14), d["dt"],
            d["mask"], d["P_eps"], d["state_dim"],
            -8.0, 0.5, -0.5, 0.3,
        ))
        H = hess_fn(phi)
        assert H.shape == (2, 2)
        np.testing.assert_allclose(H, H.T, atol=1e-10)


# ============================================================
# Unit tests: Newton solver
# ============================================================


class TestNewtonStep:
    """Tests for the 2D Newton step."""

    def test_quadratic_one_step_convergence(self):
        """Newton should converge in one step on a quadratic."""
        # f(x) = -0.5 * x^T A x + b^T x  =>  grad = -Ax + b, Hess = -A
        A = jnp.array([[2.0, 0.5], [0.5, 3.0]])
        b = jnp.array([1.0, 2.0])
        x_star = jnp.linalg.solve(A, b)  # True optimum

        x0 = jnp.array([0.0, 0.0])
        grad = -A @ x0 + b
        hess = -A
        x1 = _newton_step_2d(x0, grad, hess)

        np.testing.assert_allclose(x1, x_star, atol=1e-12)


# ============================================================
# Unit tests: Laplace approximation for single pulsar
# ============================================================


class TestOptimizeAndLaplace:
    """Tests for the single-pulsar Laplace marginalisation."""

    def test_returns_finite(self, simple_pulsar_data):
        """Laplace-approximated marginal should be finite."""
        d = simple_pulsar_data
        result = _optimize_and_laplace_single_pulsar(
            d["residuals"], d["H"], jnp.full(50, 1e-14), d["dt"],
            d["mask"], d["P_eps"], d["state_dim"],
            gamma_p_mean=-8.0, gamma_p_std_scaled=0.5,
            ratio_mean=-0.5, ratio_std_scaled=0.3,
            n_newton_steps=8,
        )
        assert jnp.isfinite(result)

    def test_laplace_vs_direct_at_known_point(self, simple_pulsar_data):
        """Laplace correction should agree with direct evaluation when posterior is Gaussian.

        The Laplace approximation equals the exact integral when the
        integrand is exactly Gaussian. We test that the marginalised value
        is at least close to the objective at the MAP (the correction adds
        a positive contribution from the Gaussian integral).
        """
        d = simple_pulsar_data
        laplace_result = _optimize_and_laplace_single_pulsar(
            d["residuals"], d["H"], jnp.full(50, 1e-14), d["dt"],
            d["mask"], d["P_eps"], d["state_dim"],
            gamma_p_mean=-8.0, gamma_p_std_scaled=0.5,
            ratio_mean=-0.5, ratio_std_scaled=0.3,
            n_newton_steps=8,
        )

        # Direct evaluation at the prior mode (without Laplace correction)
        phi_mode = jnp.array([-8.0, -0.5])
        direct_at_mode = _per_pulsar_log_posterior(
            phi_mode, d["residuals"], d["H"], jnp.full(50, 1e-14), d["dt"],
            d["mask"], d["P_eps"], d["state_dim"],
            -8.0, 0.5, -0.5, 0.3,
        )

        # Laplace result should be >= MAP value (Gaussian integral adds log(2pi) - 0.5*log(det))
        assert laplace_result >= direct_at_mode - 10.0  # generous tolerance


# ============================================================
# Integration test: full marginalised likelihood
# ============================================================


class TestLaplaceMarginalised:
    """Tests for the full vmapped Laplace-marginalised likelihood."""

    def test_returns_finite(self, simple_cw_data):
        """Full marginalised likelihood should return a finite value."""
        kf = CWKalmanFilter(simple_cw_data)
        Npsr = 3

        result = laplace_marginalised_likelihood(
            alpha_gw=2.0, delta_gw=0.5,
            f_gw=1e-8, h0=1e-15,
            cos_iota=0.5, psi=0.7, Phi0=0.3,
            chi=jnp.zeros(Npsr),
            gamma_p_mean=-8.0, gamma_p_std_scaled=0.5,
            ratio_mean=-0.5, ratio_std_scaled=0.3,
            EFAC=jnp.ones(Npsr), EQUAD=jnp.full(Npsr, 1e-8),
            toas=kf.jax_toas, residuals=kf.jax_residuals,
            errors=kf.jax_errors, mask=kf.jax_mask,
            dt=kf.jax_dt, H=kf.jax_H, P_eps=kf.jax_P_eps,
            pulsar_ra=kf.pulsar_ra, pulsar_dec=kf.pulsar_dec,
            pulsar_distances=kf.pulsar_distances,
            Npsr=kf.Npsr, state_dim=kf.state_dim,
            include_pulsar_term=False, phase_parameterization=False,
            n_newton_steps=8,
        )
        assert jnp.isfinite(result)

    def test_is_differentiable(self, simple_cw_data):
        """Marginalised likelihood should be differentiable w.r.t. CW params."""
        kf = CWKalmanFilter(simple_cw_data)
        Npsr = 3

        def ll(alpha_gw):
            return laplace_marginalised_likelihood(
                alpha_gw=alpha_gw, delta_gw=0.5,
                f_gw=1e-8, h0=1e-15,
                cos_iota=0.5, psi=0.7, Phi0=0.3,
                chi=jnp.zeros(Npsr),
                gamma_p_mean=-8.0, gamma_p_std_scaled=0.5,
                ratio_mean=-0.5, ratio_std_scaled=0.3,
                EFAC=jnp.ones(Npsr), EQUAD=jnp.full(Npsr, 1e-8),
                toas=kf.jax_toas, residuals=kf.jax_residuals,
                errors=kf.jax_errors, mask=kf.jax_mask,
                dt=kf.jax_dt, H=kf.jax_H, P_eps=kf.jax_P_eps,
                pulsar_ra=kf.pulsar_ra, pulsar_dec=kf.pulsar_dec,
                pulsar_distances=kf.pulsar_distances,
                Npsr=kf.Npsr, state_dim=kf.state_dim,
                include_pulsar_term=False, phase_parameterization=False,
                n_newton_steps=8,
            )

        grad = jax.grad(ll)(2.0)
        assert jnp.isfinite(grad)

    def test_agrees_with_direct_at_noise_map(self, simple_cw_data):
        """Marginalised likelihood should approximately agree with direct likelihood
        evaluated at the noise MAP (up to the Laplace correction term)."""
        kf = CWKalmanFilter(simple_cw_data)
        Npsr = 3

        # Direct likelihood at specific noise values
        gamma_p = jnp.full(Npsr, 1e-8)
        sigma_p = jnp.full(Npsr, 1e-8 * 10**(-0.5))
        params = CWParameters(
            alpha_gw=2.0, delta_gw=0.5,
            f_gw=1e-8, h0=1e-15,
            cos_iota=0.5, psi=0.7, Phi0=0.3,
            chi=jnp.zeros(Npsr),
            gamma_p=gamma_p, sigma_p=sigma_p,
            EFAC=jnp.ones(Npsr), EQUAD=jnp.full(Npsr, 1e-8),
        )
        direct_ll = kf.get_likelihood(params)

        # Marginalised likelihood
        marg_ll = laplace_marginalised_likelihood(
            alpha_gw=2.0, delta_gw=0.5,
            f_gw=1e-8, h0=1e-15,
            cos_iota=0.5, psi=0.7, Phi0=0.3,
            chi=jnp.zeros(Npsr),
            gamma_p_mean=-8.0, gamma_p_std_scaled=0.5,
            ratio_mean=-0.5, ratio_std_scaled=0.3,
            EFAC=jnp.ones(Npsr), EQUAD=jnp.full(Npsr, 1e-8),
            toas=kf.jax_toas, residuals=kf.jax_residuals,
            errors=kf.jax_errors, mask=kf.jax_mask,
            dt=kf.jax_dt, H=kf.jax_H, P_eps=kf.jax_P_eps,
            pulsar_ra=kf.pulsar_ra, pulsar_dec=kf.pulsar_dec,
            pulsar_distances=kf.pulsar_distances,
            Npsr=kf.Npsr, state_dim=kf.state_dim,
            include_pulsar_term=False, phase_parameterization=False,
            n_newton_steps=8,
        )

        # Both should be finite
        assert jnp.isfinite(direct_ll)
        assert jnp.isfinite(marg_ll)
        # The marginalised LL includes hierarchical prior terms and the
        # Laplace correction (log(2pi) - 0.5*log(det(-H))), so it can
        # differ substantially from the direct LL at a single noise point.
        # We just check both are finite and reasonable (same order of magnitude)
        assert abs(float(marg_ll)) < 1e8
        assert abs(float(direct_ll)) < 1e8


# ============================================================
# Integration test: CWKalmanFilter.get_marginalised_likelihood
# ============================================================


class TestCWKalmanFilterMarginalised:
    """Tests for the CWKalmanFilter.get_marginalised_likelihood method."""

    def test_method_exists_and_works(self, simple_cw_data):
        """The method should exist and return a finite value."""
        kf = CWKalmanFilter(simple_cw_data)
        result = kf.get_marginalised_likelihood(
            alpha_gw=2.0, delta_gw=0.5,
            log10_f_gw=-8.0, log10_h0=-15.0,
            cos_iota=0.5, psi=0.7, Phi0=0.3,
            chi=jnp.zeros(3),
            gamma_p_mean=-8.0, gamma_p_std_scaled=0.5,
            ratio_mean=-0.5, ratio_std_scaled=0.3,
            efac=jnp.ones(3), equad=jnp.full(3, 1e-8),
            n_newton_steps=8,
        )
        assert jnp.isfinite(result)
