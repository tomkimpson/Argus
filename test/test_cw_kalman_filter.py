"""Tests for the CW per-pulsar scalar Kalman filter."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from argus.cw_kalman_filter import (
    _build_per_pulsar_H_vectors,
    _build_per_pulsar_P_eps_padded,
    _cw_scalar_update,
    _cw_scalar_log_likelihood,
    _build_F_single,
    _build_Q_single,
    _initialize_single_pulsar,
    _run_single_pulsar_filter,
    _cw_likelihood,
    CWKalmanFilter,
)
from argus.bayesian_inference import CWParameters


# ============================================================
# Setup logger for tests
# ============================================================

import logging

_logger = logging.getLogger("argus")
if not _logger.handlers:
    _logger.addHandler(logging.NullHandler())
    _logger.setLevel(logging.WARNING)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def simple_cw_data():
    """Create simple synthetic per-pulsar data for testing."""
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

    # Metadata
    import pandas as pd

    metadata = pd.DataFrame({
        "name": ["J0001+0001", "J0002+0002", "J0003+0003"],
        "dim_M": [5, 6, 5],
        "RA": [0.5, 1.5, 3.0],
        "DEC": [0.3, -0.2, 0.8],
        "F0": [200.0, 300.0, 150.0],
    })

    # Design matrices (random for testing)
    design_matrices = []
    for n in range(Npsr):
        M_n = metadata["dim_M"].iloc[n]
        nobs_n = nobs_per_pulsar[n]
        dm = np.random.randn(nobs_n, M_n) * 0.01
        design_matrices.append(dm)

    # P_eps matrices
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


@pytest.fixture
def sample_cw_params():
    """Create sample CW parameter values for testing."""
    return CWParameters(
        alpha_gw=2.0,
        delta_gw=0.5,
        f_gw=1e-8,
        h0=1e-15,
        cos_iota=0.5,
        psi=0.7,
        Phi0=0.3,
        chi=jnp.zeros(3),
        gamma_p=jnp.array([1e-8, 1e-8, 1e-8]),
        sigma_p=jnp.array([1e-15, 1e-15, 1e-15]),
        EFAC=jnp.ones(3),
        EQUAD=jnp.full(3, 1e-8),
    )


# ============================================================
# Tests for building blocks
# ============================================================


class TestBuildPerPulsarHVectors:
    """Tests for H vector construction."""

    def test_shape(self):
        """H vectors should have correct shape."""
        Npsr, max_nobs, max_M = 3, 50, 6
        f0 = np.array([200.0, 300.0, 150.0])
        M_dims = np.array([5, 6, 5])
        design_matrices = [np.random.randn(50, 5), np.random.randn(50, 6), np.random.randn(50, 5)]

        H = _build_per_pulsar_H_vectors(Npsr, max_nobs, max_M, f0, design_matrices, M_dims)
        assert H.shape == (Npsr, max_nobs, 2 + max_M)

    def test_spin_coefficient(self):
        """First entry of H should be 1/f0 for each pulsar."""
        Npsr, max_nobs, max_M = 2, 10, 3
        f0 = np.array([200.0, 300.0])
        M_dims = np.array([3, 3])
        design_matrices = [np.zeros((10, 3)), np.zeros((10, 3))]

        H = _build_per_pulsar_H_vectors(Npsr, max_nobs, max_M, f0, design_matrices, M_dims)
        assert np.allclose(H[0, 0, 0], 1.0 / 200.0)
        assert np.allclose(H[1, 0, 0], 1.0 / 300.0)

    def test_zero_padding(self):
        """Unused entries should be zero."""
        Npsr, max_nobs, max_M = 2, 10, 5
        f0 = np.array([200.0, 300.0])
        M_dims = np.array([3, 5])
        design_matrices = [np.random.randn(8, 3), np.random.randn(10, 5)]

        H = _build_per_pulsar_H_vectors(Npsr, max_nobs, max_M, f0, design_matrices, M_dims)
        # Pulsar 0 has M=3, so entries for M indices 3,4 should be zero
        assert np.allclose(H[0, :, 2 + 3:], 0.0)
        # Pulsar 0 has 8 obs, so entries beyond index 8 should be zero
        assert np.allclose(H[0, 8:, :], 0.0)


class TestScalarKalmanUpdate:
    """Tests for the scalar Kalman update."""

    def test_innovation_correct(self):
        """Innovation should be z - h^T x."""
        state_dim = 4
        x = jnp.ones((state_dim, 1)) * 0.5
        P = jnp.eye(state_dim) * 0.1
        h = jnp.array([1.0, 0.0, 0.5, 0.0])
        R = 0.01
        z = 1.0

        x_new, P_new, nu, S = _cw_scalar_update(x, P, h, R, z)

        expected_nu = z - h @ x[:, 0]
        assert jnp.allclose(nu, expected_nu, atol=1e-14)

    def test_innovation_variance_positive(self):
        """S should always be positive."""
        state_dim = 5
        x = jnp.zeros((state_dim, 1))
        P = jnp.eye(state_dim) * 0.01
        h = jnp.array([1.0, 0.0, 0.1, 0.2, 0.0])
        R = 0.001
        z = 0.5

        _, _, _, S = _cw_scalar_update(x, P, h, R, z)
        assert S > 0

    def test_joseph_form_symmetric(self):
        """Updated covariance should be symmetric."""
        state_dim = 4
        x = jnp.zeros((state_dim, 1))
        P = jnp.eye(state_dim)
        h = jnp.array([1.0, 0.5, 0.0, 0.2])
        R = 0.1
        z = 0.3

        _, P_new, _, _ = _cw_scalar_update(x, P, h, R, z)
        assert jnp.allclose(P_new, P_new.T, atol=1e-14)


class TestBuildFSingle:
    """Tests for per-pulsar transition matrix."""

    def test_identity_for_timing(self):
        """Timing model block should be identity."""
        state_dim = 7  # 2 spin + 5 timing
        F = _build_F_single(1e-8, state_dim, 86400.0)
        assert jnp.allclose(F[2:, 2:], jnp.eye(5), atol=1e-14)

    def test_spin_block_matches_get_F_block(self):
        """Spin block should match get_F_block output."""
        from argus.model import get_F_block

        gamma_p = 1e-8
        dt = 86400.0
        state_dim = 7

        F = _build_F_single(gamma_p, state_dim, dt)
        F_block_expected = get_F_block(gamma_p, dt)
        assert jnp.allclose(F[:2, :2], F_block_expected, atol=1e-14)


# ============================================================
# Tests for the full filter
# ============================================================


class TestCWKalmanFilter:
    """Tests for the CWKalmanFilter class."""

    def test_initialization(self, simple_cw_data):
        """Filter should initialize correctly from data dict."""
        kf = CWKalmanFilter(simple_cw_data)
        assert kf.Npsr == 3
        assert kf.state_dim == 2 + 6  # max_M = 6
        assert kf.max_nobs == 50
        assert kf.jax_toas.shape == (3, 50)
        assert kf.jax_H.shape == (3, 50, 8)

    def test_likelihood_finite(self, simple_cw_data, sample_cw_params):
        """Likelihood should return a finite value."""
        kf = CWKalmanFilter(simple_cw_data)
        ll = kf.get_likelihood(sample_cw_params)
        assert jnp.isfinite(ll)

    def test_zero_cw_signal(self, simple_cw_data):
        """With h0=0, CW signal is zero and filter reduces to noise-only."""
        kf = CWKalmanFilter(simple_cw_data)

        params_zero = CWParameters(
            alpha_gw=2.0,
            delta_gw=0.5,
            f_gw=1e-8,
            h0=0.0,
            cos_iota=0.5,
            psi=0.7,
            Phi0=0.3,
            chi=jnp.zeros(3),
            gamma_p=jnp.array([1e-8, 1e-8, 1e-8]),
            sigma_p=jnp.array([1e-15, 1e-15, 1e-15]),
            EFAC=jnp.ones(3),
            EQUAD=jnp.full(3, 1e-8),
        )

        params_nonzero = CWParameters(
            alpha_gw=2.0,
            delta_gw=0.5,
            f_gw=1e-8,
            h0=0.0,  # Still zero
            cos_iota=0.5,
            psi=0.7,
            Phi0=1.5,  # Different phase, but h0=0 so doesn't matter
            chi=jnp.zeros(3),
            gamma_p=jnp.array([1e-8, 1e-8, 1e-8]),
            sigma_p=jnp.array([1e-15, 1e-15, 1e-15]),
            EFAC=jnp.ones(3),
            EQUAD=jnp.full(3, 1e-8),
        )

        ll1 = kf.get_likelihood(params_zero)
        ll2 = kf.get_likelihood(params_nonzero)
        # With h0=0, changing other CW params shouldn't matter
        assert jnp.allclose(ll1, ll2, atol=1e-10)

    def test_likelihood_gradient(self, simple_cw_data):
        """JAX should be able to compute gradient of likelihood w.r.t. CW params."""
        kf = CWKalmanFilter(simple_cw_data)

        def ll_fn(h0, f_gw, alpha_gw):
            params = CWParameters(
                alpha_gw=alpha_gw,
                delta_gw=0.5,
                f_gw=f_gw,
                h0=h0,
                cos_iota=0.5,
                psi=0.7,
                Phi0=0.3,
                chi=jnp.zeros(3),
                gamma_p=jnp.array([1e-8, 1e-8, 1e-8]),
                sigma_p=jnp.array([1e-15, 1e-15, 1e-15]),
                EFAC=jnp.ones(3),
                EQUAD=jnp.full(3, 1e-8),
            )
            return kf.get_likelihood(params)

        grad_fn = jax.grad(ll_fn, argnums=(0, 1, 2))
        grads = grad_fn(1e-15, 1e-8, 2.0)
        for g in grads:
            assert jnp.isfinite(g), f"Non-finite gradient: {g}"

    def test_likelihood_changes_with_h0(self, simple_cw_data):
        """Likelihood should change when h0 changes."""
        kf = CWKalmanFilter(simple_cw_data)

        def make_params(h0):
            return CWParameters(
                alpha_gw=2.0,
                delta_gw=0.5,
                f_gw=1e-8,
                h0=h0,
                cos_iota=0.5,
                psi=0.7,
                Phi0=0.3,
                chi=jnp.zeros(3),
                gamma_p=jnp.array([1e-8, 1e-8, 1e-8]),
                sigma_p=jnp.array([1e-15, 1e-15, 1e-15]),
                EFAC=jnp.ones(3),
                EQUAD=jnp.full(3, 1e-8),
            )

        ll1 = kf.get_likelihood(make_params(0.0))
        ll2 = kf.get_likelihood(make_params(1e-5))  # Large amplitude to ensure measurable difference
        assert not jnp.allclose(ll1, ll2)


class TestPhaseParameterizedLikelihood:
    """Tests for phase-reparameterized pulsar term likelihood."""

    def test_phase_param_likelihood_finite(self, simple_cw_data):
        """Likelihood with phase parameterization should return finite value."""
        kf = CWKalmanFilter(
            simple_cw_data, include_pulsar_term=True, phase_parameterization=True,
        )
        params = CWParameters(
            alpha_gw=2.0, delta_gw=0.5, f_gw=1e-8, h0=1e-15,
            cos_iota=0.5, psi=0.7, Phi0=0.3,
            chi=jnp.array([1.0, 2.0, 3.0]),
            gamma_p=jnp.array([1e-8, 1e-8, 1e-8]),
            sigma_p=jnp.array([1e-15, 1e-15, 1e-15]),
            EFAC=jnp.ones(3), EQUAD=jnp.full(3, 1e-8),
        )
        ll = kf.get_likelihood(params)
        assert jnp.isfinite(ll)

    def test_phase_param_gradient_wrt_chi(self, simple_cw_data):
        """Gradients w.r.t. chi elements should be finite (required for NUTS)."""
        kf = CWKalmanFilter(
            simple_cw_data, include_pulsar_term=True, phase_parameterization=True,
        )

        def ll_fn(chi):
            params = CWParameters(
                alpha_gw=2.0, delta_gw=0.5, f_gw=1e-8, h0=1e-15,
                cos_iota=0.5, psi=0.7, Phi0=0.3, chi=chi,
                gamma_p=jnp.array([1e-8, 1e-8, 1e-8]),
                sigma_p=jnp.array([1e-15, 1e-15, 1e-15]),
                EFAC=jnp.ones(3), EQUAD=jnp.full(3, 1e-8),
            )
            return kf.get_likelihood(params)

        grad = jax.grad(ll_fn)(jnp.array([1.0, 2.0, 3.0]))
        assert jnp.all(jnp.isfinite(grad))

    def test_phase_param_chi_changes_likelihood(self, simple_cw_data):
        """Different chi values should produce different likelihoods."""
        kf = CWKalmanFilter(
            simple_cw_data, include_pulsar_term=True, phase_parameterization=True,
        )

        def make_params(chi):
            return CWParameters(
                alpha_gw=2.0, delta_gw=0.5, f_gw=1e-8, h0=1e-5,
                cos_iota=0.5, psi=0.7, Phi0=0.3, chi=chi,
                gamma_p=jnp.array([1e-8, 1e-8, 1e-8]),
                sigma_p=jnp.array([1e-15, 1e-15, 1e-15]),
                EFAC=jnp.ones(3), EQUAD=jnp.full(3, 1e-8),
            )

        ll1 = kf.get_likelihood(make_params(jnp.zeros(3)))
        ll2 = kf.get_likelihood(make_params(jnp.array([1.0, 2.0, 3.0])))
        assert not jnp.allclose(ll1, ll2)


class TestScalarLogLikelihood:
    """Tests for scalar log-likelihood computation."""

    def test_negative_for_large_innovation(self):
        """Large innovation should give negative log-likelihood."""
        ll = _cw_scalar_log_likelihood(100.0, 1.0)
        assert ll < 0

    def test_larger_with_smaller_innovation(self):
        """Smaller innovation should give larger log-likelihood."""
        ll_small = _cw_scalar_log_likelihood(0.01, 1.0)
        ll_large = _cw_scalar_log_likelihood(10.0, 1.0)
        assert ll_small > ll_large
