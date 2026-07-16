"""Unit tests for jax_kalman_filter module."""

import pytest
import numpy as np
import jax.numpy as jnp
from unittest.mock import Mock, patch
from argus import jax_kalman_filter, bayesian_inference


class TestComputePredictedState:
    """Tests for compute_predicted_state function."""

    def test_basic_state_prediction(self):
        """Test basic state prediction."""
        # Setup simple transition matrices
        F_gw = jnp.eye(4)  # Identity for GW states
        F_spin = jnp.eye(4)  # Identity for spin states
        F_list = (F_gw, F_spin)

        # Current state: 4 GW + 4 spin + 2 timing = 10 total
        x = jnp.arange(10).reshape(-1, 1).astype(float)

        x_pred = jax_kalman_filter.compute_predicted_state(
            F_list, x, gw_size=4, spin_size=4
        )

        # With identity matrices, prediction should equal input
        assert jnp.allclose(x_pred, x)

    def test_timing_states_unchanged(self):
        """Test that timing states remain unchanged."""
        F_gw = jnp.array([[1.0, 0.1], [0.0, 0.9]])  # 2x2
        F_spin = jnp.array([[1.0, 0.2], [0.0, 0.8]])  # 2x2
        F_list = (F_gw, F_spin)

        x = jnp.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])  # 6 states

        x_pred = jax_kalman_filter.compute_predicted_state(
            F_list, x, gw_size=2, spin_size=2
        )

        # Timing states (last 2) should be unchanged
        assert jnp.allclose(x_pred[4:], x[4:])


class TestComputePredictedCovariance:
    """Tests for compute_predicted_covariance function."""

    def test_basic_covariance_prediction(self):
        """Test basic covariance prediction."""
        # Simple test with identity matrices
        P = jnp.eye(6)
        F_gw = jnp.eye(2)
        F_spin = jnp.eye(2)
        Q_gw = jnp.zeros((2, 2))
        Q_spin = jnp.zeros((2, 2))

        P_pred = jax_kalman_filter.compute_predicted_covariance(
            P, (F_gw, F_spin), (Q_gw, Q_spin), gw_size=2, spin_size=2
        )

        # With identity F and zero Q, should get identity P back
        assert jnp.allclose(P_pred, P)

    def test_process_noise_addition(self):
        """Test that process noise is added correctly."""
        P = jnp.zeros((6, 6))
        F_gw = jnp.eye(2)
        F_spin = jnp.eye(2)
        Q_gw = jnp.eye(2) * 0.1
        Q_spin = jnp.eye(2) * 0.2

        P_pred = jax_kalman_filter.compute_predicted_covariance(
            P, (F_gw, F_spin), (Q_gw, Q_spin), gw_size=2, spin_size=2
        )

        # GW block should have Q_gw
        assert jnp.allclose(P_pred[:2, :2], Q_gw)
        # Spin block should have Q_spin
        assert jnp.allclose(P_pred[2:4, 2:4], Q_spin)


class TestLogLikelihood:
    """Tests for _log_likelihood function."""

    def test_basic_likelihood(self):
        """Test basic log likelihood calculation."""
        y = jnp.array([[0.1], [0.2]])
        cov = jnp.eye(2)

        ll = jax_kalman_filter._log_likelihood(y, cov)

        # Should return a (1, 1) array, not a scalar
        assert ll.shape == (1, 1)
        assert jnp.isfinite(ll)
        # For non-zero innovation, likelihood should be negative
        assert ll[0, 0] < 0

    def test_zero_innovation(self):
        """Test likelihood with zero innovation."""
        y = jnp.zeros((2, 1))
        cov = jnp.eye(2)

        ll = jax_kalman_filter._log_likelihood(y, cov)

        # Log likelihood of zero innovation should be relatively high
        assert ll > -10


class TestUpdate:
    """Tests for _update function."""

    def test_basic_update(self):
        """Test basic Kalman filter update step."""
        xp = jnp.zeros((4, 1))
        Pp = jnp.eye(4)
        H = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        R = jnp.eye(2) * 0.01
        z = jnp.array([0.1, 0.2])

        x, P, y, S = jax_kalman_filter._update(xp, Pp, H, R, z)

        # Updated state should be closer to measurement
        assert not jnp.allclose(x, xp)
        # Innovation should be non-zero
        assert not jnp.allclose(y, 0.0)
        # Updated covariance should be smaller
        assert jnp.trace(P) < jnp.trace(Pp)

    def test_joseph_form_symmetry(self):
        """Test that Joseph form maintains symmetry."""
        xp = jnp.ones((3, 1))
        Pp = jnp.eye(3)
        H = jnp.array([[1.0, 0.5, 0.0]])
        R = jnp.array([[0.01]])
        z = jnp.array([1.5])

        x, P, y, S = jax_kalman_filter._update(xp, Pp, H, R, z)

        # P should remain symmetric
        assert jnp.allclose(P, P.T)


class TestJaxKalmanFilterInitialization:
    """Tests for JaxKalmanFilter initialization."""

    @patch("argus.io_manager.get_argus_logger")
    def test_basic_initialization(self, mock_logger, sample_pulsar_data):
        """Test basic Kalman filter initialization."""
        mock_logger.return_value = Mock()

        kf = jax_kalman_filter.JaxKalmanFilter(data=sample_pulsar_data, use_gw=True)

        assert kf.Npsr == 2
        assert kf.use_gw is True
        assert kf.nx > 0

    @patch("argus.io_manager.get_argus_logger")
    def test_no_gw_initialization(self, mock_logger, sample_pulsar_data):
        """Test initialization without GW."""
        mock_logger.return_value = Mock()

        kf = jax_kalman_filter.JaxKalmanFilter(data=sample_pulsar_data, use_gw=False)

        assert kf.use_gw is False

    @patch("argus.io_manager.get_argus_logger")
    def test_jax_array_conversion(self, mock_logger, sample_pulsar_data):
        """Test that numpy arrays are converted to JAX arrays."""
        mock_logger.return_value = Mock()

        kf = jax_kalman_filter.JaxKalmanFilter(data=sample_pulsar_data, use_gw=True)

        # Check JAX array types
        assert isinstance(kf.jax_data, jnp.ndarray)
        assert isinstance(kf.jax_data_errors, jnp.ndarray)
        assert isinstance(kf.jax_t_diffs, jnp.ndarray)

    @patch("argus.io_manager.get_argus_logger")
    def test_float64_precision(self, mock_logger, sample_pulsar_data):
        """Test that arrays are 64-bit precision."""
        mock_logger.return_value = Mock()

        kf = jax_kalman_filter.JaxKalmanFilter(data=sample_pulsar_data, use_gw=True)

        # All arrays should be float64
        assert kf.jax_data.dtype == jnp.float64
        assert kf.jax_data_errors.dtype == jnp.float64
        assert kf.jax_H_matrices.dtype == jnp.float64


class TestGetLikelihood:
    """Tests for get_likelihood method."""

    @patch("argus.io_manager.get_argus_logger")
    def test_likelihood_computation(
        self, mock_logger, sample_pulsar_data, sample_noise_parameters
    ):
        """Test basic likelihood computation."""
        mock_logger.return_value = Mock()

        kf = jax_kalman_filter.JaxKalmanFilter(data=sample_pulsar_data, use_gw=True)

        # Create test parameters
        params = bayesian_inference.Parameters(
            log10_gamma_a=-9.0,
            γa=1e-9,
            ha=1e-15,
            γp=sample_noise_parameters["gamma_p"],
            σp=sample_noise_parameters["sigma_p"],
            EFAC=sample_noise_parameters["efac"],
            EQUAD=sample_noise_parameters["equad"],
        )

        ll = kf.get_likelihood(params)

        # Likelihood should be a scalar
        assert ll.shape == ()
        # Should be finite
        assert jnp.isfinite(ll)


class TestPrecomputeTransitionMatrices:
    """Tests for _precompute_transition_matrices function."""

    def test_matrix_precomputation(self):
        """Test precomputation of F and Q matrices."""
        γa = 1e-9
        γp = jnp.array([1e-8, 2e-8])
        σa2 = jnp.eye(2) * 1e-30
        σp2 = jnp.array([1e-30, 2e-30])
        dt_array = jnp.array([1.0, 2.0, 3.0])
        Npsr = 2
        M_sum = 10

        F_matrices, Q_matrices = jax_kalman_filter._precompute_transition_matrices(
            γa, γp, σa2, σp2, dt_array, Npsr, M_sum
        )

        F_gw_all, F_spin_all = F_matrices
        Q_gw_all, Q_spin_all = Q_matrices

        # Should have matrices for each time step
        assert F_gw_all.shape[0] == 3
        assert F_spin_all.shape[0] == 3
        assert Q_gw_all.shape[0] == 3
        assert Q_spin_all.shape[0] == 3


class TestInitializeKalmanFilter:
    """Tests for _initialize_kalman_filter function."""

    def test_initialization_shapes(self):
        """Test that initialization produces correct shapes."""
        nx = 20
        Npsr = 2
        P_eps = jnp.eye(12)  # Timing parameters covariance
        σa2 = jnp.eye(2) * 1e-30
        γa = 1e-9
        σp2 = jnp.array([1e-30, 2e-30])
        γp = jnp.array([1e-8, 2e-8])

        x0, P0 = jax_kalman_filter._initialize_kalman_filter(
            nx, Npsr, P_eps, σa2, γa, σp2, γp
        )

        # Check shapes
        assert x0.shape == (nx, 1)
        assert P0.shape == (nx, nx)

    def test_initialization_values(self):
        """Test initial values are reasonable."""
        nx = 10
        Npsr = 1
        P_eps = jnp.eye(6)
        σa2 = jnp.array([[1e-30]])
        γa = 1e-9
        σp2 = jnp.array([1e-30])
        γp = jnp.array([1e-8])

        x0, P0 = jax_kalman_filter._initialize_kalman_filter(
            nx, Npsr, P_eps, σa2, γa, σp2, γp
        )

        # Initial state should be zero
        assert jnp.allclose(x0, 0.0)

        # Initial covariance should be positive definite
        eigenvalues = jnp.linalg.eigvalsh(P0)
        assert jnp.all(eigenvalues >= 0)

    def test_gw_block_structure(self):
        """Test GW block has correct structure."""
        nx = 10
        Npsr = 1
        P_eps = jnp.eye(6)
        σa2 = jnp.array([[1e-30]])
        γa = 1e-9
        σp2 = jnp.array([1e-30])
        γp = jnp.array([1e-8])

        x0, P0 = jax_kalman_filter._initialize_kalman_filter(
            nx, Npsr, P_eps, σa2, γa, σp2, γp
        )

        # GW 'r' states should have very small variance
        assert P0[0, 0] < 1e-30

        # GW 'a' states should have variance proportional to σa2 / (2*γa)
        expected_var_a = σa2[0, 0] / (2.0 * γa)
        assert jnp.isclose(P0[1, 1], expected_var_a, rtol=0.1)


def _reference_sequential_filter(m0, P0, Fs, Qs, Hs, Rs, zs):
    """Independent textbook linear-Gaussian Kalman filter for tests.

    Matches the Argus epoch convention: epoch 0 is update-only against the prior
    (m0, P0); epochs 1..T-1 predict with (Fs[k-1], Qs[k-1]) then update. Uses the
    standard-form covariance update (I - K H) P to match the associative element
    covariance C. Returns stacked filtered means (T, d, 1) and covariances (T, d, d).
    """
    d = m0.shape[0]
    I = jnp.eye(d)

    def update(mp, Pp, H, R, z):
        S = H @ Pp @ H.T + R
        K = Pp @ H.T @ jnp.linalg.solve(S, jnp.eye(S.shape[0]))
        m = mp + K @ (z[:, None] - H @ mp)
        P = (I - K @ H) @ Pp
        return m, 0.5 * (P + P.T)

    m, P = update(m0, P0, Hs[0], Rs[0], zs[0])
    means, covs = [m], [P]
    for k in range(1, len(zs)):
        mp = Fs[k - 1] @ m
        Pp = Fs[k - 1] @ P @ Fs[k - 1].T + Qs[k - 1]
        m, P = update(mp, Pp, Hs[k], Rs[k], zs[k])
        means.append(m)
        covs.append(P)
    return jnp.stack(means), jnp.stack(covs)


class TestAssociativeScanFilter:
    """Tests for the associative-scan (temporally parallel) filter primitives."""

    def _random_system(self, seed=0, d=3, m=2, T=6):
        """Build a small well-conditioned linear-Gaussian system for testing."""
        rng = np.random.default_rng(seed)
        m0 = jnp.array(rng.standard_normal((d, 1)))
        A = rng.standard_normal((d, d))
        P0 = jnp.array(A @ A.T + d * np.eye(d))  # SPD prior
        Fs = jnp.array(
            [0.9 * np.eye(d) + 0.05 * rng.standard_normal((d, d)) for _ in range(T - 1)]
        )
        Qs = jnp.array(
            [(lambda B: B @ B.T + 0.1 * np.eye(d))(rng.standard_normal((d, d))) for _ in range(T - 1)]
        )
        Hs = jnp.array([rng.standard_normal((m, d)) for _ in range(T)])
        Rs = jnp.array(
            [(lambda B: B @ B.T + 0.5 * np.eye(m))(rng.standard_normal((m, m))) for _ in range(T)]
        )
        zs = jnp.array(rng.standard_normal((T, m)))
        return m0, P0, Fs, Qs, Hs, Rs, zs

    def _build_elements(self, m0, P0, Fs, Qs, Hs, Rs, zs):
        import jax

        first = jax_kalman_filter._first_filtering_element(m0, P0, Hs[0], Rs[0], zs[0])
        generic = jax.vmap(jax_kalman_filter._generic_filtering_element)(
            Fs, Qs, Hs[1:], Rs[1:], zs[1:]
        )
        return jax.tree_util.tree_map(
            lambda f, g: jnp.concatenate([f[None], g], axis=0), first, generic
        )

    def test_first_element_zero_blocks(self):
        """First (epoch-0) element has A = eta = J = 0 and correct shapes."""
        m0, P0, Fs, Qs, Hs, Rs, zs = self._random_system()
        d = m0.shape[0]
        A, b, C, eta, J = jax_kalman_filter._first_filtering_element(
            m0, P0, Hs[0], Rs[0], zs[0]
        )
        assert A.shape == (d, d) and jnp.allclose(A, 0.0)
        assert eta.shape == (d, 1) and jnp.allclose(eta, 0.0)
        assert J.shape == (d, d) and jnp.allclose(J, 0.0)
        assert b.shape == (d, 1)
        assert C.shape == (d, d)
        assert jnp.allclose(C, C.T)  # symmetric

    def test_generic_element_shapes(self):
        """Generic element returns (A, b, C, eta, J) with the right shapes."""
        m0, P0, Fs, Qs, Hs, Rs, zs = self._random_system()
        d = m0.shape[0]
        A, b, C, eta, J = jax_kalman_filter._generic_filtering_element(
            Fs[0], Qs[0], Hs[1], Rs[1], zs[1]
        )
        assert A.shape == (d, d)
        assert b.shape == (d, 1)
        assert C.shape == (d, d) and jnp.allclose(C, C.T)
        assert eta.shape == (d, 1)
        assert J.shape == (d, d) and jnp.allclose(J, J.T)

    def test_operator_associativity(self):
        """(e1 ⊗ e2) ⊗ e3 == e1 ⊗ (e2 ⊗ e3) on valid elements."""
        m0, P0, Fs, Qs, Hs, Rs, zs = self._random_system(seed=1)
        e1 = jax_kalman_filter._generic_filtering_element(Fs[0], Qs[0], Hs[1], Rs[1], zs[1])
        e2 = jax_kalman_filter._generic_filtering_element(Fs[1], Qs[1], Hs[2], Rs[2], zs[2])
        e3 = jax_kalman_filter._generic_filtering_element(Fs[2], Qs[2], Hs[3], Rs[3], zs[3])

        def batch(e):
            return tuple(x[None] for x in e)

        def unbatch(e):
            return tuple(x[0] for x in e)

        left = unbatch(
            jax_kalman_filter._filtering_operator(
                jax_kalman_filter._filtering_operator(batch(e1), batch(e2)), batch(e3)
            )
        )
        right = unbatch(
            jax_kalman_filter._filtering_operator(
                batch(e1), jax_kalman_filter._filtering_operator(batch(e2), batch(e3))
            )
        )
        for a, b in zip(left, right):
            assert jnp.allclose(a, b, atol=1e-8, rtol=1e-6)

    def test_matches_sequential_small_system(self):
        """associative_scan of elements reproduces the sequential filtered states."""
        import jax

        m0, P0, Fs, Qs, Hs, Rs, zs = self._random_system(seed=2)
        elements = self._build_elements(m0, P0, Fs, Qs, Hs, Rs, zs)
        _, b_filt, C_filt, _, _ = jax.lax.associative_scan(
            jax_kalman_filter._filtering_operator, elements
        )

        m_ref, P_ref = _reference_sequential_filter(m0, P0, Fs, Qs, Hs, Rs, zs)

        assert jnp.allclose(b_filt, m_ref, atol=1e-8, rtol=1e-6)
        assert jnp.allclose(C_filt, P_ref, atol=1e-8, rtol=1e-6)
