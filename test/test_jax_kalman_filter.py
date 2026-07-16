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


def _realistic_pulsar_data(seed=0):
    """Build a small, well-conditioned pulsar-data dict for equivalence testing.

    Mirrors the structure the real data loader produces: unit-scaled design matrices
    and a GLS timing-model prior ``P_eps = (Mᵀ N⁻¹ M)⁻¹`` consistent with the design and
    the TOA errors. This is the regime the filter actually runs in. (The generic
    `sample_pulsar_data` fixture uses an arbitrary O(1) timing prior with 1e-6-scale
    residuals — a physically inconsistent configuration whose extreme dynamic range makes
    it a poor equivalence test, even though both backends are individually well-defined.)
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    n_epochs, n_psr = 12, 2
    dims = [5, 6]
    err_scale = 1e-6  # ~microsecond TOA errors

    metadata = pd.DataFrame(
        {
            "name": ["A", "B"],
            "dim_M": dims,
            "RA": [0.5, 1.2],
            "DEC": [0.3, -0.1],
            "F0": [200.0, 150.0],
            "par_file": ["a", "b"],
            "tim_file": ["a", "b"],
        }
    )
    residuals = rng.standard_normal((n_epochs, n_psr)) * err_scale
    errors = np.ones((n_epochs, n_psr)) * err_scale

    design_matrices, parameter_covariances = [], []
    for i in range(n_psr):
        M = rng.standard_normal((n_epochs, dims[i]))
        M = M / np.sqrt(np.sum(M**2, axis=0))  # unit-norm columns (as data_loader does)
        Ninv = np.diag(1.0 / errors[:, i] ** 2)
        design_matrices.append(M)
        parameter_covariances.append(np.linalg.inv(M.T @ Ninv @ M))  # GLS prior

    return {
        "processed_residuals": {
            "toas": np.linspace(0, 1000, n_epochs) * 86400,
            "residuals": residuals,
            "errors": errors,
        },
        "metadata": metadata,
        "design_matrices": design_matrices,
        "parameter_covariances": parameter_covariances,
        "hd_correlation": np.array([[1.0, 0.5], [0.5, 1.0]]),
    }


class TestMarginalFilter:
    """Tests for the marginalized (Rao-Blackwellized) timing-model filter.

    The marginal filter integrates the static timing-model parameters out of the
    propagated state analytically instead of augmenting the state with them. It is
    mathematically equivalent to the default sequential filter, so the two must return
    the same log likelihood; the marginal path just propagates a smaller state.
    """

    def _params(self):
        return bayesian_inference.Parameters(
            log10_gamma_a=-9.0,
            γa=1e-9,
            ha=1e-15,
            γp=jnp.full(2, 1e-8),
            σp=jnp.full(2, 1e-15),
            EFAC=jnp.ones(2),
            EQUAD=jnp.full(2, 1e-6),
        )

    @patch("argus.io_manager.get_argus_logger")
    def test_matches_sequential(self, mock_logger):
        """Marginal and sequential backends must agree on the log likelihood."""
        mock_logger.return_value = Mock()
        data = _realistic_pulsar_data()

        kf_seq = jax_kalman_filter.JaxKalmanFilter(
            data=data, use_gw=True, use_marginal=False
        )
        kf_marg = jax_kalman_filter.JaxKalmanFilter(
            data=data, use_gw=True, use_marginal=True
        )

        ll_seq = kf_seq.get_likelihood(self._params())
        ll_marg = kf_marg.get_likelihood(self._params())

        assert ll_marg.shape == ()
        assert jnp.isfinite(ll_marg)
        assert jnp.isclose(
            ll_marg, ll_seq, rtol=1e-6, atol=1e-4
        ), f"marginal {ll_marg} != sequential {ll_seq}"

    @patch("argus.io_manager.get_argus_logger")
    def test_null_gw_matches_sequential(self, mock_logger):
        """Equivalence must also hold with GW terms disabled (use_gw=False)."""
        mock_logger.return_value = Mock()
        data = _realistic_pulsar_data()

        kf_seq = jax_kalman_filter.JaxKalmanFilter(
            data=data, use_gw=False, use_marginal=False
        )
        kf_marg = jax_kalman_filter.JaxKalmanFilter(
            data=data, use_gw=False, use_marginal=True
        )

        ll_seq = kf_seq.get_likelihood(self._params())
        ll_marg = kf_marg.get_likelihood(self._params())

        assert jnp.isclose(
            ll_marg, ll_seq, rtol=1e-6, atol=1e-4
        ), f"marginal {ll_marg} != sequential {ll_seq}"

    @patch("argus.io_manager.get_argus_logger")
    def test_P_eps_inv_is_prior_inverse(self, mock_logger, sample_pulsar_data):
        """P_eps_inv must be the exact inverse of the augmented filter's prior block,
        with per-pulsar blocks in the same order as the epsilon columns of H."""
        mock_logger.return_value = Mock()

        kf = jax_kalman_filter.JaxKalmanFilter(
            data=sample_pulsar_data, use_gw=True, use_marginal=True
        )

        M_sum = kf.M_sum
        assert kf.P_eps_inv.shape == (M_sum, M_sum)
        # P_eps_inv @ P_eps == I  (validates block ordering + inverse relationship)
        identity = kf.P_eps_inv @ kf.P_eps
        assert jnp.allclose(identity, jnp.eye(M_sum), atol=1e-6)
