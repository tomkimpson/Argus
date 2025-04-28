import pytest
import numpy as np
import jax
import jax.numpy as jnp


# --- JAX Configuration for CPU ---
# Force JAX to use the CPU platform only.
# This is useful for testing environments without GPUs/TPUs
# or to ensure consistent testing behavior.
print("Configuring JAX to use CPU...") # Optional: Add a print statement for confirmation
jax.config.update("jax_platforms", "cpu")
# ---------------------------------
# Enable 64-bit precision in JAX for numerical stability
jax.config.update("jax_enable_x64", True)


# from collections import namedtuple # No longer needed for params
from unittest.mock import ANY
from flax import struct # Import flax struct

# Assume the file is named jax_kalman_filter.py
from argus import jax_kalman_filter as jk



# --- Define the Flax Dataclass for Parameters ---
@struct.dataclass
class Parameters:
    """Define a struct to store the parameters of the Kalman filter model"""

    #GW parameters
    γa: float  # s⁻¹
    ha: float  # GWB amplitude

    #Pulsar parameters for the OU process
    γp: jnp.ndarray  # Pulsar-specific gamma values
    σp: jnp.ndarray  # Pulsar-specific sigma values

    #Measurement noise parameters
    EFAC: jnp.ndarray  # Error factors
    EQUAD: jnp.ndarray # Extra quadrature noise


# --- Fixtures for Test Setup ---

# Fixture for dimensions used in class tests (remains the same)
@pytest.fixture(scope="module")
def class_test_dims():
    """Dimensions for the full class tests, matching original assumptions."""
    Npsr = 2
    obs_dim = 32 # Assumed observation dimension from R/H handling in _update
    P_eps_dim = Npsr # Simplest epsilon state assumption
    n_states = Npsr * 2 + Npsr * 2 + P_eps_dim # GW+Spin+Eps = 4+4+2 = 10
    return {
        'Npsr': Npsr,
        'obs_dim': obs_dim,
        'P_eps_dim': P_eps_dim,
        'n_states': n_states,
        'N_timesteps': 10
    }

# Updated fixture for parameters using the Flax Dataclass
@pytest.fixture(scope="module")
def mock_flax_params(class_test_dims):
    """Provides an instance of the Parameters flax dataclass."""
    Npsr = class_test_dims['Npsr'] # Get Npsr from the dims fixture
    # Ensure array parameters are float64 for consistency
    dtype = jnp.float64
    return Parameters(
        γa = 1e-9,
        ha = 1e-15,
        # Ensure pulsar-specific arrays have the correct shape (Npsr,)
        γp = jnp.full(Npsr, 1e-10, dtype=dtype),
        σp = jnp.full(Npsr, 1e-11, dtype=dtype),
        EFAC = jnp.ones(Npsr, dtype=dtype),
        EQUAD = jnp.zeros(Npsr, dtype=dtype)
    )

# Other fixtures remain largely the same
@pytest.fixture(scope="module")
def basic_dims():
    """Basic dimensions for simple tests."""
    # Keep Npsr=1 here consistent with potential simple internal tests
    return {'Npsr': 1, 'obs_dim': 2, 'P_eps_dim': 1}

@pytest.fixture
def mock_model_instance(class_test_dims, mocker):
    """Provides a mocked model object for JaxKalmanFilter class tests."""
    dims = class_test_dims
    mock_model = mocker.MagicMock()
    mock_model.Npsr = dims['Npsr']
    mock_model.M_sum = 1 # Example
    mock_model.nx = dims['n_states']
    # Mock precompute_H_matrix to return something with the right shape
    mock_H_matrices_np = np.random.randn(dims['N_timesteps'], dims['obs_dim'], dims['n_states']).astype(np.float64)
    # ** Store H on the mock **
    mock_model.H_matrices = jnp.array(mock_H_matrices_np) # Store as jax array
    mock_model.precompute_H_matrix.return_value = mock_H_matrices_np # Original function returns numpy

    # Mock HD matrix
    mock_model.hd_correlation_matrix = np.eye(dims['Npsr']).astype(np.float64)
    return mock_model

@pytest.fixture
def numpy_setup_data(class_test_dims):
    """Provides numpy arrays for initial filter setup."""
    dims = class_test_dims
    # Mock Observations (NumPy arrays initially)
    mock_toa_np = np.linspace(0, (dims['N_timesteps'] - 1) * 86400, dims['N_timesteps']).astype(np.float64)
    mock_data_np = np.random.randn(dims['N_timesteps'], dims['obs_dim']).astype(np.float64) * 1e-7
    mock_data_errors_np = np.ones((dims['Npsr'], dims['N_timesteps'])).astype(np.float64) * 1e-7
    observations_np = [mock_toa_np, mock_data_np, mock_data_errors_np]

    # Mock Initial State/Covariance (NumPy)
    mock_x0_np = np.zeros(dims['n_states']).astype(np.float64)
    mock_P0_np = np.eye(dims['n_states']).astype(np.float64)
    mock_Peps_np = np.eye(dims['P_eps_dim']).astype(np.float64) * 1e-28

    return {
        "observations": observations_np,
        "x0": mock_x0_np,
        "P0": mock_P0_np,
        "Peps": mock_Peps_np,
        "toa": mock_toa_np,
        "data": mock_data_np,
        "data_errors": mock_data_errors_np
    }



# --- Test for the Main Scan Function ---
def test_run_kalman_filter_scan(mocker, class_test_dims, numpy_setup_data, mock_flax_params):
    """Test the main Kalman filter loop (_run_kalman_filter_scan)."""
    # Setup parameters and dimensions from fixtures
    dims = class_test_dims
    Npsr = dims['Npsr']
    M_sum = 10*Npsr # Example
    n_states = dims['n_states']
    obs_dim = dims['obs_dim']
    N_timesteps = dims['N_timesteps']
    dim_x_predict = 2 * Npsr

    # Use the flax dataclass instance from the fixture
    params = mock_flax_params

    # Get data from fixture and convert to JAX arrays
    setup_data = numpy_setup_data
    mock_data = jnp.array(setup_data['data'])
    mock_data_errors = jnp.array(setup_data['data_errors'])
    mock_dt_array = jnp.diff(jnp.array(setup_data['toa']))
    mock_P_eps = jnp.array(setup_data['Peps'])

    # Mock H matrices: shape (N_timesteps, obs_dim, n_states)
    mock_H_matrices = jnp.array(np.random.randn(N_timesteps, obs_dim, n_states), dtype=jnp.float64)

    # Mock Hellings-Downs Matrix
    mock_hd_matrix = jnp.eye(Npsr, dtype=jnp.float64) # Simple case

    # --- Mock Return Values---
    mock_x0 = jnp.zeros((n_states, 1), dtype=jnp.float64)
    mock_P0 = jnp.eye(n_states, dtype=jnp.float64)
    mock_init = mocker.patch('argus.jax_kalman_filter._initialize_kalman_filter', return_value=(mock_x0, mock_P0))

    mock_R_matrices = jnp.array([jnp.eye(obs_dim, dtype=jnp.float64) * 1e-14] * N_timesteps)
    mock_precompute_R = mocker.patch('argus.jax_kalman_filter.precompute_R_matrices', return_value=mock_R_matrices)

    mock_F_gw = jnp.eye(2 * Npsr, dtype=jnp.float64) * 0.99
    mock_F_spin = jnp.eye(2 * Npsr, dtype=jnp.float64) * 0.98
    mock_F_matrices = mocker.patch('argus.jax_kalman_filter.F_matrices_non_precomputed', return_value=(mock_F_gw, mock_F_spin))

    mock_Q_gw = jnp.eye(2 * Npsr, dtype=jnp.float64) * 1e-20
    mock_Q_spin = jnp.eye(2 * Npsr, dtype=jnp.float64) * 1e-22
    mock_Q_matrices = mocker.patch('argus.jax_kalman_filter.Q_matrices_non_precomputed', return_value=(mock_Q_gw, mock_Q_spin))
    # --- End Mock Return Values ---

    # Run the function
    log_likelihood = jk._run_kalman_filter_scan(
        θ=params, # Pass the flax dataclass instance
        data=mock_data,
        data_errors=mock_data_errors,
        H_matrices=mock_H_matrices,
        Npsr=Npsr,
        M_sum=M_sum,
        hellings_downs_matrix=mock_hd_matrix,
        dt_array=mock_dt_array,
        dim_x=dim_x_predict,
        n_states=n_states,
        P_eps=mock_P_eps
    )

    # Check that the log likelihood is a jax scalar
    assert isinstance(log_likelihood, jax.Array)
    assert log_likelihood.shape == ()

    # # Check mocks were called (assertions remain the same, checking values from the params object)
    # mock_init.assert_called_once_with(n_states, Npsr, mock_P_eps, params.ha**2, params.γa)
    # # Check precompute_R call includes EFAC/EQUAD from params
    # mock_precompute_R.assert_called_once()
    # # Example of checking args more specifically if needed (requires knowing the exact call structure)
    # precompute_R_call_args, _ = mock_precompute_R.call_args
    # np.testing.assert_allclose(precompute_R_call_args[0], mock_data_errors) # Assuming errors is the first arg
    # np.testing.assert_allclose(precompute_R_call_args[1], params.EFAC)     # Assuming EFAC is the second
    # np.testing.assert_allclose(precompute_R_call_args[2], params.EQUAD)    # Assuming EQUAD is the third


    # assert mock_F_matrices.call_count == N_timesteps - 1
    # assert mock_Q_matrices.call_count == N_timesteps - 1
    # # Check specific call args using values from the flax params object
    # mock_F_matrices.assert_any_call(params.γa, params.γp, mock_dt_array[0], Npsr, M_sum)
    # mock_Q_matrices.assert_any_call(params.γa, ANY, params.γp, params.σp**2, mock_dt_array[0])

# --- Tests for the JaxKalmanFilter Class (Update to use mock_flax_params) ---

# class TestJaxKalmanFilterClass:

#     # test_init_and_prepare_arrays remains the same as it doesn't directly use parameters

#     # test_prepare_arrays_raises_error_on_wrong_dtype remains the same

#     # Update test_get_likelihood_calls_scan to use the flax parameter fixture
#     def test_get_likelihood_calls_scan(self, mocker, mock_model_instance, numpy_setup_data, mock_flax_params): # Use new fixture
#         """Test that get_likelihood calls _run_kalman_filter_scan correctly."""
#         setup_data = numpy_setup_data
#         model = mock_model_instance
#         params = mock_flax_params # Use the flax dataclass instance

#         # Setup the filter instance
#         kf = jk.JaxKalmanFilter(
#             model=model,
#             observations=setup_data['observations'],
#             x0=setup_data['x0'],
#             P0=setup_data['P0'],
#             Peps=setup_data['Peps']
#         )

#         # Mock the target function _run_kalman_filter_scan
#         mock_ll_value = jnp.array(-123.456)
#         mock_run_scan = mocker.patch('jax_kalman_filter._run_kalman_filter_scan', return_value=mock_ll_value)

#         # Call get_likelihood with the flax dataclass instance
#         result_ll = kf.get_likelihood(params)

#         # Check the return value
#         assert result_ll == mock_ll_value

#         # Check that _run_kalman_filter_scan was called once with the correct arguments
#         mock_run_scan.assert_called_once()
#         call_args, call_kwargs = mock_run_scan.call_args

#         # Check keyword arguments passed to the scan function
#         # The comparison should work for flax dataclasses
#         assert call_kwargs['θ'] == params
#         np.testing.assert_allclose(call_kwargs['data'], kf.jax_data)
#         np.testing.assert_allclose(call_kwargs['data_errors'], kf.jax_data_errors)
#         np.testing.assert_allclose(call_kwargs['H_matrices'], kf.jax_H_matrices)
#         assert call_kwargs['Npsr'] == model.Npsr
#         assert call_kwargs['M_sum'] == model.M_sum
#         np.testing.assert_allclose(call_kwargs['hellings_downs_matrix'], kf.hellings_downs_matrix)
#         np.testing.assert_allclose(call_kwargs['dt_array'], kf.jax_t_diffs)
#         assert call_kwargs['dim_x'] == 2 * model.Npsr
#         assert call_kwargs['n_states'] == model.nx
#         np.testing.assert_allclose(call_kwargs['P_eps'], kf.P_eps)
#         assert call_kwargs['P_eps'].dtype == jnp.float64

# # --- Tests for Internal Functions (Copied from previous version, unchanged) ---
# # Added here for completeness if running as a single file

# class TestJaxKalmanFilterInternals:

#     @pytest.mark.parametrize("y, cov, expected_ll_approx", [
#         (jnp.array([2.0]), jnp.array([[4.0]]), -2.112), # Scalar case
#         (jnp.array([1.0, -1.0]), jnp.array([[2.0, 0.5], [0.5, 1.0]]), -3.260), # Vector case
#     ])
#     def test_log_likelihood(self, y, cov, expected_ll_approx):
#         """Test the _log_likelihood calculation."""
#         sign, logdet = jnp.linalg.slogdet(2.0 * jnp.pi * cov)
#         inv_cov = jnp.linalg.inv(cov)
#         quad_term = y.T @ inv_cov @ y
#         expected_ll = -0.5 * (logdet + quad_term)
#         ll = jk._log_likelihood(y, cov)
#         np.testing.assert_allclose(ll, expected_ll, rtol=1e-5)
#         np.testing.assert_allclose(ll, expected_ll_approx, rtol=1e-3)

#     def test_predict(self, mocker):
#         """Test the _predict step, mocking external compute functions."""
#         dim_x = 4
#         x = jnp.ones((dim_x, 1), dtype=jnp.float64)
#         P = jnp.eye(dim_x, dtype=jnp.float64) * 2.0
#         F_list = (jnp.eye(dim_x, dtype=jnp.float64) * 1.1,)
#         Q_list = (jnp.eye(dim_x, dtype=jnp.float64) * 0.1,)
#         mock_xp = jnp.ones((dim_x, 1), dtype=jnp.float64) * 1.1
#         mock_Pp = (jnp.eye(dim_x) * 1.1) @ P @ (jnp.eye(dim_x) * 1.1).T + (jnp.eye(dim_x) * 0.1)
#         mock_compute_pred_state = mocker.patch('jax_kalman_filter.compute_predicted_state', return_value=mock_xp)
#         mock_compute_pred_cov = mocker.patch('jax_kalman_filter.compute_predicted_covariance', return_value=mock_Pp)
#         xp, Pp = jk._predict(x, P, F_list, Q_list, dim_x)
#         mock_compute_pred_state.assert_called_once_with(F_list, x, dim_x, dim_x)
#         mock_compute_pred_cov.assert_called_once_with(P, F_list, Q_list, dim_x, dim_x)
#         np.testing.assert_allclose(xp, mock_xp)
#         np.testing.assert_allclose(Pp, mock_Pp)

#     def test_update(self, basic_dims):
#         """Test the _update step."""
#         dim_x = 4
#         obs_dim = basic_dims['obs_dim'] # 2
#         xp = jnp.array([1.0, 0.1, 2.0, 0.2], dtype=jnp.float64).reshape(-1, 1)
#         Pp = jnp.diag(jnp.array([0.5, 0.1, 0.5, 0.1], dtype=jnp.float64))
#         H = jnp.zeros((obs_dim, dim_x), dtype=jnp.float64)
#         H = H.at[0, 0].set(1.0).at[1, 2].set(1.0)
#         R_small = jnp.eye(obs_dim, dtype=jnp.float64) * 0.2
#         z_small = jnp.array([1.1, 2.1], dtype=jnp.float64).reshape(obs_dim, 1)
#         y_expected = z_small - H @ xp
#         S_expected_calc = H @ Pp @ H.T + R_small
#         Sinv_expected = jnp.linalg.inv(S_expected_calc)
#         K_expected_calc = Pp @ H.T @ Sinv_expected
#         x_expected_calc = xp + K_expected_calc @ y_expected
#         I_KH = jnp.eye(dim_x) - K_expected_calc @ H
#         P_expected = I_KH @ Pp @ I_KH.T + K_expected_calc @ R_small @ K_expected_calc.T
#         target_obs_dim = 32
#         z_reshaped_for_func = jnp.zeros((target_obs_dim, 1), dtype=jnp.float64).at[0:obs_dim, 0].set(z_small.flatten())
#         H_reshaped_for_func = jnp.zeros((target_obs_dim, dim_x), dtype=jnp.float64).at[0:obs_dim, :].set(H)
#         R_reshaped_for_func = jnp.zeros((target_obs_dim, target_obs_dim), dtype=jnp.float64).at[0:obs_dim, 0:obs_dim].set(R_small)
#         x, P, y, S = jk._update(xp, Pp, H_reshaped_for_func, R_reshaped_for_func, z_reshaped_for_func)
#         np.testing.assert_allclose(x, x_expected_calc, rtol=1e-6)
#         np.testing.assert_allclose(P, P_expected, rtol=1e-6, atol=1e-9)
#         y_relevant = y[:obs_dim, 0]
#         S_relevant = S[:obs_dim, :obs_dim]
#         np.testing.assert_allclose(y_relevant, y_expected.flatten(), rtol=1e-6)
#         np.testing.assert_allclose(S_relevant, S_expected_calc, rtol=1e-6)

#     def test_initialize_kalman_filter(self):
#         """Test the initialization of state and covariance."""
#         Npsr = 2
#         P_eps_dim = Npsr
#         nx = Npsr * 2 + Npsr * 2 + P_eps_dim
#         P_eps = jnp.eye(P_eps_dim, dtype=jnp.float64) * 1e-28
#         h2 = 1e-29
#         gamma_a = 1e-9
#         x0, P0 = jk._initialize_kalman_filter(nx, Npsr, P_eps, h2, gamma_a)
#         assert x0.shape == (nx, 1)
#         np.testing.assert_allclose(x0, jnp.zeros((nx, 1)))
#         assert P0.shape == (nx, nx)
#         P_GW_expected = jnp.eye(Npsr * 2, dtype=jnp.float64)
#         P_GW_expected = P_GW_expected.at[0::2, 0::2].multiply(1e-40)
#         sigma2_expected = (h2 / 12) * gamma_a
#         P_GW_expected = P_GW_expected.at[1::2, 1::2].multiply(sigma2_expected / (2 * gamma_a))
#         np.testing.assert_allclose(P0[:Npsr*2, :Npsr*2], P_GW_expected, rtol=1e-6)
#         P_spin_expected = jnp.eye(Npsr * 2, dtype=jnp.float64)
#         P_spin_expected = P_spin_expected.at[0::2, 0::2].multiply(1e-40)
#         P_spin_expected = P_spin_expected.at[1::2, 1::2].multiply(1e-20)
#         np.testing.assert_allclose(P0[Npsr*2:Npsr*4, Npsr*2:Npsr*4], P_spin_expected, rtol=1e-6)
#         np.testing.assert_allclose(P0[Npsr*4:, Npsr*4:], P_eps, rtol=1e-6)
#         assert jnp.sum(jnp.abs(P0[:Npsr*4, Npsr*4:])) == 0
#         assert jnp.sum(jnp.abs(P0[Npsr*4:, :Npsr*4])) == 0

#     # Copied TestJaxKalmanFilterClass tests that were unchanged
#     def test_init_and_prepare_arrays(self, mock_model_instance, numpy_setup_data, class_test_dims):
#         """Test initialization and _prepare_jax_arrays."""
#         setup_data = numpy_setup_data
#         dims = class_test_dims
#         model = mock_model_instance
#         kf = jk.JaxKalmanFilter(
#             model=model,
#             observations=setup_data['observations'],
#             x0=setup_data['x0'],
#             P0=setup_data['P0'],
#             Peps=setup_data['Peps']
#         )
#         assert kf.model == model
#         assert kf.N_timesteps == dims['N_timesteps']
#         np.testing.assert_array_equal(kf.toa, setup_data['toa'])
#         np.testing.assert_array_equal(kf.data, setup_data['data'])
#         np.testing.assert_array_equal(kf.data_errors, setup_data['data_errors'])
#         np.testing.assert_array_equal(kf.x0, setup_data['x0'])
#         np.testing.assert_array_equal(kf.P0, setup_data['P0'])
#         np.testing.assert_array_equal(kf.P_eps, setup_data['Peps'])
#         model.precompute_H_matrix.assert_called_once()
#         np.testing.assert_array_equal(kf.Hmat, model.precompute_H_matrix.return_value)
#         assert isinstance(kf.jax_data, jax.Array)
#         assert kf.jax_data.dtype == jnp.float64
#         np.testing.assert_allclose(kf.jax_data, setup_data['data'])
#         assert isinstance(kf.jax_data_errors, jax.Array)
#         assert kf.jax_data_errors.dtype == jnp.float64
#         np.testing.assert_allclose(kf.jax_data_errors, setup_data['data_errors'])
#         assert isinstance(kf.jax_t_diffs, jax.Array)
#         assert kf.jax_t_diffs.dtype == jnp.float64
#         np.testing.assert_allclose(kf.jax_t_diffs, np.diff(setup_data['toa']))
#         assert isinstance(kf.jax_x0, jax.Array)
#         assert kf.jax_x0.dtype == jnp.float64
#         assert kf.jax_x0.shape == (dims['n_states'], 1)
#         np.testing.assert_allclose(kf.jax_x0, setup_data['x0'].reshape(-1, 1))
#         assert isinstance(kf.jax_P0, jax.Array)
#         assert kf.jax_P0.dtype == jnp.float64
#         np.testing.assert_allclose(kf.jax_P0, setup_data['P0'])
#         assert isinstance(kf.jax_H_matrices, jax.Array)
#         assert kf.jax_H_matrices.dtype == jnp.float64
#         np.testing.assert_allclose(kf.jax_H_matrices, model.precompute_H_matrix.return_value)
#         assert isinstance(kf.hellings_downs_matrix, jax.Array)
#         assert kf.hellings_downs_matrix.dtype == jnp.float64
#         np.testing.assert_allclose(kf.hellings_downs_matrix, model.hd_correlation_matrix)

#     def test_prepare_arrays_raises_error_on_wrong_dtype(self, mock_model_instance, numpy_setup_data):
#         """Test that _prepare_jax_arrays raises ValueError for non-float64."""
#         setup_data = numpy_setup_data
#         bad_obs = [
#             setup_data['toa'].astype(np.float32),
#             setup_data['data'].astype(np.float32),
#             setup_data['data_errors'].astype(np.float32)
#         ]
#         with pytest.raises(ValueError, match="expected float64"):
#             jk.JaxKalmanFilter(
#                 model=mock_model_instance,
#                 observations=bad_obs,
#                 x0=setup_data['x0'],
#                 P0=setup_data['P0'],
#                 Peps=setup_data['Peps']
#             )




# class TestNumericalStability:

#     def test_cholesky_stability(self, mock_model_instance, numpy_setup_data):
#         """Test that the Cholesky decomposition is stable."""
#         setup_data = numpy_setup_data
#         model = mock_model_instance
        


# utils.check_cholesky(P_spin, "P_spin")
    # utils.check_min_eigenvalue(P_spin, "P_spin")
    # utils.check_symmetry(P_spin, "P_spin")
    # utils.check_condition_number(P_spin, "P_spin")

    #jax.debug.print('P_spin is {P_spin}', P_spin=P_spin,ordered=True)








    # utils.check_cholesky(P, "Pupdated")
    # utils.check_min_eigenvalue(P, "Pupdated")
    # utils.check_symmetry(P, "Pupdated")
    # utils.check_condition_number(P, "Pupdated")





    # check_cholesky(P0,"The initial P-matrix")
    # check_min_eigenvalue(P0, "The initial P-matrix")
    # check_symmetry(P0, "The initial P-matrix")
    # check_condition_number(P0, "The initial P-matrix")
