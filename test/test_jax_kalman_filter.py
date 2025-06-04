# import pytest
# import numpy as np
# import jax
# import jax.numpy as jnp


# # # --- JAX Configuration for CPU ---
# # Check for GPU availability and configure JAX accordingly
# try:
#     gpu_devices = jax.devices('gpu')
#     if gpu_devices:
#         print("GPU found. Configuring JAX to use GPU.")
#         jax.config.update("jax_platforms", "gpu")
#     else:
#         print("No GPU found. Configuring JAX to use CPU.")
#         jax.config.update("jax_platforms", "cpu")
# except Exception as e:
#     print(f"Error checking for GPU: {e}")
#     print("Falling back to CPU.")
#     jax.config.update("jax_platforms", "cpu")
# # ---------------------------------
# # Enable 64-bit precision in JAX for numerical stability
# jax.config.update("jax_enable_x64", True)
# from jax.scipy.linalg import block_diag


# # from collections import namedtuple # No longer needed for params
# from unittest.mock import ANY
# from flax import struct # Import flax struct

# # Assume the file is named jax_kalman_filter.py
# from argus import jax_kalman_filter as jk
# from argus import data_loader,gravitational_waves,model


# # --- Define the Flax Dataclass for Parameters ---
# @struct.dataclass
# class Parameters:
#     """Define a struct to store the parameters of the Kalman filter model"""

#     #GW parameters
#     γa: float  # s⁻¹
#     ha: float  # GWB amplitude

#     #Pulsar parameters for the OU process
#     γp: jnp.ndarray  # Pulsar-specific gamma values
#     σp: jnp.ndarray  # Pulsar-specific sigma values

#     #Measurement noise parameters
#     EFAC: jnp.ndarray  # Error factors
#     EQUAD: jnp.ndarray # Extra quadrature noise

# # --- Fixtures for Test Setup ---

# # For larger class tests
# @pytest.fixture(scope="module")
# def class_test_dims():
#     """Dimensions for the full class tests, matching original assumptions."""
#     Npsr = 2
#     dim_y = 32 # Assumed observation dimension from R/H handling in _update
#     P_eps_dim = Npsr # Simplest epsilon state assumption
#     n_states = Npsr * 2 + Npsr * 2 + P_eps_dim # GW+Spin+Eps = 4+4+2 = 10
#     return {
#         'Npsr': Npsr,
#         'dim_y': dim_y,
#         'P_eps_dim': P_eps_dim,
#         'n_states': n_states,
#         'N_timesteps': 10
#     }

# # For small simple tests
# @pytest.fixture(scope="module")
# def basic_dims():
#     """Basic dimensions for simple tests."""
#     return {'Npsr': 2, 'dim_y': 2, 'P_eps_dim': 2}


# # Updated fixture for parameters using the Flax Dataclass
# @pytest.fixture(scope="module")
# def mock_flax_params(class_test_dims):
#     """Provides an instance of the Parameters flax dataclass."""
#     Npsr = class_test_dims['Npsr'] # Get Npsr from the dims fixture
#     # Ensure array parameters are float64 for consistency
#     dtype = jnp.float64
#     return Parameters(
#         γa = 1e-9,
#         ha = 1e-15,
#         # Ensure pulsar-specific arrays have the correct shape (Npsr,)
#         γp = jnp.full(Npsr, 1e-10, dtype=dtype),
#         σp = jnp.full(Npsr, 1e-11, dtype=dtype),
#         EFAC = jnp.ones(Npsr, dtype=dtype),
#         EQUAD = jnp.zeros(Npsr, dtype=dtype)
#     )


# @pytest.fixture
# def mock_model_instance(class_test_dims, mocker):
#     """Provides a mocked model object for JaxKalmanFilter class tests."""
#     dims = class_test_dims
#     mock_model = mocker.MagicMock()
#     mock_model.Npsr = dims['Npsr']
#     mock_model.M_sum = 1 # Example
#     mock_model.nx = dims['n_states']
#     # Mock precompute_H_matrix to return something with the right shape
#     mock_H_matrices_np = np.random.randn(dims['N_timesteps'], dims['dim_y'], dims['n_states']).astype(np.float64)
#     # ** Store H on the mock **
#     mock_model.H_matrices = jnp.array(mock_H_matrices_np) # Store as jax array
#     mock_model.precompute_H_matrix.return_value = mock_H_matrices_np # Original function returns numpy

#     # Mock HD matrix
#     mock_model.hd_correlation_matrix = np.eye(dims['Npsr']).astype(np.float64)
#     return mock_model

# @pytest.fixture
# def numpy_setup_data(class_test_dims):
#     """Provides numpy arrays for initial filter setup."""
#     dims = class_test_dims
#     # Mock Observations (NumPy arrays initially)
#     mock_toa_np = np.linspace(0, (dims['N_timesteps'] - 1) * 86400, dims['N_timesteps']).astype(np.float64)
#     mock_data_np = np.random.randn(dims['N_timesteps'], dims['dim_y']).astype(np.float64) * 1e-7
#     mock_data_errors_np = np.ones((dims['Npsr'], dims['N_timesteps'])).astype(np.float64) * 1e-7
#     observations_np = [mock_toa_np, mock_data_np, mock_data_errors_np]

#     # Mock Initial State/Covariance (NumPy)
#     mock_x0_np = np.zeros(dims['n_states']).astype(np.float64)
#     mock_P0_np = np.eye(dims['n_states']).astype(np.float64)
#     mock_Peps_np = np.eye(dims['P_eps_dim']).astype(np.float64) * 1e-28

#     return {
#         "observations": observations_np,
#         "x0": mock_x0_np,
#         "P0": mock_P0_np,
#         "Peps": mock_Peps_np,
#         "toa": mock_toa_np,
#         "data": mock_data_np,
#         "data_errors": mock_data_errors_np
#     }


# import os
# @pytest.fixture(scope="module")
# def IPTA_MDC2_data():
#     #Get the data. We will use the mock data for this test

#     # Get the directory of the current script
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     # Construct the invariant directory path
#     data_path = os.path.join(script_dir, "../data/IPTA_MockDataChallenge2/dataset_2b/")


#     # Get all .par and .tim files in the directory
#     par_files = sorted(glob.glob(data_path + "*.par"))
#     tim_files = sorted(glob.glob(data_path + "*.tim"))
#     assert len(par_files) == len(tim_files)

#     #Exclude PR J1640+2224 as it has an exponent which is to small for the OU process to be valid
#     par_files = [f for f in par_files if "J1640" not in f]
#     tim_files = [f for f in tim_files if "J1640" not in f]



#     # Get the data
#     pulsar_residuals, pulsar_metadata, pulsar_design_matrices,P_eps_matrices = (
#         data_loader.LoadWidebandPulsarData.read_multiple_par_tim(par_files, tim_files)
#     )

#     # Get the separation angles and compute HD correlation
#     ra = pulsar_metadata["RA"].to_numpy(dtype=float)
#     dec = pulsar_metadata["DEC"].to_numpy(dtype=float)
#     angular_separation_matrix = gravitational_waves.pairwise_angular_separation(ra, dec)
#     hd_correlation_matrix = gravitational_waves.hellings_downs(angular_separation_matrix)

#     # Post-process the residuals    
#     processed_pulsar_residuals = data_loader.LoadWidebandPulsarData.process_pulsar_residuals_by_epoch(pulsar_residuals)

#     return processed_pulsar_residuals, pulsar_metadata, pulsar_design_matrices,P_eps_matrices,hd_correlation_matrix






# # --- Test for the Main Scan Function ---
# def test_run_kalman_filter_scan(mocker, class_test_dims, numpy_setup_data, mock_flax_params):
#     """Test the main Kalman filter loop (_run_kalman_filter_scan)."""
#     # Setup parameters and dimensions from fixtures
#     dims = class_test_dims
#     Npsr = dims['Npsr']
#     M_sum = 10*Npsr # Example
#     n_states = dims['n_states']
#     dim_y = dims['dim_y']
#     N_timesteps = dims['N_timesteps']
#     dim_x_predict = 2 * Npsr

#     # Use the flax dataclass instance from the fixture
#     params = mock_flax_params

#     # Get data from fixture and convert to JAX arrays
#     setup_data = numpy_setup_data
#     mock_data = jnp.array(setup_data['data'])
#     mock_data_errors = jnp.array(setup_data['data_errors'])
#     mock_dt_array = jnp.diff(jnp.array(setup_data['toa']))
#     mock_P_eps = jnp.array(setup_data['Peps'])

#     # Mock H matrices: shape (N_timesteps, dim_y, n_states)
#     mock_H_matrices = jnp.array(np.random.randn(N_timesteps, dim_y, n_states), dtype=jnp.float64)

#     # Mock Hellings-Downs Matrix
#     mock_hd_matrix = jnp.eye(Npsr, dtype=jnp.float64) # Simple case

#     # --- Mock Return Values---
#     mock_x0 = jnp.zeros((n_states, 1), dtype=jnp.float64)
#     mock_P0 = jnp.eye(n_states, dtype=jnp.float64)
#     mock_init = mocker.patch('argus.jax_kalman_filter._initialize_kalman_filter', return_value=(mock_x0, mock_P0))

#     mock_R_matrices = jnp.array([jnp.eye(dim_y, dtype=jnp.float64) * 1e-14] * N_timesteps)
#     mock_precompute_R = mocker.patch('argus.model.precompute_R_matrices', return_value=mock_R_matrices)

#     mock_F_gw = jnp.eye(2 * Npsr, dtype=jnp.float64) * 0.99
#     mock_F_spin = jnp.eye(2 * Npsr, dtype=jnp.float64) * 0.98
#     mock_F_matrices = mocker.patch('argus.model.get_F', return_value=(mock_F_gw, mock_F_spin))

#     mock_Q_gw = jnp.eye(2 * Npsr, dtype=jnp.float64) * 1e-20
#     mock_Q_spin = jnp.eye(2 * Npsr, dtype=jnp.float64) * 1e-22
#     mock_Q_matrices = mocker.patch('argus.model.get_Q', return_value=(mock_Q_gw, mock_Q_spin))
    
    
#     mock_σa2_matrix = jnp.eye(Npsr, dtype=jnp.float64) * 1e-20
#     mock_σa2 = mocker.patch('argus.jax_kalman_filter._compute_sigma_matrix', return_value=mock_σa2_matrix)
 
    

#     # --- End Mock Return Values ---

#     # Run the function
#     log_likelihood = jk._run_kalman_filter_scan(
#         θ=params, # Pass the flax dataclass instance
#         data=mock_data,
#         data_errors=mock_data_errors,
#         H_matrices=mock_H_matrices,
#         Npsr=Npsr,
#         M_sum=M_sum,
#         hellings_downs_matrix=mock_hd_matrix,
#         dt_array=mock_dt_array,
#         dim_x=dim_x_predict,
#         n_states=n_states,
#         P_eps=mock_P_eps
#     )

#     # Check that the log likelihood is a jax scalar
#     assert isinstance(log_likelihood, jax.Array)
#     assert log_likelihood.shape == ()

#     # Check these methods were only called once
#     mock_init.assert_called_once()
#     mock_precompute_R.assert_called_once()
#     mock_σa2.assert_called_once()
    

#     # Assertions for functions called inside lax.scan (under JIT)
#     # Expect call_count == 1 due to JIT tracing lax.scan body once
#     assert mock_F_matrices.call_count == 1
#     assert mock_Q_matrices.call_count == 1


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
#         mock_compute_pred_state = mocker.patch('argus.jax_kalman_filter.compute_predicted_state', return_value=mock_xp)
#         mock_compute_pred_cov = mocker.patch('argus.jax_kalman_filter.compute_predicted_covariance', return_value=mock_Pp)
#         xp, Pp = jk._predict(x, P, F_list, Q_list, dim_x)
#         mock_compute_pred_state.assert_called_once_with(F_list, x, dim_x, dim_x)
#         mock_compute_pred_cov.assert_called_once_with(P, F_list, Q_list, dim_x, dim_x)
#         np.testing.assert_allclose(xp, mock_xp)
#         np.testing.assert_allclose(Pp, mock_Pp)

#     def test_update(self, basic_dims):
#         """Test the _update step."""
#         dim_x = 4
#         dim_y = basic_dims['dim_y'] # 2


#         # Write out the update step manually to check the calculations are correct
#         xp = jnp.array([1.0, 0.1, 2.0, 0.2], dtype=jnp.float64).reshape(-1, 1)
#         Pp = jnp.diag(jnp.array([0.5, 0.1, 0.5, 0.1], dtype=jnp.float64))
#         H = jnp.zeros((dim_y, dim_x), dtype=jnp.float64)
#         H = H.at[0, 0].set(1.0).at[1, 2].set(1.0)
#         R_small = jnp.eye(dim_y, dtype=jnp.float64) * 0.2
#         z_small = jnp.array([1.1, 2.1], dtype=jnp.float64).reshape(dim_y, 1)
#         y_expected = z_small - H @ xp
#         S_expected_calc = H @ Pp @ H.T + R_small

#         Sinv_expected = jnp.linalg.inv(S_expected_calc)

#         K_expected_calc = Pp @ H.T @ Sinv_expected
#         x_expected_calc = xp + K_expected_calc @ y_expected
#         I_KH = jnp.eye(dim_x) - K_expected_calc @ H
#         P_expected = I_KH @ Pp @ I_KH.T + K_expected_calc @ R_small @ K_expected_calc.T
        
        
#         #Now call the update step in the Argus code
#         z_reshaped_for_func = jnp.zeros((dim_y, 1), dtype=jnp.float64).at[0:dim_y, 0].set(z_small.flatten())
#         H_reshaped_for_func = jnp.zeros((dim_y, dim_x), dtype=jnp.float64).at[0:dim_y, :].set(H)
#         R_reshaped_for_func = jnp.zeros((dim_y, dim_y), dtype=jnp.float64).at[0:dim_y, 0:dim_y].set(R_small)
#         x, P, y, S = jk._update(xp, Pp, H_reshaped_for_func, R_reshaped_for_func, z_reshaped_for_func)
        
        
#         np.testing.assert_allclose(x, x_expected_calc, rtol=1e-6)
#         np.testing.assert_allclose(P, P_expected, rtol=1e-6, atol=1e-9)
#         np.testing.assert_allclose(y, y_expected, rtol=1e-6)
#         np.testing.assert_allclose(S, S_expected_calc, rtol=1e-6)


#     def test_initialize_kalman_filter(self):
#         """Test the initialization of state and covariance."""
#         Npsr = 2
#         P_eps_dim = Npsr
#         nx = Npsr * 2 + Npsr * 2 + P_eps_dim
#         P_eps = jnp.eye(P_eps_dim, dtype=jnp.float64) * 1e-28
        
        
        
#         σa2 = np.ones((Npsr,Npsr)) 
#         γa = 2
#         σp = np.ones(Npsr) * 1e-8
#         γp = np.ones(Npsr) * 1e-8

#         x0, P0 = jk._initialize_kalman_filter(nx, Npsr, P_eps, σa2, γa,σp**2,γp)
#         assert x0.shape == (nx, 1)
#         np.testing.assert_allclose(x0, jnp.zeros((nx, 1)))
#         assert P0.shape == (nx, nx)
        
        

#         expected_P_aa_init = σa2 / (2.0 * γa)
        
    
#         # Construct expected 4x4 P_GW
#         expected_P_GW = np.zeros((Npsr * 2, Npsr * 2))
#         # Set r variances (indices 0, 2)
#         expected_P_GW[0, 0] = 1e-40
#         expected_P_GW[2, 2] = 1e-40
#         # Set P_aa block (indices 1, 3)
#         expected_P_GW[1, 1] = expected_P_aa_init[0, 0]
#         expected_P_GW[1, 3] = expected_P_aa_init[0, 1]
#         expected_P_GW[3, 1] = expected_P_aa_init[1, 0]
#         expected_P_GW[3, 3] = expected_P_aa_init[1, 1]


#         # P_GW is the top-left 4x4 block of P0
#         P_GW_actual = P0[0:Npsr*2, 0:Npsr*2]      
#         np.testing.assert_allclose(P_GW_actual, expected_P_GW, rtol=1e-6)
        


#         expected_f_variances = σp**2 / (2.0 * γp)

#         expected_P_spin = np.zeros((Npsr * 2, Npsr * 2))
#         # Set phi variances (indices 0, 2)
#         expected_P_spin[0, 0] = 1e-40
#         expected_P_spin[2, 2] = 1e-40
#         # Set f variances (indices 1, 3)
#         expected_P_spin[1, 1] = expected_f_variances[0]
#         expected_P_spin[3, 3] = expected_f_variances[1]


#         # P_spin is the second 4x4 block of P0
#         start_idx = Npsr * 2
#         end_idx = Npsr * 4
#         P_spin_actual = P0[start_idx:end_idx, start_idx:end_idx]
#         np.testing.assert_allclose(P_spin_actual, expected_P_spin, rtol=1e-6)


        
#     def test_init_and_prepare_arrays(self, mock_model_instance, numpy_setup_data, class_test_dims):
#         """Test initialization and _prepare_jax_arrays."""
#         setup_data = numpy_setup_data
#         dims = class_test_dims
#         model = mock_model_instance
#         kf = jk.JaxKalmanFilter(
#             model=model,
#             observations=setup_data['observations'],
#             Peps=setup_data['Peps']
#         )
#         assert kf.model == model
#         np.testing.assert_array_equal(kf.toa, setup_data['toa'])
#         np.testing.assert_array_equal(kf.data, setup_data['data'])
#         np.testing.assert_array_equal(kf.data_errors, setup_data['data_errors'])
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
#         with pytest.raises(ValueError, match="expected"):
#             jk.JaxKalmanFilter(
#                 model=mock_model_instance,
#                 observations=bad_obs,
#                 Peps=setup_data['Peps']
#             )




# import glob 
# from .utils import check_cholesky,check_minimum_eigenvalue
# class TestNumericalStability:


#     def test_for_numerical_stability(self, IPTA_MDC2_data):

#         processed_pulsar_residuals, pulsar_metadata, pulsar_design_matrices,P_eps_matrices,hd_correlation_matrix = IPTA_MDC2_data

#         #Calculate P0 based on the maximum value of the design matrix, and a delta tolerance
#         model_obj = model.StochasticGWBackgroundModel(pulsar_metadata, hd_correlation_matrix, pulsar_design_matrices)

    
#         P0 = block_diag(*P_eps_matrices)


#         KF = jk.JaxKalmanFilter(
#             model=model_obj, 
#             observations=processed_pulsar_residuals, 
#             Peps=P0
#         )
#         #Set the parameters
#         Npsr = model_obj.Npsr
#         params = Parameters(

#             #GW parameters
#             γa=1e-9,
#             ha=1e-15,

#             #Spin parameters
#             γp=jnp.ones(Npsr)*1e-8, #approximate magnitudes
#             σp=jnp.ones(Npsr)*1e-15, #approximate magnitudes

#             #Measurement noise parameters
#             EFAC=jnp.ones(Npsr),
#             EQUAD=jnp.ones(Npsr)*1e-6
#         )


#         #Now implement the Kalman filter manually 
#         θ=params
#         data=KF.jax_data
#         data_errors=KF.jax_data_errors
#         H_matrices=KF.jax_H_matrices
#         Npsr=KF.model.Npsr
#         M_sum=KF.model.M_sum
#         hellings_downs_matrix=KF.hellings_downs_matrix
#         dt_array=KF.jax_t_diffs
#         dim_x=2*KF.model.Npsr
#         n_states=KF.model.nx
#         P_eps=KF.P_eps



#         # This is the start of _run_kalman_filter_scan
#         # def _run_kalman_filter_scan(θ, data, data_errors, H_matrices, Npsr, M_sum,hellings_downs_matrix, dt_array, dim_x,n_states,P_eps):
#         σa2 = jk._compute_sigma_matrix(θ.ha**2, θ.γa, hellings_downs_matrix)
#         x0,P0 = jk._initialize_kalman_filter(n_states,Npsr,P_eps,σa2, θ.γa,θ.σp**2, θ.γp)

    
#         # Precompute the R matrix for this parameter set and these data errors    
#         R_matrices = model.precompute_R_matrices(data_errors,θ.EFAC, θ.EQUAD)


#         # First update
#         x, P, y, S = jk._update(xp=x0, Pp=P0, H=H_matrices[0,:,:], R=R_matrices[0,:,:], z=data[0])
#         ll0 = jk._log_likelihood(y, S)

    

#         #Standard loop, not using lax.scan
#         for i in range(len(data)-1):
#             dt = dt_array[i]
#             F_gw, F_spin = model.get_F(θ.γa, θ.γp, dt, Npsr, M_sum)
#             F = (F_gw, F_spin)
            
#             Q_gw, Q_spin = model.get_Q(θ.γa, σa2, θ.γp, θ.σp**2, dt)
#             Q = (Q_gw, Q_spin)

#             x_predict, P_predict = jk._predict(x, P, F, Q, dim_x)
            
#             x, P, y, S = jk._update(x_predict, P_predict, H_matrices[i+1,:,:], R_matrices[i+1,:,:], data[i+1])

#             assert check_cholesky(S)
#             assert check_cholesky(P)

#             assert check_minimum_eigenvalue(S)
#             assert check_minimum_eigenvalue(P)

#             ll = jk._log_likelihood(y, S)
#             assert ll.shape == (1,1) #make sure the likelihood is a scalar
#             ll0 += ll



#         #Also do the full call and check the result is the same
#         ll = jk._run_kalman_filter_scan(θ, data, data_errors, H_matrices, Npsr, M_sum,hellings_downs_matrix, dt_array, dim_x,n_states,P_eps)


#         print("ll = ", ll)
#         print("ll0 = ", ll0)
#         np.testing.assert_allclose(ll0,ll)
      


       












# #     ll = KF.get_likelihood(params) 
# #     ll.block_until_ready()
# #     print("Likelihood: ",ll)









# # utils.check_cholesky(P_spin, "P_spin")
#     # utils.check_min_eigenvalue(P_spin, "P_spin")
#     # utils.check_symmetry(P_spin, "P_spin")
#     # utils.check_condition_number(P_spin, "P_spin")

#     #jax.debug.print('P_spin is {P_spin}', P_spin=P_spin,ordered=True)








#     # utils.check_cholesky(P, "Pupdated")
#     # utils.check_min_eigenvalue(P, "Pupdated")
#     # utils.check_symmetry(P, "Pupdated")
#     # utils.check_condition_number(P, "Pupdated")





#     # check_cholesky(P0,"The initial P-matrix")
#     # check_min_eigenvalue(P0, "The initial P-matrix")
#     # check_symmetry(P0, "The initial P-matrix")
#     # check_condition_number(P0, "The initial P-matrix")
