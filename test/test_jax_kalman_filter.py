import pytest
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd


# # --- JAX Configuration for CPU ---
# Check for GPU availability and configure JAX accordingly
try:
    gpu_devices = jax.devices('gpu')
    if gpu_devices:
        print("GPU found. Configuring JAX to use GPU.")
        jax.config.update("jax_platforms", "gpu")
    else:
        print("No GPU found. Configuring JAX to use CPU.")
        jax.config.update("jax_platforms", "cpu")
except Exception as e:
    print(f"Error checking for GPU: {e}")
    print("Falling back to CPU.")
    jax.config.update("jax_platforms", "cpu")
# ---------------------------------
# Enable 64-bit precision in JAX for numerical stability
jax.config.update("jax_enable_x64", True)
from jax.scipy.linalg import block_diag


# from collections import namedtuple # No longer needed for params
from unittest.mock import ANY
from flax import struct # Import flax struct

# Assume the file is named jax_kalman_filter.py
from argus import jax_kalman_filter as jk
from argus import data_loader,model


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

# For larger class tests
@pytest.fixture(scope="module")
def class_test_dims():
    """Dimensions for the full class tests, matching original assumptions."""
    Npsr = 3
    dim_y = Npsr # Assumed observation dimension 
    P_eps_dim = 2*Npsr # Simplest epsilon state assumption. 2 parameter per pulsar
    n_states = Npsr * 2 + Npsr * 2 + P_eps_dim # GW+Spin+Eps = 4+4+2 = 10
    return {
        'Npsr': Npsr,
        'dim_y': dim_y,
        'P_eps_dim': P_eps_dim,
        'n_states': n_states,
        'N_timesteps': 17
    }

# For small simple tests
@pytest.fixture(scope="module")
def basic_dims():
    """Basic dimensions for simple tests."""
    return {'Npsr': 2, 'dim_y': 2, 'P_eps_dim': 2}


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


@pytest.fixture
def mock_model_instance(class_test_dims, mocker):
    """Provides a mocked model object for JaxKalmanFilter class tests."""
    dims = class_test_dims
    mock_model = mocker.MagicMock()
    mock_model.Npsr = dims['Npsr']
    mock_model.M_sum = dims['P_eps_dim'] #1 # Example
    mock_model.nx = dims['n_states']
    
    # Create mock H matrices with correct structure
    N_timesteps = dims['N_timesteps']
    Npsr = dims['Npsr']
    nx = dims['n_states']
    
    # Initialize H matrices for all time steps
    mock_H_matrices_np = np.zeros((N_timesteps, Npsr, nx), dtype=np.float64)
    

    #M_start_indices = np.cumsum([0] + [m for m in dims['dim_M']]) + 4 * Npsr


    # # For each time step
    # for t in range(N_timesteps):
    #     # For each pulsar
    #     for psr_idx in range(Npsr):
    #         # GW terms (redshift)
    #         redshift_idx = 2 * psr_idx
    #         spin_idx = Npsr * 2 + 2 * psr_idx
    #         tm_start_idx = M_start_indices[psr_idx]  # Assuming 1 timing parameter per pulsar
    #         tm_end_idx = M_start_indices[psr_idx + 1]


    #         #design_row = pulsar_design_matrices[psr_idx][t, :]

            
    #         mock_H_matrices_np[t, psr_idx, redshift_idx] = -1.0
            
    #         # Spin noise terms
            
    #         mock_H_matrices_np[t, psr_idx, spin_idx] = 1.0 / 100.0  # Using 100 Hz as mock frequency
            
    #         # Timing model terms (using simple design matrix)
    #         mock_H_matrices_np[t, psr_idx, tm_start_idx] = 1.0
    
    # Store H on the mock
    mock_model.H_matrices = jnp.array(mock_H_matrices_np) # Store as jax array
    mock_model.precompute_H_matrices.return_value = mock_H_matrices_np # Original function returns numpy

    # Mock HD matrix
    mock_model.hd_correlation_matrix = np.eye(dims['Npsr']).astype(np.float64)
    return mock_model

@pytest.fixture
def numpy_setup_data(class_test_dims):
    """Provides numpy arrays for initial filter setup."""
    dims = class_test_dims
    
    # Create mock processed residuals
    mock_toa_np = np.linspace(0, (dims['N_timesteps'] - 1) * 86400, dims['N_timesteps']).astype(np.float64)
    mock_data_np = np.random.randn(dims['N_timesteps'], dims['Npsr']).astype(np.float64) * 1e-7
    mock_data_errors_np = np.ones((dims['N_timesteps'], dims['Npsr'])).astype(np.float64) * 1e-7
    
    processed_residuals = {
        'average_toas': mock_toa_np,
        'residuals': mock_data_np,
        'error': mock_data_errors_np
    }
    
    # Create mock metadata DataFrame
    metadata = pd.DataFrame({
        'name': [f'PSR{i}' for i in range(dims['Npsr'])],
        'dim_M': [2] * dims['Npsr'],  # Assuming 1 design parameters per pulsar
        'RA': np.random.uniform(0, 2*np.pi, dims['Npsr']),
        'DEC': np.random.uniform(-np.pi/2, np.pi/2, dims['Npsr']),
        'F0': np.ones(dims['Npsr']) * 100  # 100 Hz spin frequency
    })
    
    # Create mock design matrices
    design_matrices = [np.ones((dims['N_timesteps'], 2)) for _ in range(dims['Npsr'])]
    
    # Create mock parameter covariance matrices
    parameter_covariances = [np.eye(2) * 1e-28 for _ in range(dims['Npsr'])]
    
    # Create mock Hellings-Downs correlation matrix
    hd_correlation = np.eye(dims['Npsr'])
    
    return {
        'processed_residuals': processed_residuals,
        'metadata': metadata,
        'design_matrices': design_matrices,
        'parameter_covariances': parameter_covariances,
        'hd_correlation': hd_correlation
    }


import os
@pytest.fixture(scope="module")
def IPTA_MDC2_data():
    #Get the data. We will use the IPTA2 mock data for this test
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "../data/IPTA_MockDataChallenge2/dataset_2b/")
    return data_loader.get_processed_residuals(data_path,excluded_psrs=["J1640"])




# --- Test for the Main Scan Function ---
def test_run_kalman_filter_scan(mocker, class_test_dims, numpy_setup_data, mock_flax_params):
    """Test the main Kalman filter loop (_run_kalman_filter_scan)."""
    # Setup parameters and dimensions from fixtures
    dims = class_test_dims
    Npsr = dims['Npsr']
    M_sum = 10*Npsr # Example
    n_states = dims['n_states']
    dim_y = dims['dim_y']
    N_timesteps = dims['N_timesteps']
    dim_x_predict = 2 * Npsr

    # Use the flax dataclass instance from the fixture
    params = mock_flax_params

    # Get data from fixture and convert to JAX arrays
    setup_data = numpy_setup_data
    mock_data = jnp.array(setup_data['processed_residuals']['residuals'])
    mock_data_errors = jnp.array(setup_data['processed_residuals']['error'])
    mock_dt_array = jnp.diff(jnp.array(setup_data['processed_residuals']['average_toas']))
    mock_P_eps = jnp.array(setup_data['parameter_covariances'][0])

    # Mock H matrices: shape (N_timesteps, dim_y, n_states)
    mock_H_matrices = jnp.array(np.random.randn(N_timesteps, dim_y, n_states), dtype=jnp.float64)

    # Mock Hellings-Downs Matrix
    mock_hd_matrix = jnp.eye(Npsr, dtype=jnp.float64) # Simple case

    # --- Mock Return Values---
    mock_x0 = jnp.zeros((n_states, 1), dtype=jnp.float64)
    mock_P0 = jnp.eye(n_states, dtype=jnp.float64)
    mock_init = mocker.patch('argus.jax_kalman_filter._initialize_kalman_filter', return_value=(mock_x0, mock_P0))

    mock_R_matrices = jnp.array([jnp.eye(dim_y, dtype=jnp.float64) * 1e-14] * N_timesteps)
    mock_precompute_R = mocker.patch('argus.jax_kalman_filter.precompute_R_matrices', return_value=mock_R_matrices)

    mock_F_gw = jnp.eye(2 * Npsr, dtype=jnp.float64) * 0.99
    mock_F_spin = jnp.eye(2 * Npsr, dtype=jnp.float64) * 0.98
    mock_F_matrices = mocker.patch('argus.jax_kalman_filter.get_F', return_value=(mock_F_gw, mock_F_spin))

    mock_Q_gw = jnp.eye(2 * Npsr, dtype=jnp.float64) * 1e-20
    mock_Q_spin = jnp.eye(2 * Npsr, dtype=jnp.float64) * 1e-22
    mock_Q_matrices = mocker.patch('argus.jax_kalman_filter.get_Q', return_value=(mock_Q_gw, mock_Q_spin))
    
    
    mock_σa2_matrix = jnp.eye(Npsr, dtype=jnp.float64) * 1e-20
    mock_σa2 = mocker.patch('argus.jax_kalman_filter._compute_sigma_matrix', return_value=mock_σa2_matrix)
 
    

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

    # Check these methods were only called once
    mock_init.assert_called_once()
    mock_precompute_R.assert_called_once()
    mock_σa2.assert_called_once()
    

    # Assertions for functions called inside lax.scan (under JIT)
    # Expect call_count == 1 due to JIT tracing lax.scan body once
    assert mock_F_matrices.call_count == 1
    assert mock_Q_matrices.call_count == 1


class TestJaxKalmanFilterInternals:

    @pytest.mark.parametrize("y, cov, expected_ll_approx", [
        (jnp.array([2.0]), jnp.array([[4.0]]), -2.112), # Scalar case
        (jnp.array([1.0, -1.0]), jnp.array([[2.0, 0.5], [0.5, 1.0]]), -3.260), # Vector case
    ])
    def test_log_likelihood(self, y, cov, expected_ll_approx):
        """Test the _log_likelihood calculation."""
        sign, logdet = jnp.linalg.slogdet(2.0 * jnp.pi * cov)
        inv_cov = jnp.linalg.inv(cov)
        quad_term = y.T @ inv_cov @ y
        expected_ll = -0.5 * (logdet + quad_term)
        ll = jk._log_likelihood(y, cov)
        np.testing.assert_allclose(ll, expected_ll, rtol=1e-5)
        np.testing.assert_allclose(ll, expected_ll_approx, rtol=1e-3)

    def test_predict(self, mocker):
        """Test the _predict step, mocking external compute functions."""
        dim_x = 4
        x = jnp.ones((dim_x, 1), dtype=jnp.float64)
        P = jnp.eye(dim_x, dtype=jnp.float64) * 2.0
        F_list = (jnp.eye(dim_x, dtype=jnp.float64) * 1.1,)
        Q_list = (jnp.eye(dim_x, dtype=jnp.float64) * 0.1,)
        mock_xp = jnp.ones((dim_x, 1), dtype=jnp.float64) * 1.1
        mock_Pp = (jnp.eye(dim_x) * 1.1) @ P @ (jnp.eye(dim_x) * 1.1).T + (jnp.eye(dim_x) * 0.1)
        mock_compute_pred_state = mocker.patch('argus.jax_kalman_filter.compute_predicted_state', return_value=mock_xp)
        mock_compute_pred_cov = mocker.patch('argus.jax_kalman_filter.compute_predicted_covariance', return_value=mock_Pp)
        xp, Pp = jk._predict(x, P, F_list, Q_list, dim_x)
        mock_compute_pred_state.assert_called_once_with(F_list, x, dim_x, dim_x)
        mock_compute_pred_cov.assert_called_once_with(P, F_list, Q_list, dim_x, dim_x)
        np.testing.assert_allclose(xp, mock_xp)
        np.testing.assert_allclose(Pp, mock_Pp)

    def test_update(self, basic_dims):
        """Test the _update step."""
        dim_x = 4
        dim_y = basic_dims['dim_y'] # 2


        # Write out the update step manually to check the calculations are correct
        xp = jnp.array([1.0, 0.1, 2.0, 0.2], dtype=jnp.float64).reshape(-1, 1)
        Pp = jnp.diag(jnp.array([0.5, 0.1, 0.5, 0.1], dtype=jnp.float64))
        H = jnp.zeros((dim_y, dim_x), dtype=jnp.float64)
        H = H.at[0, 0].set(1.0).at[1, 2].set(1.0)
        R_small = jnp.eye(dim_y, dtype=jnp.float64) * 0.2
        z_small = jnp.array([1.1, 2.1], dtype=jnp.float64).reshape(dim_y, 1)
        y_expected = z_small - H @ xp
        S_expected_calc = H @ Pp @ H.T + R_small

        Sinv_expected = jnp.linalg.inv(S_expected_calc)

        K_expected_calc = Pp @ H.T @ Sinv_expected
        x_expected_calc = xp + K_expected_calc @ y_expected
        I_KH = jnp.eye(dim_x) - K_expected_calc @ H
        P_expected = I_KH @ Pp @ I_KH.T + K_expected_calc @ R_small @ K_expected_calc.T
        
        
        #Now call the update step in the Argus code
        z_reshaped_for_func = jnp.zeros((dim_y, 1), dtype=jnp.float64).at[0:dim_y, 0].set(z_small.flatten())
        H_reshaped_for_func = jnp.zeros((dim_y, dim_x), dtype=jnp.float64).at[0:dim_y, :].set(H)
        R_reshaped_for_func = jnp.zeros((dim_y, dim_y), dtype=jnp.float64).at[0:dim_y, 0:dim_y].set(R_small)
        x, P, y, S = jk._update(xp, Pp, H_reshaped_for_func, R_reshaped_for_func, z_reshaped_for_func)
        
        
        np.testing.assert_allclose(x, x_expected_calc, rtol=1e-6)
        np.testing.assert_allclose(P, P_expected, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(y, y_expected, rtol=1e-6)
        np.testing.assert_allclose(S, S_expected_calc, rtol=1e-6)


    def test_initialize_kalman_filter(self):
        """Test the initialization of state and covariance."""
        Npsr = 2
        P_eps_dim = Npsr
        nx = Npsr * 2 + Npsr * 2 + P_eps_dim
        P_eps = jnp.eye(P_eps_dim, dtype=jnp.float64) * 1e-28
        
        
        
        σa2 = np.ones((Npsr,Npsr)) 
        γa = 2
        σp = np.ones(Npsr) * 1e-8
        γp = np.ones(Npsr) * 1e-8

        x0, P0 = jk._initialize_kalman_filter(nx, Npsr, P_eps, σa2, γa,σp**2,γp)
        assert x0.shape == (nx, 1)
        np.testing.assert_allclose(x0, jnp.zeros((nx, 1)))
        assert P0.shape == (nx, nx)
        
        

        expected_P_aa_init = σa2 / (2.0 * γa)
        
    
        # Construct expected 4x4 P_GW
        expected_P_GW = np.zeros((Npsr * 2, Npsr * 2))
        # Set r variances (indices 0, 2)
        expected_P_GW[0, 0] = 1e-40
        expected_P_GW[2, 2] = 1e-40
        # Set P_aa block (indices 1, 3)
        expected_P_GW[1, 1] = expected_P_aa_init[0, 0]
        expected_P_GW[1, 3] = expected_P_aa_init[0, 1]
        expected_P_GW[3, 1] = expected_P_aa_init[1, 0]
        expected_P_GW[3, 3] = expected_P_aa_init[1, 1]


        # P_GW is the top-left 4x4 block of P0
        P_GW_actual = P0[0:Npsr*2, 0:Npsr*2]      
        np.testing.assert_allclose(P_GW_actual, expected_P_GW, rtol=1e-6)
        


        expected_f_variances = σp**2 / (2.0 * γp)

        expected_P_spin = np.zeros((Npsr * 2, Npsr * 2))
        # Set phi variances (indices 0, 2)
        expected_P_spin[0, 0] = 1e-40
        expected_P_spin[2, 2] = 1e-40
        # Set f variances (indices 1, 3)
        expected_P_spin[1, 1] = expected_f_variances[0]
        expected_P_spin[3, 3] = expected_f_variances[1]


        # P_spin is the second 4x4 block of P0
        start_idx = Npsr * 2
        end_idx = Npsr * 4
        P_spin_actual = P0[start_idx:end_idx, start_idx:end_idx]
        np.testing.assert_allclose(P_spin_actual, expected_P_spin, rtol=1e-6)


        
    def test_init_and_prepare_arrays(self, mock_model_instance, numpy_setup_data):
        """Test initialization and _prepare_jax_arrays."""
        setup_data = numpy_setup_data



        print("Checking out shapes")
        kf = jk.JaxKalmanFilter(
            data_dict=numpy_setup_data,
            P0_scaling=1.0,
            use_gw=True
        )

        

        # Test the data was loaded correctly
        np.testing.assert_array_equal(kf.toa, setup_data['processed_residuals']['average_toas'])
        np.testing.assert_array_equal(kf.data, setup_data['processed_residuals']['residuals'])
        np.testing.assert_array_equal(kf.data_errors, setup_data['processed_residuals']['error'])
        np.testing.assert_array_equal(kf.P_eps, block_diag(*setup_data['parameter_covariances']))
        
        # # Test the H matrix was computed correctly
        # mock_model_instance.precompute_H_matrices.assert_called_once()
        # np.testing.assert_array_equal(kf.Hmat, mock_model_instance.precompute_H_matrices.return_value)
        
        # Test JAX array conversions
        assert isinstance(kf.jax_data, jax.Array)
        assert kf.jax_data.dtype == jnp.float64
        np.testing.assert_allclose(kf.jax_data, setup_data['processed_residuals']['residuals'])
        
        assert isinstance(kf.jax_data_errors, jax.Array)
        assert kf.jax_data_errors.dtype == jnp.float64
        np.testing.assert_allclose(kf.jax_data_errors, setup_data['processed_residuals']['error'])
        
        assert isinstance(kf.jax_t_diffs, jax.Array)
        assert kf.jax_t_diffs.dtype == jnp.float64
        np.testing.assert_allclose(kf.jax_t_diffs, np.diff(setup_data['processed_residuals']['average_toas']))
        
        assert isinstance(kf.jax_H_matrices, jax.Array)
        assert kf.jax_H_matrices.dtype == jnp.float64



        #to be checked
        #np.testing.assert_allclose(kf.jax_H_matrices, mock_model_instance.precompute_H_matrices.return_value)
        
        assert isinstance(kf.hellings_downs_matrix, jax.Array)
        assert kf.hellings_downs_matrix.dtype == jnp.float64
        np.testing.assert_allclose(kf.hellings_downs_matrix, setup_data['hd_correlation'])

    def test_prepare_arrays_raises_error_on_wrong_dtype(self, mock_model_instance, numpy_setup_data):
        """Test that _prepare_jax_arrays raises ValueError for non-float64."""
        setup_data = numpy_setup_data
        bad_obs = [
            setup_data['processed_residuals']['average_toas'].astype(np.float32),
            setup_data['processed_residuals']['residuals'].astype(np.float32),
            setup_data['processed_residuals']['error'].astype(np.float32)
        ]
        with pytest.raises(ValueError, match="expected"):
            jk.JaxKalmanFilter(
                model=mock_model_instance,
                observations=bad_obs,
                Peps=setup_data['parameter_covariances'][0]
            )




import glob 
from argus import model
from .utils import check_cholesky,check_minimum_eigenvalue
class TestNumericalStability:


    def test_for_numerical_stability(self, IPTA_MDC2_data):
        KF = jk.JaxKalmanFilter(data_dict=IPTA_MDC2_data,P0_scaling=1.0,use_gw=True)
        
        #Set the parameters
        Npsr = KF.Npsr
        params = Parameters(

            #GW parameters
            γa=1e-9,
            ha=1e-15,

            #Spin parameters
            γp=jnp.ones(Npsr)*1e-8, #approximate magnitudes
            σp=jnp.ones(Npsr)*1e-15, #approximate magnitudes

            #Measurement noise parameters
            EFAC=jnp.ones(Npsr),
            EQUAD=jnp.ones(Npsr)*1e-6
        )


        #Now implement the Kalman filter manually 
        θ=params
        data=KF.jax_data
        data_errors=KF.jax_data_errors
        H_matrices=KF.jax_H_matrices
        Npsr=KF.model.Npsr
        M_sum=KF.model.M_sum
        hellings_downs_matrix=KF.hellings_downs_matrix
        dt_array=KF.jax_t_diffs
        dim_x=2*KF.model.Npsr
        n_states=KF.model.nx
        P_eps=KF.P_eps



        # This is the start of _run_kalman_filter_scan
        # def _run_kalman_filter_scan(θ, data, data_errors, H_matrices, Npsr, M_sum,hellings_downs_matrix, dt_array, dim_x,n_states,P_eps):
        σa2 = jk._compute_sigma_matrix(θ.ha**2, θ.γa, hellings_downs_matrix)
        x0,P0 = jk._initialize_kalman_filter(n_states,Npsr,P_eps,σa2, θ.γa,θ.σp**2, θ.γp)

    
        # Precompute the R matrix for this parameter set and these data errors    
        R_matrices = model.precompute_R_matrices(data_errors,θ.EFAC, θ.EQUAD)


        # First update
        x, P, y, S = jk._update(xp=x0, Pp=P0, H=H_matrices[0,:,:], R=R_matrices[0,:,:], z=data[0])
        ll0 = jk._log_likelihood(y, S)

    

        #Standard loop, not using lax.scan
        for i in range(len(data)-1):
            dt = dt_array[i]
            F_gw, F_spin = model.get_F(θ.γa, θ.γp, dt, Npsr, M_sum)
            F = (F_gw, F_spin)
            
            Q_gw, Q_spin = model.get_Q(θ.γa, σa2, θ.γp, θ.σp**2, dt)
            Q = (Q_gw, Q_spin)

            x_predict, P_predict = jk._predict(x, P, F, Q, dim_x)
            
            x, P, y, S = jk._update(x_predict, P_predict, H_matrices[i+1,:,:], R_matrices[i+1,:,:], data[i+1])

            assert check_cholesky(S)
            assert check_cholesky(P)

            assert check_minimum_eigenvalue(S)
            assert check_minimum_eigenvalue(P)

            ll = jk._log_likelihood(y, S)
            assert ll.shape == (1,1) #make sure the likelihood is a scalar
            ll0 += ll



        #Also do the full call and check the result is the same
        ll = jk._run_kalman_filter_scan(θ, data, data_errors, H_matrices, Npsr, M_sum,hellings_downs_matrix, dt_array, dim_x,n_states,P_eps)


        print("ll = ", ll)
        print("ll0 = ", ll0)
        np.testing.assert_allclose(ll0,ll)
      


       





