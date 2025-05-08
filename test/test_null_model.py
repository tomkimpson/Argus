"""Tests for the null model functionality in StochasticGWBackgroundModel."""

import jax
jax.config.update("jax_enable_x64", True)

import pytest
import numpy as np
import jax.numpy as jnp
import pandas as pd
from flax import struct
from argus.models import StochasticGWBackgroundModel
from argus.jax_kalman_filter import JaxKalmanFilter

@struct.dataclass
class MockParameters:
    """Mock class to simulate the parameters object used in the Kalman filter."""
    #GW parameters
    γa: float  # s⁻¹
    ha: float  # GWB amplitude

    #Pulsar parameters for the OU process
    γp: jnp.ndarray  # Pulsar-specific gamma values
    σp: jnp.ndarray  # Pulsar-specific sigma values 

    #Measurement noise parameters
    EFAC: jnp.ndarray  # Error factors
    EQUAD: jnp.ndarray # Extra quadrature noise





@pytest.fixture
def mock_data():
    """Create mock data for testing."""
    # Create a simple dataset with 2 pulsars and 3 time steps
    Npsr = 2
    Ntime = 3
    
    # Mock pulsar data as a pandas DataFrame
    df_psr = pd.DataFrame({
        "dim_M": [2, 2],  # 2 design parameters per pulsar
        "gamma_p": [1e-8, 1e-8],
        "sigma_p": [1e-15, 1e-15],
        "F0": [100, 100],  # 100 Hz spin frequency
    })
    
    # Mock Hellings-Downs matrix
    hd_matrix = np.eye(Npsr)
    
    # Mock design matrices (2 parameters per pulsar)
    design_matrices = [
        np.ones((Ntime, 2)),  # Pulsar 1
        np.ones((Ntime, 2))   # Pulsar 2
    ]
    
    # Mock observations - properly structured as a 3D array
    # Create arrays with consistent shapes (Ntime, Npsr)
    toa = np.tile(np.array([0, 1, 2])[:, np.newaxis], (1, Npsr))  # Shape: (Ntime, Npsr)
    data = np.random.randn(Ntime, Npsr)  # Shape: (Ntime, Npsr)
    data_errors = np.ones((Ntime, Npsr))  # Shape: (Ntime, Npsr)
    
    # Stack the observations into a single 3D array
    observations = np.stack([toa, data, data_errors], axis=0)
    
    return {
        "df_psr": df_psr,
        "hd_matrix": hd_matrix,
        "design_matrices": design_matrices,
        "observations": observations
    }

def test_h_matrix_without_gw(mock_data):
    """Test that H matrix is correctly computed without GW terms."""
    model = StochasticGWBackgroundModel(
        mock_data["df_psr"],
        mock_data["hd_matrix"],
        mock_data["design_matrices"],
        use_gw=False
    )
    
    H = model.compute_H_matrix_for_step(0)
    
    # Check that GW terms (first two columns) are zero
    assert np.all(H[:, :2] == 0)
    
    # Check that spin and timing terms are present
    assert not np.all(H[:, 4:8] == 0)  # Spin terms
    assert not np.all(H[:, 8:] == 0)   # Timing terms

def test_h_matrix_with_gw(mock_data):
    """Test that H matrix is correctly computed with GW terms."""
    model = StochasticGWBackgroundModel(
        mock_data["df_psr"],
        mock_data["hd_matrix"],
        mock_data["design_matrices"],
        use_gw=True
    )
    

    H = model.compute_H_matrix_for_step(0)
    print("The H matrix is: ", H)
    print(H.shape)
    print(H)
    
    # Check that GW terms are present 
    ## For the first pulsar
    assert H[0, 0] == -1.0
    assert H[0, 1] == 0.0 #Second column should be zero

    ## For the second pulsar
    assert H[1, 2] == -1.0
    assert H[1, 3] == 0.0 #Second column should be zero
    
    # Check that spin and timing terms are present
    assert not np.all(H[:, 4:8] == 0)  # Spin terms
    assert not np.all(H[:, 8:] == 0)   # Timing terms

def test_likelihood_independence(mock_data):
    """Test that likelihood is independent of GW parameters when use_gw=False."""
    # Create two models with different GW parameters
    model1 = StochasticGWBackgroundModel(
        mock_data["df_psr"],
        mock_data["hd_matrix"],
        mock_data["design_matrices"],
        use_gw=False
    )
    
    model2 = StochasticGWBackgroundModel(
        mock_data["df_psr"],
        mock_data["hd_matrix"],
        mock_data["design_matrices"],
        use_gw=False
    )
    
    # Create Kalman filters
    kf1 = JaxKalmanFilter(model1, mock_data["observations"], np.eye(4))
    kf2 = JaxKalmanFilter(model2, mock_data["observations"], np.eye(4))
    
    # Create parameters with different GW values
    σp = np.array([1e-15, 1e-15])
    γp = np.array([1e-8, 1e-8])
    EFAC = np.array([1.0, 1.0])
    EQUAD = np.array([1e-6, 1e-6])
    params1 = MockParameters(ha=1e-15, γa=1e-8, σp=σp, γp=γp, EFAC=EFAC, EQUAD=EQUAD)
    params2 = MockParameters(ha=1e-14, γa=1e-7, σp=σp, γp=γp, EFAC=EFAC, EQUAD=EQUAD)  # Different GW parameters


    print("the first paramers are ")
    print(params1)
    print('--------------------')
    
    # Get likelihoods
    ll1 = kf1.get_likelihood(params1)
    ll2 = kf2.get_likelihood(params2)
    
    # Likelihoods should be equal despite different GW parameters
    assert np.isclose(ll1, ll2)

def test_likelihood_consistency(mock_data):
    """Test that likelihood is consistent between use_gw=True and use_gw=False 
    when GW parameters are zero."""
    # Create models with and without GW
    model_with_gw = StochasticGWBackgroundModel(
        mock_data["df_psr"],
        mock_data["hd_matrix"],
        mock_data["design_matrices"],
        use_gw=True
    )
    
    model_without_gw = StochasticGWBackgroundModel(
        mock_data["df_psr"],
        mock_data["hd_matrix"],
        mock_data["design_matrices"],
        use_gw=False
    )
    
    # Create Kalman filters
    kf_with_gw = JaxKalmanFilter(model_with_gw, mock_data["observations"], np.eye(4))
    kf_without_gw = JaxKalmanFilter(model_without_gw, mock_data["observations"], np.eye(4))
    
    # Create parameters with zero GW effect
    params = MockParameters(ha=0.0, γa=1e-8, σp=1e-15, γp=1e-8, EFAC=1.0, EQUAD=1e-9)  # Zero GW amplitude
    
    # Get likelihoods
    ll_with_gw = kf_with_gw.get_likelihood(params)
    ll_without_gw = kf_without_gw.get_likelihood(params)
    
    # Likelihoods should be equal when GW effect is zero
    assert np.isclose(ll_with_gw, ll_without_gw) 