"""Tests for the null model functionality in StochasticGWBackgroundModel."""

import pytest
import numpy as np
import jax.numpy as jnp
from argus.models import StochasticGWBackgroundModel
from argus.jax_kalman_filter import JaxKalmanFilter

class MockParameters:
    """Mock class to simulate the parameters object used in the Kalman filter."""
    def __init__(self, ha=1e-15, γa=1e-8, σp=1e-15, γp=1e-8, EFAC=1.0, EQUAD=1e-9):
        self.ha = ha
        self.γa = γa
        self.σp = σp
        self.γp = γp
        self.EFAC = EFAC
        self.EQUAD = EQUAD

@pytest.fixture
def mock_data():
    """Create mock data for testing."""
    # Create a simple dataset with 2 pulsars and 3 time steps
    Npsr = 2
    Ntime = 3
    
    # Mock pulsar data
    df_psr = {
        "dim_M": [2, 2],  # 2 design parameters per pulsar
        "gamma_p": [1e-8, 1e-8],
        "sigma_p": [1e-15, 1e-15],
        "F0": [100, 100],  # 100 Hz spin frequency
    }
    
    # Mock Hellings-Downs matrix
    hd_matrix = np.eye(Npsr)
    
    # Mock design matrices (2 parameters per pulsar)
    design_matrices = [
        np.ones((Ntime, 2)),  # Pulsar 1
        np.ones((Ntime, 2))   # Pulsar 2
    ]
    
    # Mock observations
    toa = np.array([0, 1, 2])  # Time of arrivals
    data = np.random.randn(Ntime, Npsr)  # Random residuals
    data_errors = np.ones((Ntime, Npsr))  # Unit errors
    
    return {
        "df_psr": df_psr,
        "hd_matrix": hd_matrix,
        "design_matrices": design_matrices,
        "observations": np.array([toa, data, data_errors])
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
    assert not np.all(H[:, 2:4] == 0)  # Spin terms
    assert not np.all(H[:, 4:] == 0)   # Timing terms

def test_h_matrix_with_gw(mock_data):
    """Test that H matrix is correctly computed with GW terms."""
    model = StochasticGWBackgroundModel(
        mock_data["df_psr"],
        mock_data["hd_matrix"],
        mock_data["design_matrices"],
        use_gw=True
    )
    
    H = model.compute_H_matrix_for_step(0)
    
    # Check that GW terms are present (-1.0 in first column)
    assert np.all(H[:, 0] == -1.0)
    assert np.all(H[:, 1] == 0)  # Second column should be zero
    
    # Check that spin and timing terms are present
    assert not np.all(H[:, 2:4] == 0)  # Spin terms
    assert not np.all(H[:, 4:] == 0)   # Timing terms

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
    params1 = MockParameters(ha=1e-15, γa=1e-8)
    params2 = MockParameters(ha=1e-14, γa=1e-7)  # Different GW parameters
    
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
    params = MockParameters(ha=0.0, γa=1e-8)  # Zero GW amplitude
    
    # Get likelihoods
    ll_with_gw = kf_with_gw.get_likelihood(params)
    ll_without_gw = kf_without_gw.get_likelihood(params)
    
    # Likelihoods should be equal when GW effect is zero
    assert np.isclose(ll_with_gw, ll_without_gw) 