import pytest
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from argus.bayesian_inference import (
    Parameters,
    configurable_prior_model,
    gw_prior_model,
    null_prior_model,
    jaxns_log_likelihood,
    print_parameters
)

# Test Parameters class
def test_parameters_class():
    # Create test data
    Npsr = 3
    γa = 1e-9
    ha = 1e-15
    γp = jnp.array([1e-8, 1e-7, 1e-6])
    σp = jnp.array([1e-15, 1e-14, 1e-13])
    EFAC = jnp.array([1.0, 1.1, 1.2])
    EQUAD = jnp.array([1e-8, 1e-7, 1e-6])

    # Create Parameters instance
    params = Parameters(
        γa=γa,
        ha=ha,
        γp=γp,
        σp=σp,
        EFAC=EFAC,
        EQUAD=EQUAD
    )

    # Test that all fields are correctly assigned
    assert params.γa == γa
    assert params.ha == ha
    assert jnp.allclose(params.γp, γp)
    assert jnp.allclose(params.σp, σp)
    assert jnp.allclose(params.EFAC, EFAC)
    assert jnp.allclose(params.EQUAD, EQUAD)

# Test prior models
def test_configurable_prior_model():
    Npsr = 3
    # Test with fixed values
    model = configurable_prior_model(
        Npsr=Npsr,
        log10_ha_spec=-15.0,
        gamma_a_spec=1e-9,
        log10_gamma_p_spec=jnp.array([-8.0, -7.0, -6.0]),
        log10_sigma_p_spec=jnp.array([-15.0, -14.0, -13.0]),
        efac_spec=jnp.array([1.0, 1.1, 1.2]),
        equad_spec=jnp.array([1e-8, 1e-7, 1e-6])
    )
    
    # Test that the model yields the correct number of parameters
    params = list(model)
    assert len(params) == 6  # log10_ha, γa, log10_γp, log10_σp, efac, equad

def test_gw_prior_model():
    # Test that the model yields the correct number of parameters
    model = gw_prior_model()
    params = list(model)
    assert len(params) == 6  # log10_ha, γa, log10_γp, log10_σp, efac, equad

def test_null_prior_model():
    Npsr = 3
    efac_array = jnp.array([1.0, 1.1, 1.2])
    equad_array = jnp.array([1e-8, 1e-7, 1e-6])
    
    # Test that the model yields the correct number of parameters
    model = null_prior_model(Npsr, efac_array, equad_array)
    params = list(model)
    assert len(params) == 6  # log10_ha, γa, log10_γp, log10_σp, efac, equad

# Test likelihood function
def test_jaxns_log_likelihood():
    # Create a mock KalmanFilter class for testing
    class MockKF:
        def get_likelihood(self, params):
            return -0.5 * jnp.sum(params.γp**2 + params.σp**2)
    
    # Create test parameters
    Npsr = 3
    log10_ha = -15.0
    γa = 1e-9
    log10_γp = jnp.array([-8.0, -7.0, -6.0])
    log10_σp = jnp.array([-15.0, -14.0, -13.0])
    efac = jnp.array([1.0, 1.1, 1.2])
    equad = jnp.array([1e-8, 1e-7, 1e-6])
    
    # Create mock KF instance
    mock_kf = MockKF()
    
    # Test likelihood calculation
    log_likelihood = jaxns_log_likelihood(
        mock_kf, log10_ha, γa, log10_γp, log10_σp, efac, equad
    )
    
    # Verify that the likelihood is a scalar
    assert isinstance(log_likelihood, (float, jnp.ndarray))
    assert log_likelihood.shape == ()

# Test utility function
def test_print_parameters(capsys):
    # Create test parameters
    params = Parameters(
        γa=1e-9,
        ha=1e-15,
        γp=jnp.array([1e-8, 1e-7, 1e-6]),
        σp=jnp.array([1e-15, 1e-14, 1e-13]),
        EFAC=jnp.array([1.0, 1.1, 1.2]),
        EQUAD=jnp.array([1e-8, 1e-7, 1e-6])
    )
    
    # Call print_parameters
    print_parameters(params)
    
    # Capture the output
    captured = capsys.readouterr()
    
    # Verify that all fields are printed
    assert "γa:" in captured.out
    assert "ha:" in captured.out
    assert "γp:" in captured.out
    assert "σp:" in captured.out
    assert "EFAC:" in captured.out
    assert "EQUAD:" in captured.out 