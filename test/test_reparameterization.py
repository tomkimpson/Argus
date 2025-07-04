"""Test module for log10_ha reparameterization functionality."""

import pytest
import numpy as np
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from unittest.mock import MagicMock
import configparser

# Add parent directory to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argus.bayesian_inference import get_prior_model_specs, configurable_prior_model, display_prior_summary

tfpd = tfp.distributions


@pytest.fixture
def test_config():
    """Create test configuration."""
    config = configparser.ConfigParser()
    
    # Add sections and values for testing
    config.add_section('PriorModel')
    
    # Test parameters for log10_ha
    config.set('PriorModel', 'log10_ha_fixed', 'false')
    config.set('PriorModel', 'log10_ha_min', '-16.0')
    config.set('PriorModel', 'log10_ha_max', '-14.0')
    
    # Other parameters (fixed for simplicity)
    config.set('PriorModel', 'gamma_a_fixed', 'true')
    config.set('PriorModel', 'gamma_a_value', '1e-9')
    config.set('PriorModel', 'psr_noise_fixed', 'true')
    config.set('PriorModel', 'efac_equad_fixed', 'true')
    
    return config


@pytest.fixture
def test_arrays():
    """Create test arrays."""
    return {
        'n_pulsars': 3,
        'sigma_p_array': jnp.array([1e-15, 2e-15, 3e-15]),
        'gamma_p_array': jnp.array([1e-8, 2e-8, 3e-8]),
        'efac_array': jnp.array([1.0, 1.1, 1.2]),
        'equad_array': jnp.array([1e-7, 2e-7, 3e-7])
    }


def test_transformation_parameters_calculation(test_config, test_arrays):
    """Test that transformation parameters are calculated correctly."""
    prior_specs = get_prior_model_specs(
        test_config, 
        test_arrays['n_pulsars'], 
        test_arrays['sigma_p_array'], 
        test_arrays['gamma_p_array'], 
        test_arrays['efac_array'], 
        test_arrays['equad_array']
    )
    
    # Check that transformation parameters exist
    assert prior_specs['log10_ha_transform_params'] is not None
    
    transform_params = prior_specs['log10_ha_transform_params']
    
    # Expected values for U(-16, -14)
    expected_mean = (-16.0 + -14.0) / 2.0  # -15.0
    expected_std = (-14.0 - (-16.0)) / np.sqrt(12.0)  # 2.0 / sqrt(12) ≈ 0.577
    expected_min = -16.0
    expected_max = -14.0
    
    # Test calculated values
    assert abs(transform_params['mean'] - expected_mean) < 1e-6
    assert abs(transform_params['std'] - expected_std) < 1e-6
    assert transform_params['min'] == expected_min
    assert transform_params['max'] == expected_max


def test_fixed_parameter_no_transformation(test_config, test_arrays):
    """Test that fixed parameters don't get transformation parameters."""
    # Modify config for fixed log10_ha
    test_config.set('PriorModel', 'log10_ha_fixed', 'true')
    test_config.set('PriorModel', 'log10_ha_value', '-15.0')
    
    prior_specs = get_prior_model_specs(
        test_config, 
        test_arrays['n_pulsars'], 
        test_arrays['sigma_p_array'],
        test_arrays['gamma_p_array'], 
        test_arrays['efac_array'], 
        test_arrays['equad_array']
    )
    
    # Should not have transformation parameters for fixed values
    assert prior_specs['log10_ha_transform_params'] is None
    # Should have the fixed value
    assert prior_specs['log10_ha_spec'] == -15.0


def test_prior_spec_is_normal_distribution(test_config, test_arrays):
    """Test that log10_ha_spec is N(0,1) when reparameterized."""
    prior_specs = get_prior_model_specs(
        test_config, 
        test_arrays['n_pulsars'], 
        test_arrays['sigma_p_array'],
        test_arrays['gamma_p_array'], 
        test_arrays['efac_array'], 
        test_arrays['equad_array']
    )
    
    # Should be a Normal(0,1) distribution
    ha_spec = prior_specs['log10_ha_spec']
    assert isinstance(ha_spec, tfpd.Normal)
    assert float(ha_spec.loc) == 0.0
    assert float(ha_spec.scale) == 1.0


def test_transformation_mathematical_consistency(test_config, test_arrays):
    """Test that the transformation covers the correct range."""
    prior_specs = get_prior_model_specs(
        test_config, 
        test_arrays['n_pulsars'], 
        test_arrays['sigma_p_array'],
        test_arrays['gamma_p_array'], 
        test_arrays['efac_array'], 
        test_arrays['equad_array']
    )
    
    transform_params = prior_specs['log10_ha_transform_params']
    
    # Test with reasonable extreme values (±4 standard deviations)
    log10_ha_prime_min = -4.0
    log10_ha_prime_max = +4.0
    
    log10_ha_min_mapped = transform_params['mean'] + log10_ha_prime_min * transform_params['std']
    log10_ha_max_mapped = transform_params['mean'] + log10_ha_prime_max * transform_params['std']
    
    # Should be approximately at the bounds (within tolerance due to Normal distribution)
    min_bound = transform_params['min']
    max_bound = transform_params['max']
    
    # The mapping should extend well beyond the original bounds for extreme values
    assert log10_ha_min_mapped < min_bound
    assert log10_ha_max_mapped > max_bound


def test_display_prior_summary_with_reparameterization(test_config, test_arrays):
    """Test that display_prior_summary works with reparameterization."""
    prior_specs = get_prior_model_specs(
        test_config, 
        test_arrays['n_pulsars'], 
        test_arrays['sigma_p_array'],
        test_arrays['gamma_p_array'], 
        test_arrays['efac_array'], 
        test_arrays['equad_array']
    )
    
    # Mock logger to capture output
    mock_logger = MagicMock()
    
    # Should not raise an exception
    display_prior_summary(prior_specs, test_arrays['n_pulsars'], logger=mock_logger)
    
    # Check that logger.info was called (indicating output was generated)
    assert mock_logger.info.called
    
    # Check that reparameterization information is included in output
    all_calls = [str(call) for call in mock_logger.info.call_args_list]
    output_text = ' '.join(all_calls)
    
    assert 'REPARAMETERIZED' in output_text
    assert 'log10_ha_prime' in output_text
    assert 'N(0, 1)' in output_text


def test_prior_specs_return_structure(test_config, test_arrays):
    """Test that get_prior_model_specs returns expected keys."""
    prior_specs = get_prior_model_specs(
        test_config, 
        test_arrays['n_pulsars'], 
        test_arrays['sigma_p_array'],
        test_arrays['gamma_p_array'], 
        test_arrays['efac_array'], 
        test_arrays['equad_array']
    )
    
    expected_keys = {
        'log10_ha_spec',
        'log10_ha_transform_params', 
        'gamma_a_spec',
        'log10_gamma_p_spec',
        'log10_sigma_p_spec',
        'efac_spec',
        'equad_spec'
    }
    
    assert set(prior_specs.keys()) == expected_keys


def test_different_uniform_ranges(test_config, test_arrays):
    """Test reparameterization with different uniform ranges."""
    # Test with different range
    test_config.set('PriorModel', 'log10_ha_min', '-18.0')
    test_config.set('PriorModel', 'log10_ha_max', '-12.0')
    
    prior_specs = get_prior_model_specs(
        test_config, 
        test_arrays['n_pulsars'], 
        test_arrays['sigma_p_array'],
        test_arrays['gamma_p_array'], 
        test_arrays['efac_array'], 
        test_arrays['equad_array']
    )
    
    transform_params = prior_specs['log10_ha_transform_params']
    
    # Expected values for U(-18, -12)
    expected_mean = (-18.0 + -12.0) / 2.0  # -15.0
    expected_std = (-12.0 - (-18.0)) / np.sqrt(12.0)  # 6.0 / sqrt(12) ≈ 1.732
    
    assert abs(transform_params['mean'] - expected_mean) < 1e-6
    assert abs(transform_params['std'] - expected_std) < 1e-6
    assert transform_params['min'] == -18.0
    assert transform_params['max'] == -12.0


def test_simple_math_verification():
    """Simple test to verify our transformation math is correct."""
    # Test case: U(-16, -14)
    a, b = -16.0, -14.0
    
    # Calculate using our formulas
    mean = (a + b) / 2.0
    std = (b - a) / np.sqrt(12.0)
    
    # Expected values
    assert mean == -15.0
    assert abs(std - 0.5773502691896257) < 1e-10  # 2/sqrt(12)
    
    # Test that ±3 sigma covers most of the range
    range_3sigma = 6 * std  # ±3 sigma = 6 * std total range
    original_range = b - a  # 2.0
    
    # 3-sigma range should be larger than original range (normal extends beyond uniform bounds)
    assert range_3sigma > original_range


def test_gradient_filtering_logic(test_config, test_arrays):
    """Test that gradient filtering correctly identifies sampled vs fixed parameters."""
    from argus.inference_runners import calculate_and_display_gradients
    from unittest.mock import MagicMock
    
    # Get prior specs for a case where only log10_ha is sampled
    prior_specs = get_prior_model_specs(
        test_config, 
        test_arrays['n_pulsars'], 
        test_arrays['sigma_p_array'],
        test_arrays['gamma_p_array'], 
        test_arrays['efac_array'], 
        test_arrays['equad_array']
    )
    
    # Check parameter sampling status
    log10_ha_sampled = (prior_specs['log10_ha_transform_params'] is not None or 
                       isinstance(prior_specs['log10_ha_spec'], tfpd.Distribution))
    gamma_a_sampled = isinstance(prior_specs['gamma_a_spec'], tfpd.Distribution)
    psr_noise_sampled = isinstance(prior_specs['log10_gamma_p_spec'], tfpd.Distribution)
    efac_equad_sampled = isinstance(prior_specs['efac_spec'], tfpd.Distribution)
    
    # Based on test config: only log10_ha should be sampled
    assert log10_ha_sampled == True  # log10_ha_fixed = false
    assert gamma_a_sampled == False  # gamma_a_fixed = true  
    assert psr_noise_sampled == False  # psr_noise_fixed = true
    assert efac_equad_sampled == False  # efac_equad_fixed = true


def test_all_parameters_fixed_case(test_arrays):
    """Test gradient display when all parameters are fixed."""
    from unittest.mock import MagicMock
    import configparser
    
    # Create config where all parameters are fixed
    config = configparser.ConfigParser()
    config.add_section('PriorModel')
    
    # All parameters fixed
    config.set('PriorModel', 'log10_ha_fixed', 'true')
    config.set('PriorModel', 'log10_ha_value', '-15.0')
    config.set('PriorModel', 'gamma_a_fixed', 'true')
    config.set('PriorModel', 'gamma_a_value', '1e-9')
    config.set('PriorModel', 'psr_noise_fixed', 'true')
    config.set('PriorModel', 'efac_equad_fixed', 'true')
    
    prior_specs = get_prior_model_specs(
        config, 
        test_arrays['n_pulsars'], 
        test_arrays['sigma_p_array'],
        test_arrays['gamma_p_array'], 
        test_arrays['efac_array'], 
        test_arrays['equad_array']
    )
    
    # Check that no parameters are being sampled
    log10_ha_sampled = (prior_specs['log10_ha_transform_params'] is not None or 
                       isinstance(prior_specs['log10_ha_spec'], tfpd.Distribution))
    gamma_a_sampled = isinstance(prior_specs['gamma_a_spec'], tfpd.Distribution)
    psr_noise_sampled = isinstance(prior_specs['log10_gamma_p_spec'], tfpd.Distribution)
    efac_equad_sampled = isinstance(prior_specs['efac_spec'], tfpd.Distribution)
    
    # All should be False (fixed)
    assert not log10_ha_sampled
    assert not gamma_a_sampled
    assert not psr_noise_sampled  
    assert not efac_equad_sampled