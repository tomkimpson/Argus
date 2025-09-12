"""Parameter sampling functions for NumPyro models.

This module provides functions for sampling parameters from their priors
in NumPyro models, including support for hierarchical modeling and 
reparameterization techniques for improved NUTS sampling.
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import tensorflow_probability.substrates.jax as tfp

tfpd = tfp.distributions


def sample_gw_parameters(prior_specs):
    """Sample gravitational wave parameters from their priors.
    
    Parameters
    ----------
    prior_specs : dict
        Prior distributions dictionary
        
    Returns
    -------
    tuple
        (log10_ha, log10_gamma_a, γa) values
    """
    # Handle reparameterization for log10_ha if needed
    if prior_specs['log10_ha_transform_params'] is not None:
        # Sample log10_ha_prime ~ N(0,1) and transform to log10_ha
        transform_params = prior_specs['log10_ha_transform_params']
        log10_ha_prime = numpyro.sample("log10_ha_prime", dist.Normal(0.0, 1.0))
        log10_ha = numpyro.deterministic("log10_ha", 
            transform_params['mean'] + log10_ha_prime * transform_params['std'])
    elif isinstance(prior_specs['log10_ha_spec'], tfpd.Distribution):
        # Check if it's a uniform distribution (backward compatibility)
        if hasattr(prior_specs['log10_ha_spec'], 'low'):
            log10_ha = numpyro.sample("log10_ha", 
                dist.Uniform(prior_specs['log10_ha_spec'].low, prior_specs['log10_ha_spec'].high))
        else:
            # Other distribution types
            raise NotImplementedError(f"Distribution type {type(prior_specs['log10_ha_spec'])} not implemented")
    else:
        # Fixed value
        log10_ha = numpyro.deterministic("log10_ha", prior_specs['log10_ha_spec'])
    
    if isinstance(prior_specs['log10_gamma_a_spec'], tfpd.Distribution):
        log10_gamma_a = numpyro.sample("log10_gamma_a", 
            dist.Uniform(prior_specs['log10_gamma_a_spec'].low, prior_specs['log10_gamma_a_spec'].high))
        γa = numpyro.deterministic("γa", 10.0 ** log10_gamma_a)
    else:
        log10_gamma_a = numpyro.deterministic("log10_gamma_a", prior_specs['log10_gamma_a_spec'])
        γa = numpyro.deterministic("γa", 10.0 ** log10_gamma_a)
    
    return log10_ha, log10_gamma_a, γa


def sample_hierarchical_gamma_parameters(hierarchical_specs, n_pulsars):
    """Sample hierarchical gamma parameters with gradient balancing.
    
    Parameters
    ----------
    hierarchical_specs : dict
        Hierarchical modeling specifications
    n_pulsars : int
        Number of pulsars
        
    Returns
    -------
    jax.Array
        Sampled log10_γp values
    """
    # Hierarchical modeling for log10_gamma_p with gradient balancing
    log10_gamma_p_mean_raw = numpyro.sample("log10_gamma_p_mean_raw", dist.Normal(0.0, 1.0))
    log10_gamma_p_std_raw = numpyro.sample("log10_gamma_p_std_raw", dist.Normal(0.0, 1.0))
    
    # Transform to appropriate ranges with balanced gradients
    mean_low = hierarchical_specs['log10_gamma_p_mean_spec'].low
    mean_high = hierarchical_specs['log10_gamma_p_mean_spec'].high
    std_low = hierarchical_specs['log10_gamma_p_std_spec'].low
    std_high = hierarchical_specs['log10_gamma_p_std_spec'].high
    
    # Apply gradient-balanced transforms
    log10_gamma_p_mean = numpyro.deterministic("log10_gamma_p_mean", 
        (mean_low + mean_high) / 2.0 + log10_gamma_p_mean_raw * (mean_high - mean_low) / 6.0)
    log10_gamma_p_std = numpyro.deterministic("log10_gamma_p_std", 
        (std_low + std_high) / 2.0 + log10_gamma_p_std_raw * (std_high - std_low) / 6.0)
    
    # Sample individual pulsar parameters with scaled gradients
    log10_γp_raw = numpyro.sample("log10_γp_raw", 
        dist.Normal(jnp.zeros(n_pulsars), jnp.ones(n_pulsars)))
    log10_γp = numpyro.deterministic("log10_γp", 
        log10_gamma_p_mean + log10_γp_raw * log10_gamma_p_std / jnp.sqrt(n_pulsars))
    
    return log10_γp


def sample_uniform_parameters(prior_spec, param_name, n_pulsars):
    """Sample parameters from uniform distribution with improved parameterization.
    
    Parameters
    ----------
    prior_spec : tfpd.Distribution
        Uniform distribution specification
    param_name : str
        Name of the parameter for sampling
    n_pulsars : int
        Number of pulsars
        
    Returns
    -------
    jax.Array
        Sampled parameter values
    """
    # Use improved standardized parameterization with gradient balancing
    low = prior_spec.low
    high = prior_spec.high
    
    mean = (low + high) / 2.0
    std = (high - low) / 6.0  # 3-sigma rule
    
    param_standardized = numpyro.sample(f"{param_name}_standardized", 
        dist.Normal(jnp.zeros(n_pulsars), jnp.ones(n_pulsars) / jnp.sqrt(n_pulsars)))
    param_values = numpyro.deterministic(param_name, 
        mean + param_standardized * std)
    
    return param_values


def sample_log_ratio_parameters(hierarchical_specs, log10_γp, n_pulsars):
    """Sample log-ratio parameters for sigma_p derivation.
    
    Parameters
    ----------
    hierarchical_specs : dict
        Hierarchical modeling specifications
    log10_γp : jax.Array
        Log10 gamma_p values
    n_pulsars : int
        Number of pulsars
        
    Returns
    -------
    jax.Array
        Derived log10_σp values
    """
    # Log-ratio parameterization: σp derived from γp + ratio with gradient balancing
    log10_ratio_mean_raw = numpyro.sample("log10_ratio_mean_raw", dist.Normal(0.0, 1.0))
    log10_ratio_std_raw = numpyro.sample("log10_ratio_std_raw", dist.Normal(0.0, 1.0))
    
    # Transform to appropriate ranges with balanced gradients
    mean_low = hierarchical_specs['log10_ratio_mean_spec'].low
    mean_high = hierarchical_specs['log10_ratio_mean_spec'].high
    std_low = hierarchical_specs['log10_ratio_std_spec'].low
    std_high = hierarchical_specs['log10_ratio_std_spec'].high
    
    # Apply gradient-balanced transforms
    log10_ratio_mean = numpyro.deterministic("log10_ratio_mean", 
        (mean_low + mean_high) / 2.0 + log10_ratio_mean_raw * (mean_high - mean_low) / 6.0)
    log10_ratio_std = numpyro.deterministic("log10_ratio_std", 
        (std_low + std_high) / 2.0 + log10_ratio_std_raw * (std_high - std_low) / 6.0)
    
    # Sample individual ratio parameters with scaled gradients
    log10_ratio_raw = numpyro.sample("log10_ratio_raw", 
        dist.Normal(jnp.zeros(n_pulsars), jnp.ones(n_pulsars)))
    log10_ratio = numpyro.deterministic("log10_ratio", 
        log10_ratio_mean + log10_ratio_raw * log10_ratio_std / jnp.sqrt(n_pulsars))
    
    # Derive log10_σp deterministically from γp + ratio
    log10_σp = numpyro.deterministic("log10_σp", log10_γp + log10_ratio)
    
    return log10_σp


def sample_pulsar_noise_parameters(prior_specs, n_pulsars):
    """Sample pulsar red noise parameters from their priors.
    
    Parameters
    ----------
    prior_specs : dict
        Prior distributions dictionary
    n_pulsars : int
        Number of pulsars
        
    Returns
    -------
    tuple
        (log10_γp, log10_σp) values
    """
    hierarchical_specs = prior_specs.get('hierarchical_specs')
    
    # Handle log10_gamma_p
    if hierarchical_specs and hierarchical_specs.get('hierarchical_noise', False):
        log10_γp = sample_hierarchical_gamma_parameters(hierarchical_specs, n_pulsars)
    elif isinstance(prior_specs['log10_gamma_p_spec'], tfpd.Distribution):
        log10_γp = sample_uniform_parameters(
            prior_specs['log10_gamma_p_spec'], "log10_γp", n_pulsars)
    else:
        log10_γp = numpyro.deterministic("log10_γp", prior_specs['log10_gamma_p_spec'])
    
    # Handle log10_sigma_p - always use log-ratio parameterization when hierarchical_specs exist
    if hierarchical_specs and hierarchical_specs.get('log_ratio_parameterization', False):
        log10_σp = sample_log_ratio_parameters(hierarchical_specs, log10_γp, n_pulsars)
    elif isinstance(prior_specs['log10_sigma_p_spec'], tfpd.Distribution):
        log10_σp = sample_uniform_parameters(
            prior_specs['log10_sigma_p_spec'], "log10_σp", n_pulsars)
    else:
        log10_σp = numpyro.deterministic("log10_σp", prior_specs['log10_sigma_p_spec'])

    return log10_γp, log10_σp


def sample_measurement_noise_parameters(prior_specs, n_pulsars):
    """Sample measurement noise parameters from their priors.
    
    Parameters
    ----------
    prior_specs : dict
        Prior distributions dictionary
    n_pulsars : int
        Number of pulsars
        
    Returns
    -------
    tuple
        (efac, equad) values
    """
    if isinstance(prior_specs['efac_spec'], tfpd.Distribution):
        # Use improved standardized parameterization: sample z ~ N(0,1) and transform
        low = prior_specs['efac_spec'].low
        high = prior_specs['efac_spec'].high
        
        # Improved parameterization: center around expected value with tighter scaling
        # Use 3-sigma rule instead of uniform standard deviation for better convergence
        mean = (low + high) / 2.0
        std = (high - low) / 6.0  # 3-sigma rule: 99.7% of samples within range
        
        efac_standardized = numpyro.sample("efac_standardized", 
                                          dist.Normal(jnp.zeros(n_pulsars), jnp.ones(n_pulsars)))
        efac = numpyro.deterministic("efac", 
                                    mean + efac_standardized * std)
    else:
        efac = numpyro.deterministic("efac", prior_specs['efac_spec'])
    
    if isinstance(prior_specs['equad_spec'], dict) and prior_specs['equad_spec'].get('use_log10', False):
        # log10(EQUAD) parameterization - follow same pattern as log10_gamma_a
        log10_equad_spec = prior_specs['equad_spec']['log10_equad_spec']
        log10_equad = numpyro.sample("log10_equad", 
            dist.Uniform(log10_equad_spec.low, log10_equad_spec.high))
        equad = numpyro.deterministic("equad", 10.0 ** log10_equad)
    elif isinstance(prior_specs['equad_spec'], tfpd.Distribution):
        # Regular uniform distribution - use improved standardized parameterization
        low = prior_specs['equad_spec'].low
        high = prior_specs['equad_spec'].high
        
        # Improved parameterization: center around expected value with tighter scaling
        # Use 3-sigma rule instead of uniform standard deviation for better convergence
        mean = (low + high) / 2.0
        std = (high - low) / 6.0  # 3-sigma rule: 99.7% of samples within range
        
        equad_standardized = numpyro.sample("equad_standardized", 
                                           dist.Normal(jnp.zeros(n_pulsars), jnp.ones(n_pulsars)))
        equad = numpyro.deterministic("equad", 
                                     mean + equad_standardized * std)
    else:
        equad = numpyro.deterministic("equad", prior_specs['equad_spec'])
    
    return efac, equad


def count_free_parameters(prior_specs, n_pulsars):
    """Count the total number of free (non-fixed) parameters for NUTS sampling.
    
    Parameters
    ----------
    prior_specs : dict
        Prior distributions dictionary
    n_pulsars : int
        Number of pulsars
        
    Returns
    -------
    int
        Total number of free parameters
    """
    import tensorflow_probability.substrates.jax as tfp
    tfpd = tfp.distributions
    
    count = 0
    
    # GW amplitude parameter
    if (prior_specs['log10_ha_transform_params'] is not None or 
        isinstance(prior_specs['log10_ha_spec'], tfpd.Distribution)):
        count += 1
    
    # GW spectral index parameter  
    if isinstance(prior_specs['log10_gamma_a_spec'], tfpd.Distribution):
        count += 1
    
    # Pulsar red noise parameters
    hierarchical_specs = prior_specs.get('hierarchical_specs')
    if hierarchical_specs and hierarchical_specs.get('hierarchical_noise', False):
        # Hierarchical modeling: 2 hyperparameters + n_pulsars individual parameters
        count += 2  # log10_gamma_p_mean and log10_gamma_p_std
        count += n_pulsars  # Individual pulsar gamma parameters
    elif isinstance(prior_specs['log10_gamma_p_spec'], tfpd.Distribution):
        count += n_pulsars  # One per pulsar
    
    if hierarchical_specs and hierarchical_specs.get('log_ratio_parameterization', False):
        # Log-ratio parameterization: 2 hyperparameters + n_pulsars ratio parameters
        count += 2  # log10_ratio_mean and log10_ratio_std
        count += n_pulsars  # Individual pulsar ratio parameters (σp derived deterministically)
    elif isinstance(prior_specs['log10_sigma_p_spec'], tfpd.Distribution):
        count += n_pulsars  # One per pulsar
    
    # Measurement noise parameters
    if isinstance(prior_specs['efac_spec'], tfpd.Distribution):
        count += n_pulsars  # One per pulsar
    if isinstance(prior_specs['equad_spec'], tfpd.Distribution):
        count += n_pulsars  # One per pulsar
    
    return count