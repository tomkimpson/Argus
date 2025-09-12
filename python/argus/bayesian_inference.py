"""Bayesian inference module for pulsar timing array analysis.

This module provides functionality for performing Bayesian parameter estimation
on pulsar timing array data using NumPyro NUTS sampling. It includes:

- Prior definitions for gravitational wave background and pulsar noise parameters
- Likelihood calculations using a Kalman filter implementation
- NUTS sampling routines using NumPyro

The module is designed to work with the JAX framework for automatic differentiation
and GPU acceleration. It handles parameters like:

- Gravitational wave background amplitude (ha) and spectral index (γa)
- Pulsar-specific red noise parameters (γp, σp)
- White noise parameters (EFAC, EQUAD)

The implementation uses the Hellings-Downs correlation pattern for the 
gravitational wave background and models pulsar red noise as an 
Ornstein-Uhlenbeck process.

Uses NumPyro for JAX-native NUTS sampling with automatic differentiation.
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import arviz as az
import tensorflow_probability.substrates.jax as tfp
from flax import struct
from numpyro.infer import MCMC, NUTS
import time

jax.config.update("jax_enable_x64", True)
tfpd = tfp.distributions



@struct.dataclass
class Parameters:
    """Define a struct to store the parameters of the Kalman filter model."""
    
    #GW parameters
    log10_gamma_a: float  # log10(γa) - log10 of GW spectral index
    γa: float  # s⁻¹ - GW spectral index (derived from log10_gamma_a)
    ha: float  # GWB amplitude

    #Pulsar parameters for the OU process
    γp: jnp.ndarray  # Pulsar-specific gamma values
    σp: jnp.ndarray  # Pulsar-specific sigma values 

    #Measurement noise parameters
    EFAC: jnp.ndarray  # Error factors
    EQUAD: jnp.ndarray # Extra quadrature noise


def log_likelihood_fn(KF, log10_ha, log10_gamma_a, log10_γp, log10_σp, efac, equad):
    """Calculate log likelihood for NumPyro sampling."""
    ha = 10.0 ** log10_ha
    γa = 10.0 ** log10_gamma_a
    γp = 10.0 ** log10_γp
    σp = 10.0 ** log10_σp

    params = Parameters(
        log10_gamma_a=log10_gamma_a,
        γa=γa,
        ha=ha,
        γp=γp,
        σp=σp,
        EFAC=efac,
        EQUAD=equad
    )

    return KF.get_likelihood(params)



def _create_hierarchical_priors(config, hierarchical_noise, log_ratio_parameterization):
    """
    Create hierarchical modeling prior distributions if needed.
    
    Parameters
    ----------
    config : ConfigParser
        Configuration object
    hierarchical_noise : bool
        Whether to use hierarchical noise modeling
    log_ratio_parameterization : bool
        Whether to use log-ratio parameterization
        
    Returns
    -------
    dict or None
        Hierarchical prior distributions dictionary or None if not needed
    """
    if not (hierarchical_noise or log_ratio_parameterization):
        return None
        
    hierarchical_specs = {
        'hierarchical_noise': hierarchical_noise,
        'log_ratio_parameterization': log_ratio_parameterization
    }
    
    if hierarchical_noise:
        hierarchical_specs.update({
            'log10_gamma_p_mean_spec': tfpd.Uniform(
                config.getfloat('PriorModel', 'log10_gamma_p_mean_min'),
                config.getfloat('PriorModel', 'log10_gamma_p_mean_max')
            ),
            'log10_gamma_p_std_spec': tfpd.Uniform(
                config.getfloat('PriorModel', 'log10_gamma_p_std_min'),
                config.getfloat('PriorModel', 'log10_gamma_p_std_max')
            )
        })
    
    if log_ratio_parameterization:
        hierarchical_specs.update({
            'log10_ratio_mean_spec': tfpd.Uniform(
                config.getfloat('PriorModel', 'log10_ratio_mean_min'),
                config.getfloat('PriorModel', 'log10_ratio_mean_max')
            ),
            'log10_ratio_std_spec': tfpd.Uniform(
                config.getfloat('PriorModel', 'log10_ratio_std_min'),
                config.getfloat('PriorModel', 'log10_ratio_std_max')
            )
        })
    
    return hierarchical_specs


def _get_gw_parameter_priors(config):
    """
    Extract gravitational wave parameter prior distributions from config.
    
    Parameters
    ----------
    config : ConfigParser
        Configuration object containing prior model settings
        
    Returns
    -------
    dict
        Dictionary containing GW parameter prior distributions:
        - log10_ha_spec: Prior distribution for log10(ha)
        - log10_ha_transform_params: Transformation parameters for reparameterization
        - log10_gamma_a_spec: Prior distribution for log10(γa)
    """
    # Helper function to create prior spec based on fixed/sampled setting
    def get_prior_spec(param_name):
        is_fixed = config.getboolean('PriorModel', f'{param_name}_fixed')
        if is_fixed:
            return config.getfloat('PriorModel', f'{param_name}_value')
        else:
            min_val = config.getfloat('PriorModel', f'{param_name}_min')
            max_val = config.getfloat('PriorModel', f'{param_name}_max')
            return tfpd.Uniform(min_val, max_val)

    # Handle log10_ha with reparameterization for better NUTS sampling
    log10_ha_fixed = config.getboolean('PriorModel', 'log10_ha_fixed')
    if log10_ha_fixed:
        # Fixed value - no reparameterization needed
        log10_ha_spec = config.getfloat('PriorModel', 'log10_ha_value')
        log10_ha_transform_params = None
    else:
        # Reparameterize U(a,b) -> N(0,1) for better NUTS sampling
        min_val = config.getfloat('PriorModel', 'log10_ha_min')
        max_val = config.getfloat('PriorModel', 'log10_ha_max')
        
        # Calculate improved transformation parameters: log10_ha = mean + log10_ha_prime * std
        # Use 3-sigma rule for better convergence
        mean = (min_val + max_val) / 2.0
        std = (max_val - min_val) / 6.0  # 3-sigma rule: 99.7% of samples within range
        
        # Use N(0,1) for log10_ha_prime, store transformation parameters
        log10_ha_spec = tfpd.Normal(0.0, 1.0)  # log10_ha_prime ~ N(0,1)
        log10_ha_transform_params = {'mean': mean, 'std': std, 'min': min_val, 'max': max_val}
    
    log10_gamma_a_spec = get_prior_spec('log10_gamma_a')

    return {
        'log10_ha_spec': log10_ha_spec,
        'log10_ha_transform_params': log10_ha_transform_params,
        'log10_gamma_a_spec': log10_gamma_a_spec
    }


def _get_pulsar_noise_priors(config, Npsr, sigma_p_array, gamma_p_array):
    """
    Extract pulsar red noise parameter prior distributions from config.
    
    Parameters
    ----------
    config : ConfigParser
        Configuration object containing prior model settings
    Npsr : int
        Number of pulsars
    sigma_p_array : array
        Array of pulsar red noise sigma values
    gamma_p_array : array
        Array of pulsar red noise gamma values
        
    Returns
    -------
    dict
        Dictionary containing pulsar noise parameter prior distributions:
        - log10_gamma_p_spec: Prior distribution for log10(γp)
        - log10_sigma_p_spec: Prior distribution for log10(σp) 
        - hierarchical_specs: Hierarchical modeling prior distributions
    """
    # Check if spin_injections_path is provided to determine if red noise parameters should be fixed
    try:
        spin_injections_path = config.get('PriorModel', 'spin_injections_path')
        # If path is provided and not empty, fix red noise parameters
        log10_gamma_p_fixed = bool(spin_injections_path.strip())
        log10_sigma_p_fixed = bool(spin_injections_path.strip())
        print(f"Red noise parameters fixed via spin_injections_path: {log10_gamma_p_fixed}")
    except:
        # If no spin_injections_path, sample from priors
        log10_gamma_p_fixed = False
        log10_sigma_p_fixed = False
        print("No spin_injections_path provided, sampling red noise parameters from priors")
    
    # Check for hierarchical modeling and log-ratio parameterization
    hierarchical_noise = config.getboolean('PriorModel', 'hierarchical_noise', fallback=False)
    log_ratio_parameterization = config.getboolean('PriorModel', 'log_ratio_parameterization', fallback=False)
    hierarchical_specs = _create_hierarchical_priors(config, hierarchical_noise, log_ratio_parameterization)
    
    # Handle gamma_p specification
    if log10_gamma_p_fixed:
        if config.has_option('PriorModel', 'log10_gamma_p_value'):
            # Check if value is a string (for 'injected'/'default') or a number
            gamma_p_value_str = config.get('PriorModel', 'log10_gamma_p_value')
            if gamma_p_value_str.lower() in ['injected', 'default']:
                # Use injected values
                log10_gamma_p_spec = jnp.log10(gamma_p_array)
                print(f"Using injected gamma_p values: {gamma_p_value_str}")
            else:
                # Use explicit fixed value from config
                gamma_p_fixed_value = config.getfloat('PriorModel', 'log10_gamma_p_value')
                log10_gamma_p_spec = jnp.full(Npsr, gamma_p_fixed_value)
                print(f"Using fixed gamma_p value: {gamma_p_fixed_value}")
        else:
            # Use injected values (legacy approach)
            log10_gamma_p_spec = jnp.log10(gamma_p_array)
            print("Using injected gamma_p values (legacy mode)")
    elif hierarchical_noise:
        log10_gamma_p_spec = None  # Will be handled hierarchically
    else:
        # Free gamma_p with uniform prior
        log10_gamma_p_spec = tfpd.Uniform(
            low=jnp.full(Npsr, config.getfloat('PriorModel', 'log10_gamma_p_min')),
            high=jnp.full(Npsr, config.getfloat('PriorModel', 'log10_gamma_p_max'))
        )
    
    # Handle sigma_p specification  
    if log10_sigma_p_fixed:
        if config.has_option('PriorModel', 'log10_sigma_p_value'):
            # Check if value is a string (for 'injected'/'default') or a number
            sigma_p_value_str = config.get('PriorModel', 'log10_sigma_p_value')
            if sigma_p_value_str.lower() in ['injected', 'default']:
                # Use injected values
                log10_sigma_p_spec = jnp.log10(sigma_p_array)
                print(f"Using injected sigma_p values: {sigma_p_value_str}")
            else:
                # Use explicit fixed value from config
                sigma_p_fixed_value = config.getfloat('PriorModel', 'log10_sigma_p_value')
                log10_sigma_p_spec = jnp.full(Npsr, sigma_p_fixed_value)
                print(f"Using fixed sigma_p value: {sigma_p_fixed_value}")
        else:
            # Use injected values (legacy approach)
            log10_sigma_p_spec = jnp.log10(sigma_p_array)
            print("Using injected sigma_p values (legacy mode)")
    elif log_ratio_parameterization:
        log10_sigma_p_spec = None  # Will be derived from log-ratio
    else:
        # Free sigma_p with uniform prior
        log10_sigma_p_spec = tfpd.Uniform(
            low=jnp.full(Npsr, config.getfloat('PriorModel', 'log10_sigma_p_min')),
            high=jnp.full(Npsr, config.getfloat('PriorModel', 'log10_sigma_p_max'))
        )

    return {
        'log10_gamma_p_spec': log10_gamma_p_spec,
        'log10_sigma_p_spec': log10_sigma_p_spec,
        'hierarchical_specs': hierarchical_specs
    }


def _get_measurement_noise_priors(config, efac_array, equad_array):
    """
    Extract measurement noise parameter prior distributions from config.
    
    Parameters
    ----------
    config : ConfigParser
        Configuration object containing prior model settings
    efac_array : array
        Array of EFAC values
    equad_array : array
        Array of EQUAD values
        
    Returns
    -------
    dict
        Dictionary containing measurement noise parameter prior distributions:
        - efac_spec: Prior distribution for EFAC
        - equad_spec: Prior distribution for EQUAD
    """
    # Check if noise_params_path is provided to determine if EFAC/EQUAD should be fixed
    try:
        noise_params_path = config.get('PriorModel', 'noise_params_path')
        # If path is provided and not empty, fix EFAC/EQUAD parameters
        efac_equad_fixed = bool(noise_params_path.strip())
        print(f"EFAC/EQUAD parameters fixed via noise_params_path: {efac_equad_fixed}")
    except:
        # If no noise_params_path, sample from priors
        efac_equad_fixed = False
        print("No noise_params_path provided, sampling EFAC/EQUAD from priors")
    
    if efac_equad_fixed:
        efac_spec = efac_array
        equad_spec = equad_array
    else:
        efac_spec = tfpd.Uniform(
            low=jnp.full_like(efac_array, config.getfloat('PriorModel', 'efac_min')),
            high=jnp.full_like(efac_array, config.getfloat('PriorModel', 'efac_max'))
        )
        
        # Use log10(EQUAD) uniform prior - transformation handled in numpyro model
        log10_equad_spec = tfpd.Uniform(
            low=jnp.full_like(equad_array, config.getfloat('PriorModel', 'log10_equad_min')),
            high=jnp.full_like(equad_array, config.getfloat('PriorModel', 'log10_equad_max'))
        )
        equad_spec = {'log10_equad_spec': log10_equad_spec, 'use_log10': True}

    return {
        'efac_spec': efac_spec,
        'equad_spec': equad_spec
    }


def get_prior_model_specs(config, Npsr, sigma_p_array, gamma_p_array, efac_array, equad_array):
    """
    Create prior model distributions based on config settings.
    
    Args:
        config: ConfigParser object containing prior model settings
        Npsr: Number of pulsars
        sigma_p_array: Array of pulsar red noise sigma values, only used if psr_noise_fixed=true in config
        gamma_p_array: Array of pulsar red noise gamma values, only used if psr_noise_fixed=true in config
        efac_array: Array of EFAC values, only used if efac_equad_fixed=true in config
        equad_array: Array of EQUAD values, only used if efac_equad_fixed=true in config
        
    Returns
    -------
        dict: Dictionary containing all prior distributions with the following keys:
            - log10_ha_spec: Prior distribution for log10 of GW amplitude (fixed value or Uniform)
            - log10_gamma_a_spec: Prior distribution for log10(GW spectral index) (fixed value or Uniform)
            - log10_gamma_p_spec: Prior distribution for log10 of pulsar red noise gamma (Uniform)
            - log10_sigma_p_spec: Prior distribution for log10 of pulsar red noise sigma (Uniform)
            - efac_spec: Prior distribution for EFAC (fixed array or Uniform)
            - equad_spec: Prior distribution for EQUAD (fixed array or Uniform)
            
    Note:
        For each parameter, the prior type (fixed or Uniform) is determined by the
        corresponding *_fixed setting in the config file. If fixed=true, the *_value
        is used; if fixed=false, a Uniform distribution is created using *_min and *_max.
        For EFAC and EQUAD, if efac_equad_fixed=true, the provided arrays are used;
        otherwise, Uniform distributions are created using efac_min/max and equad_min/max.
        The same logic applies to the pulsar red noise parameters.
    """
    print("Getting prior model specs...")
    
    # Get parameter prior distributions from specialized functions
    gw_specs = _get_gw_parameter_priors(config)
    pulsar_noise_specs = _get_pulsar_noise_priors(config, Npsr, sigma_p_array, gamma_p_array)
    measurement_noise_specs = _get_measurement_noise_priors(config, efac_array, equad_array)

    return {
        'log10_ha_spec': gw_specs['log10_ha_spec'],
        'log10_ha_transform_params': gw_specs['log10_ha_transform_params'],
        'log10_gamma_a_spec': gw_specs['log10_gamma_a_spec'],
        'log10_gamma_p_spec': pulsar_noise_specs['log10_gamma_p_spec'],
        'log10_sigma_p_spec': pulsar_noise_specs['log10_sigma_p_spec'],
        'efac_spec': measurement_noise_specs['efac_spec'],
        'equad_spec': measurement_noise_specs['equad_spec'],
        'hierarchical_specs': pulsar_noise_specs['hierarchical_specs']
    }



def _sample_gw_parameters(prior_specs):
    """
    Sample gravitational wave parameters from their priors.
    
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


def _sample_pulsar_noise_parameters(prior_specs, n_pulsars):
    """
    Sample pulsar red noise parameters from their priors.
    
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
    elif isinstance(prior_specs['log10_gamma_p_spec'], tfpd.Distribution):
        # Use improved standardized parameterization with gradient balancing
        low = prior_specs['log10_gamma_p_spec'].low
        high = prior_specs['log10_gamma_p_spec'].high
        
        mean = (low + high) / 2.0
        std = (high - low) / 6.0  # 3-sigma rule
        
        log10_γp_standardized = numpyro.sample("log10_γp_standardized", 
            dist.Normal(jnp.zeros(n_pulsars), jnp.ones(n_pulsars) / jnp.sqrt(n_pulsars)))
        log10_γp = numpyro.deterministic("log10_γp", 
            mean + log10_γp_standardized * std)
    else:
        log10_γp = numpyro.deterministic("log10_γp", prior_specs['log10_gamma_p_spec'])
    
    # Handle log10_sigma_p - always use log-ratio parameterization when hierarchical_specs exist
    if hierarchical_specs and hierarchical_specs.get('log_ratio_parameterization', False):
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
    elif isinstance(prior_specs['log10_sigma_p_spec'], tfpd.Distribution):
        # Use improved standardized parameterization with gradient balancing
        low = prior_specs['log10_sigma_p_spec'].low
        high = prior_specs['log10_sigma_p_spec'].high
        
        mean = (low + high) / 2.0
        std = (high - low) / 6.0  # 3-sigma rule
        
        log10_σp_standardized = numpyro.sample("log10_σp_standardized", 
            dist.Normal(jnp.zeros(n_pulsars), jnp.ones(n_pulsars) / jnp.sqrt(n_pulsars)))
        log10_σp = numpyro.deterministic("log10_σp", 
            mean + log10_σp_standardized * std)
    else:
        log10_σp = numpyro.deterministic("log10_σp", prior_specs['log10_sigma_p_spec'])

    return log10_γp, log10_σp


def _sample_measurement_noise_parameters(prior_specs, n_pulsars):
    """
    Sample measurement noise parameters from their priors.
    
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


def numpyro_model(kalman_filter, prior_specs, n_pulsars):
    """NumPyro model definition for Bayesian inference with parameter standardization.
    
    This function defines the NumPyro probabilistic model using standardized
    parameter transformations for better NUTS sampling in high-dimensional spaces.
    
    Parameters
    ----------
    kalman_filter : object
        JAX Kalman filter with get_likelihood method
    prior_specs : dict
        Dictionary containing prior distributions from get_prior_model_specs()
    n_pulsars : int
        Number of pulsars
    """
    # Sample parameters using specialized functions
    log10_ha, log10_gamma_a, γa = _sample_gw_parameters(prior_specs)
    log10_γp, log10_σp = _sample_pulsar_noise_parameters(prior_specs, n_pulsars)
    efac, equad = _sample_measurement_noise_parameters(prior_specs, n_pulsars)
    
    # Calculate log likelihood
    log_likelihood = log_likelihood_fn(kalman_filter, log10_ha, log10_gamma_a, log10_γp, log10_σp, efac, equad)
    
    # Add likelihood to the model
    numpyro.factor("likelihood", log_likelihood)


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


def run_nuts_sampling(kalman_filter, config, n_pulsars, sigma_p_array, gamma_p_array, 
                      efac_array, equad_array):
    """Run NumPyro NUTS inference with optimizations for high-dimensional sampling.
    
    Parameters
    ----------
    kalman_filter : object
        JAX Kalman filter with get_likelihood method
    config : configparser.ConfigParser
        Configuration object
    n_pulsars : int
        Number of pulsars
    sigma_p_array : jnp.ndarray
        Pulsar red noise sigma values
    gamma_p_array : jnp.ndarray
        Pulsar red noise gamma values
    efac_array : jnp.ndarray
        EFAC values
    equad_array : jnp.ndarray
        EQUAD values
        
    Returns
    -------
    inf_data : arviz.InferenceData
        ArviZ InferenceData object containing MCMC results
    """
    import jax.random as random
    
    # Get prior model distributions
    prior_specs = get_prior_model_specs(
        config, n_pulsars, sigma_p_array, gamma_p_array, efac_array, equad_array
    )
    
    # Get NUTS parameters from config with optimized defaults for high-dimensional sampling
    num_samples = config.getint('NUTS', 'num_samples', fallback=2000)
    num_warmup = config.getint('NUTS', 'num_warmup', fallback=2000)
    num_chains = config.getint('NUTS', 'num_chains', fallback=2)
    target_accept_prob = config.getfloat('NUTS', 'target_accept_prob', fallback=0.95)  # More conservative for high-dim
    
    # Additional NUTS tuning parameters
    max_tree_depth = config.getint('NUTS', 'max_tree_depth', fallback=10)
    dense_mass = config.getboolean('NUTS', 'dense_mass', fallback=False)
    
    # Handle step_size - only set if explicitly provided in config
    nuts_kwargs = {
        'target_accept_prob': target_accept_prob,
        'max_tree_depth': max_tree_depth,
        'adapt_step_size': True,
        'adapt_mass_matrix': True,
        'dense_mass': dense_mass  # Use dense mass matrix if specified in config
    }
    
    # Only add step_size if explicitly set in config
    if config.has_option('NUTS', 'step_size'):
        step_size = config.getfloat('NUTS', 'step_size')
        nuts_kwargs['step_size'] = step_size
        print(f"Using custom step size: {step_size}")
    
    print(f"Running NumPyro NUTS inference with {n_pulsars} pulsars...")
    print(f"NUTS parameters: {num_samples} samples, {num_warmup} warmup, {num_chains} chains")
    print(f"Target accept prob: {target_accept_prob} (optimized for high-dimensional sampling)")
    print(f"Dense mass matrix: {dense_mass}")
    print(f"Max tree depth: {max_tree_depth}")
    
    # Count total number of free parameters for diagnostics
    total_params = count_free_parameters(prior_specs, n_pulsars)
    print(f"Total free parameters: {total_params}")
    
    # Check if hierarchical modeling is enabled
    hierarchical_specs = prior_specs.get('hierarchical_specs')
    if hierarchical_specs:
        hier_gamma = hierarchical_specs.get('hierarchical_noise', False)
        log_ratio = hierarchical_specs.get('log_ratio_parameterization', False)
        if hier_gamma or log_ratio:
            print("Advanced modeling enabled for pulsar noise parameters")
            if hier_gamma and log_ratio:
                print(f"γp hierarchical + σp via log-ratio parameterization")
                print(f"Effective dimensionality: 4 hyperparameters + {2*n_pulsars} constrained parameters")
                print(f"σp = γp + ratio (reduces parameter correlations)")
            elif hier_gamma:
                print(f"γp uses hierarchical priors, σp fixed")
                print(f"Effective dimensionality: 2 hyperparameters + {n_pulsars} constrained parameters")
            elif hier_sigma:
                print(f"σp uses hierarchical priors, γp independent")
                print(f"Effective dimensionality: 2 hyperparameters + {n_pulsars} constrained + {n_pulsars} independent parameters")
            elif log_ratio:
                print(f"σp via log-ratio parameterization, γp independent")
                print(f"Effective dimensionality: 2 hyperparameters + {2*n_pulsars} parameters")
    
    if total_params > 10:
        print("High-dimensional parameter space detected - using aggressive NUTS tuning")
    
    # Set up NUTS kernel with optimizations
    kernel = NUTS(
        lambda: numpyro_model(kalman_filter, prior_specs, n_pulsars),
        **nuts_kwargs
    )
    
    # Set up MCMC sampler
    sampler = MCMC(
        kernel, 
        num_samples=num_samples, 
        num_warmup=num_warmup,
        num_chains=num_chains,
        progress_bar=True
    )
    
    # Run sampling
    rng_key = random.PRNGKey(42)  # Fixed seed for reproducibility
    sampler.run(rng_key)
    
    # Print summary
    sampler.print_summary()
    
    # Convert to ArviZ format
    inf_data = az.from_numpyro(sampler)
    
    return inf_data


def display_prior_summary(prior_specs, n_pulsars, logger=None):
    """Display a readable summary of all prior distributions.
    
    Parameters
    ----------
    prior_specs : dict
        Dictionary containing prior distributions from get_prior_model_specs()
    n_pulsars : int
        Number of pulsars (for vector parameter information)
    logger : logging.Logger, optional
        Logger object for output. If None, gets the centralized argus logger.
    """
    if logger is None:
        from argus.io_manager import get_argus_logger
        logger = get_argus_logger()
    
    def log_or_print(message):
        logger.info(message)
    
    log_or_print("\n" + "="*60)
    log_or_print("PRIOR SPECIFICATIONS SUMMARY")
    log_or_print("="*60)
    
    # GW background parameters
    log_or_print("\n--- Gravitational Wave Background Parameters ---")
    
    # log10_ha parameter
    ha_spec = prior_specs['log10_ha_spec']
    ha_transform = prior_specs['log10_ha_transform_params']
    
    if ha_transform is not None:
        # Reparameterized case
        log_or_print(f"log10(h_a): REPARAMETERIZED for better NUTS sampling")
        log_or_print(f"  - Sampling: log10_ha_prime ~ N(0, 1)")
        log_or_print(f"  - Transform: log10_ha = {ha_transform['mean']:.2f} + log10_ha_prime * {ha_transform['std']:.3f}")
        log_or_print(f"  - Equivalent to: Uniform({ha_transform['min']:.1f}, {ha_transform['max']:.1f})")
    elif isinstance(ha_spec, tfpd.Distribution):
        # Direct distribution case (backward compatibility)
        if hasattr(ha_spec, 'low'):
            log_or_print(f"log10(h_a): Uniform({float(ha_spec.low):.1f}, {float(ha_spec.high):.1f})")
        else:
            log_or_print(f"log10(h_a): {type(ha_spec).__name__} distribution")
    else:
        # Fixed value case
        log_or_print(f"log10(h_a): FIXED at {float(ha_spec):.1f}")
    
    # log10_gamma_a parameter
    log10_gamma_spec = prior_specs['log10_gamma_a_spec']
    if isinstance(log10_gamma_spec, tfpd.Distribution):
        log_or_print(f"log10(γ_a): Uniform({float(log10_gamma_spec.low):.1f}, {float(log10_gamma_spec.high):.1f})")
    else:
        log_or_print(f"log10(γ_a): FIXED at {float(log10_gamma_spec):.1f}")
    
    # Pulsar red noise parameters
    log_or_print(f"\n--- Pulsar Red Noise Parameters ({n_pulsars} pulsars) ---")
    
    # log10_gamma_p parameter - check for hierarchical modeling
    gamma_p_spec = prior_specs['log10_gamma_p_spec']
    hierarchical_specs = prior_specs.get('hierarchical_specs')
    
    if hierarchical_specs and hierarchical_specs.get('hierarchical_noise', False):
        # Hierarchical modeling case
        mean_spec = hierarchical_specs['log10_gamma_p_mean_spec']
        std_spec = hierarchical_specs['log10_gamma_p_std_spec']
        log_or_print(f"log10(γ_p): HIERARCHICAL modeling")
        log_or_print(f"  - Population mean: Uniform({float(mean_spec.low):.1f}, {float(mean_spec.high):.1f})")
        log_or_print(f"  - Population std: Uniform({float(std_spec.low):.1f}, {float(std_spec.high):.1f})")
        log_or_print(f"  - Individual pulsars: Normal(population_mean, population_std)")
    elif isinstance(gamma_p_spec, tfpd.Distribution):
        log_or_print(f"log10(γ_p): Uniform({float(gamma_p_spec.low[0]):.1f}, {float(gamma_p_spec.high[0]):.1f}) for each pulsar")
    elif gamma_p_spec is not None:
        if hasattr(gamma_p_spec, '__len__') and len(gamma_p_spec) > 1:
            log_or_print(f"log10(γ_p): FIXED at individual values (range: {float(jnp.min(gamma_p_spec)):.2f} to {float(jnp.max(gamma_p_spec)):.2f})")
        else:
            log_or_print(f"log10(γ_p): FIXED at {float(gamma_p_spec):.2f}")
    else:
        log_or_print(f"log10(γ_p): ERROR - None value encountered")
    
    # log10_sigma_p parameter - check for hierarchical modeling
    sigma_p_spec = prior_specs['log10_sigma_p_spec']
    if hierarchical_specs and hierarchical_specs.get('log_ratio_parameterization', False):
        # Check if the required specs exist before accessing them
        if 'log10_ratio_mean_spec' in hierarchical_specs and 'log10_ratio_std_spec' in hierarchical_specs:
            # Log-ratio parameterization case
            mean_spec = hierarchical_specs['log10_ratio_mean_spec']
            std_spec = hierarchical_specs['log10_ratio_std_spec']
            log_or_print(f"log10(σ_p): LOG-RATIO parameterization")
            log_or_print(f"  - log10(σ_p) = log10(γ_p) + log10(ratio)")
            log_or_print(f"  - Ratio mean: Uniform({float(mean_spec.low):.1f}, {float(mean_spec.high):.1f})")
            log_or_print(f"  - Ratio std: Uniform({float(std_spec.low):.1f}, {float(std_spec.high):.1f})")
            log_or_print(f"  - Individual ratios: Normal(ratio_mean, ratio_std)")
        else:
            # Fallback: hierarchical settings enabled but specs not created (likely due to fixed params)
            log_or_print(f"log10(σ_p): FIXED (hierarchical settings detected but overridden by fixed parameters)")
    elif isinstance(sigma_p_spec, tfpd.Distribution):
        log_or_print(f"log10(σ_p): Uniform({float(sigma_p_spec.low[0]):.1f}, {float(sigma_p_spec.high[0]):.1f}) for each pulsar")
    elif sigma_p_spec is not None:
        if hasattr(sigma_p_spec, '__len__') and len(sigma_p_spec) > 1:
            log_or_print(f"log10(σ_p): FIXED at individual values (range: {float(jnp.min(sigma_p_spec)):.2f} to {float(jnp.max(sigma_p_spec)):.2f})")
        else:
            log_or_print(f"log10(σ_p): FIXED at {float(sigma_p_spec):.2f}")
    else:
        log_or_print(f"log10(σ_p): ERROR - None value encountered")
    
    # Measurement noise parameters
    log_or_print(f"\n--- Measurement Noise Parameters ({n_pulsars} pulsars) ---")
    
    # EFAC parameter
    efac_spec = prior_specs['efac_spec']
    if isinstance(efac_spec, tfpd.Distribution):
        log_or_print(f"EFAC: Uniform({float(efac_spec.low[0]):.2f}, {float(efac_spec.high[0]):.2f}) for each pulsar")
    elif efac_spec is not None:
        if hasattr(efac_spec, '__len__') and len(efac_spec) > 1:
            log_or_print(f"EFAC: FIXED at individual values (range: {float(jnp.min(efac_spec)):.3f} to {float(jnp.max(efac_spec)):.3f})")
        else:
            log_or_print(f"EFAC: FIXED at {float(efac_spec):.3f}")
    else:
        log_or_print(f"EFAC: ERROR - None value encountered")
    
    # EQUAD parameter
    equad_spec = prior_specs['equad_spec']
    if isinstance(equad_spec, dict) and equad_spec.get('use_log10', False):
        # log10(EQUAD) parameterization
        log10_equad_spec = equad_spec['log10_equad_spec']
        log10_low = float(log10_equad_spec.low[0])
        log10_high = float(log10_equad_spec.high[0])
        log_or_print(f"EQUAD: log10(EQUAD) ~ Uniform({log10_low:.1f}, {log10_high:.1f}) for each pulsar")
    elif isinstance(equad_spec, tfpd.Distribution):
        # Regular uniform distribution
        log_or_print(f"EQUAD: Uniform({float(equad_spec.low[0]):.2e}, {float(equad_spec.high[0]):.2e}) for each pulsar")
    elif equad_spec is not None:
        if hasattr(equad_spec, '__len__') and len(equad_spec) > 1:
            log_or_print(f"EQUAD: FIXED at individual values (range: {float(jnp.min(equad_spec)):.2e} to {float(jnp.max(equad_spec)):.2e})")
        else:
            log_or_print(f"EQUAD: FIXED at {float(equad_spec):.2e}")
    else:
        log_or_print(f"EQUAD: ERROR - None value encountered")
    
    log_or_print("="*60)


def test_likelihood_performance(KF, config, logger):
    """Test likelihood evaluation performance using known parameter values.
    
    This function runs a single likelihood evaluation using the same parameter
    values as in test_likelihood_value to provide users with timing and
    likelihood value information before running the full inference.
    
    Args:
        KF: Kalman filter object
        config: Configuration object
        logger: Logger object
        
    Returns
    -------
        float: The computed log likelihood value
    """
    logger.info("=== Likelihood Performance Test ===")
    logger.info("Testing likelihood evaluation with known parameter values...")
    
    # Get noise parameters using the common function
    from argus.utils import get_noise_parameters
    efac_array, equad_array, sigma_p_array, gamma_p_array = get_noise_parameters(config)
    
    # Set test parameter values (same as test_likelihood_value)
    γa_test = 1e-9 
    ha_test = 1e-15
    

    # Create parameter object
    test_params = Parameters(
        log10_gamma_a=jax.numpy.log10(γa_test),
        γa=γa_test,
        ha=ha_test,
        γp=gamma_p_array,
        σp=sigma_p_array,
        EFAC=efac_array,
        EQUAD=equad_array
    )
    
    logger.info(f"Test parameters: γa={γa_test}, ha={ha_test}")
    logger.info(f"Number of pulsars: {len(gamma_p_array)}")
    
    # Time the likelihood evaluation
    logger.info("Performing for the first time a likelihood evaluation...")
    start_time = time.perf_counter()
    
    log_likelihood = KF.get_likelihood(test_params)
    # Ensure computation is complete before stopping timer
    log_likelihood.block_until_ready()
    
    end_time = time.perf_counter()
    duration1 = end_time - start_time


    # Time the likelihood evaluation
    logger.info("Performing timed for the second time a likelihood evaluation...")
    start_time = time.perf_counter()
    
    log_likelihood = KF.get_likelihood(test_params)
    # Ensure computation is complete before stopping timer
    log_likelihood.block_until_ready()
    
    end_time = time.perf_counter()
    duration2 = end_time - start_time



    
    # Log results
    logger.info(f"Likelihood evaluation completed in {duration1:.4f} seconds the first time")
    logger.info(f"Likelihood evaluation completed in {duration2:.4f} seconds the second time")
    logger.info(f"Log likelihood value: {float(log_likelihood)}")
    logger.info("=== End Likelihood Performance Test ===")
    
    return float(log_likelihood)





