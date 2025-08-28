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



def print_parameters(params: Parameters):
    """Print all entries of a Parameters struct."""
    for field in params.__dataclass_fields__:
        value = getattr(params, field)
        print(f"{field}: {value}")

def get_prior_model_specs(config, Npsr, sigma_p_array, gamma_p_array, efac_array, equad_array):
    """
    Create prior model specifications based on config settings.
    
    Args:
        config: ConfigParser object containing prior model settings
        Npsr: Number of pulsars
        sigma_p_array: Array of pulsar red noise sigma values, only used if psr_noise_fixed=true in config
        gamma_p_array: Array of pulsar red noise gamma values, only used if psr_noise_fixed=true in config
        efac_array: Array of EFAC values, only used if efac_equad_fixed=true in config
        equad_array: Array of EQUAD values, only used if efac_equad_fixed=true in config
        
    Returns
    -------
        dict: Dictionary containing all prior specifications with the following keys:
            - log10_ha_spec: Prior for log10 of GW amplitude (fixed value or Uniform)
            - log10_gamma_a_spec: Prior for log10(GW spectral index) (fixed value or Uniform)
            - log10_gamma_p_spec: Prior for log10 of pulsar red noise gamma (Uniform)
            - log10_sigma_p_spec: Prior for log10 of pulsar red noise sigma (Uniform)
            - efac_spec: Prior for EFAC (fixed array or Uniform)
            - equad_spec: Prior for EQUAD (fixed array or Uniform)
            
    Note:
        For each parameter, the prior type (fixed or Uniform) is determined by the
        corresponding *_fixed setting in the config file. If fixed=true, the *_value
        is used; if fixed=false, a Uniform distribution is created using *_min and *_max.
        For EFAC and EQUAD, if efac_equad_fixed=true, the provided arrays are used;
        otherwise, Uniform distributions are created using efac_min/max and equad_min/max.
        The same logic applies to the pulsar red noise parameters.
    """
    print("Getting prior model specs...")
    # Helper function to create prior spec based on fixed/sampled setting
    def get_prior_spec(param_name):
        is_fixed = config.getboolean('PriorModel', f'{param_name}_fixed')
        if is_fixed:
            return config.getfloat('PriorModel', f'{param_name}_value')
        else:
            min_val = config.getfloat('PriorModel', f'{param_name}_min')
            max_val = config.getfloat('PriorModel', f'{param_name}_max')
            return tfpd.Uniform(min_val, max_val)

    # Get prior specifications for each parameter


    #GW parameters
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


    #Pulsar red noise parameters
    # Check for individual parameter control (new approach) vs legacy master switch
    log10_gamma_p_fixed = config.getboolean('PriorModel', 'log10_gamma_p_fixed', fallback=None)
    log10_sigma_p_fixed = config.getboolean('PriorModel', 'log10_sigma_p_fixed', fallback=None)
    
    # Backwards compatibility: if individual controls not set, use legacy psr_noise_fixed
    if log10_gamma_p_fixed is None and log10_sigma_p_fixed is None:
        psr_noise_fixed = config.getboolean('PriorModel', 'psr_noise_fixed')
        log10_gamma_p_fixed = psr_noise_fixed
        log10_sigma_p_fixed = psr_noise_fixed
        print(f"Using legacy psr_noise_fixed: {psr_noise_fixed}")
    else:
        # Use individual controls, defaulting to False if not specified
        if log10_gamma_p_fixed is None:
            log10_gamma_p_fixed = False
        if log10_sigma_p_fixed is None:
            log10_sigma_p_fixed = False
        print(f"Using individual controls - gamma_p_fixed: {log10_gamma_p_fixed}, sigma_p_fixed: {log10_sigma_p_fixed}")
    
    # Check for hierarchical modeling and log-ratio parameterization
    hierarchical_noise = config.getboolean('PriorModel', 'hierarchical_noise', fallback=False)
    hierarchical_sigma_p = config.getboolean('PriorModel', 'hierarchical_sigma_p', fallback=False)
    log_ratio_parameterization = config.getboolean('PriorModel', 'log_ratio_parameterization', fallback=False)
    hierarchical_specs = None
    
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
        # Set up hierarchical prior specifications
        hierarchical_specs = {
            'hierarchical_noise': hierarchical_noise,
            'hierarchical_sigma_p': hierarchical_sigma_p,
            'log_ratio_parameterization': log_ratio_parameterization
        }
        
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
    elif hierarchical_noise or hierarchical_sigma_p or log_ratio_parameterization:
        # Initialize hierarchical_specs if not already done
        if hierarchical_specs is None:
            hierarchical_specs = {
                'hierarchical_noise': hierarchical_noise,
                'hierarchical_sigma_p': hierarchical_sigma_p,
                'log_ratio_parameterization': log_ratio_parameterization
            }
        
        if hierarchical_sigma_p:
            hierarchical_specs.update({
                'log10_sigma_p_mean_spec': tfpd.Uniform(
                    config.getfloat('PriorModel', 'log10_sigma_p_mean_min'),
                    config.getfloat('PriorModel', 'log10_sigma_p_mean_max')
                ),
                'log10_sigma_p_std_spec': tfpd.Uniform(
                    config.getfloat('PriorModel', 'log10_sigma_p_std_min'),
                    config.getfloat('PriorModel', 'log10_sigma_p_std_max')
                )
            })
            log10_sigma_p_spec = None  # Will be handled hierarchically
        elif log_ratio_parameterization:
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
            log10_sigma_p_spec = None  # Will be derived from log-ratio
        else:
            # Free sigma_p with uniform prior
            log10_sigma_p_spec = tfpd.Uniform(
                low=jnp.full(Npsr, config.getfloat('PriorModel', 'log10_sigma_p_min')),
                high=jnp.full(Npsr, config.getfloat('PriorModel', 'log10_sigma_p_max'))
            )
    else:
        # Free sigma_p with uniform prior
        log10_sigma_p_spec = tfpd.Uniform(
            low=jnp.full(Npsr, config.getfloat('PriorModel', 'log10_sigma_p_min')),
            high=jnp.full(Npsr, config.getfloat('PriorModel', 'log10_sigma_p_max'))
        )

    #Measurement noise parameters
    efac_equad_fixed = config.getboolean('PriorModel', 'efac_equad_fixed')
    if efac_equad_fixed:
        efac_spec = efac_array
        equad_spec = equad_array
    else:
        efac_spec = tfpd.Uniform(
            low=jnp.full_like(efac_array, config.getfloat('PriorModel', 'efac_min')),
            high=jnp.full_like(efac_array, config.getfloat('PriorModel', 'efac_max'))
        )
        
        # Check if we should use log10 parameterization for EQUAD
        try:
            equad_log10_prior = config.getboolean('PriorModel', 'equad_log10_prior')
        except:
            equad_log10_prior = False  # Default to direct EQUAD for backward compatibility
            
        if equad_log10_prior:
            # Use log10(EQUAD) uniform prior - transformation handled in numpyro model
            log10_equad_spec = tfpd.Uniform(
                low=jnp.full_like(equad_array, config.getfloat('PriorModel', 'log10_equad_min')),
                high=jnp.full_like(equad_array, config.getfloat('PriorModel', 'log10_equad_max'))
            )
            equad_spec = {'log10_equad_spec': log10_equad_spec, 'use_log10': True}
        else:
            # Direct EQUAD uniform prior (backward compatibility)
            equad_spec = tfpd.Uniform(
                low=jnp.full_like(equad_array, config.getfloat('PriorModel', 'equad_min')),
                high=jnp.full_like(equad_array, config.getfloat('PriorModel', 'equad_max'))
            )

    return {
        'log10_ha_spec': log10_ha_spec,
        'log10_ha_transform_params': log10_ha_transform_params,
        'log10_gamma_a_spec': log10_gamma_a_spec,
        'log10_gamma_p_spec': log10_gamma_p_spec,
        'log10_sigma_p_spec': log10_sigma_p_spec,
        'efac_spec': efac_spec,
        'equad_spec': equad_spec,
        'hierarchical_specs': hierarchical_specs
    }








def numpyro_model(kalman_filter, prior_specs, n_pulsars):
    """NumPyro model definition for Bayesian inference with parameter standardization.
    
    This function defines the NumPyro probabilistic model using standardized
    parameter transformations for better NUTS sampling in high-dimensional spaces.
    
    Parameters
    ----------
    kalman_filter : object
        JAX Kalman filter with get_likelihood method
    prior_specs : dict
        Dictionary containing prior specifications from get_prior_model_specs()
    n_pulsars : int
        Number of pulsars
    """
    # Sample/determine parameters based on prior specifications
    # GW parameters
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
    
    # Pulsar red noise parameters with hierarchical or standardization modeling
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
    
    # Handle log10_sigma_p
    if hierarchical_specs and hierarchical_specs.get('hierarchical_sigma_p', False):
        # Hierarchical modeling for log10_sigma_p with gradient balancing
        log10_sigma_p_mean_raw = numpyro.sample("log10_sigma_p_mean_raw", dist.Normal(0.0, 1.0))
        log10_sigma_p_std_raw = numpyro.sample("log10_sigma_p_std_raw", dist.Normal(0.0, 1.0))
        
        # Transform to appropriate ranges with balanced gradients
        mean_low = hierarchical_specs['log10_sigma_p_mean_spec'].low
        mean_high = hierarchical_specs['log10_sigma_p_mean_spec'].high
        std_low = hierarchical_specs['log10_sigma_p_std_spec'].low
        std_high = hierarchical_specs['log10_sigma_p_std_spec'].high
        
        # Apply gradient-balanced transforms
        log10_sigma_p_mean = numpyro.deterministic("log10_sigma_p_mean", 
            (mean_low + mean_high) / 2.0 + log10_sigma_p_mean_raw * (mean_high - mean_low) / 6.0)
        log10_sigma_p_std = numpyro.deterministic("log10_sigma_p_std", 
            (std_low + std_high) / 2.0 + log10_sigma_p_std_raw * (std_high - std_low) / 6.0)
        
        # Sample individual pulsar parameters with scaled gradients
        log10_σp_raw = numpyro.sample("log10_σp_raw", 
            dist.Normal(jnp.zeros(n_pulsars), jnp.ones(n_pulsars)))
        log10_σp = numpyro.deterministic("log10_σp", 
            log10_sigma_p_mean + log10_σp_raw * log10_sigma_p_std / jnp.sqrt(n_pulsars))
    elif hierarchical_specs and hierarchical_specs.get('log_ratio_parameterization', False):
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
    
    # Measurement noise parameters with standardization
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
    
    # Calculate log likelihood
    log_likelihood = log_likelihood_fn(kalman_filter, log10_ha, log10_gamma_a, log10_γp, log10_σp, efac, equad)
    
    # Add likelihood to the model
    numpyro.factor("likelihood", log_likelihood)


def count_free_parameters(prior_specs, n_pulsars):
    """Count the total number of free (non-fixed) parameters for NUTS sampling.
    
    Parameters
    ----------
    prior_specs : dict
        Prior specifications dictionary
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
    
    if hierarchical_specs and hierarchical_specs.get('hierarchical_sigma_p', False):
        # Hierarchical modeling: 2 hyperparameters + n_pulsars individual parameters
        count += 2  # log10_sigma_p_mean and log10_sigma_p_std
        count += n_pulsars  # Individual pulsar sigma parameters
    elif hierarchical_specs and hierarchical_specs.get('log_ratio_parameterization', False):
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


def run_numpyro_inference(kalman_filter, config, n_pulsars, sigma_p_array, gamma_p_array, 
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
    
    # Get prior model specifications
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
        hier_sigma = hierarchical_specs.get('hierarchical_sigma_p', False)
        log_ratio = hierarchical_specs.get('log_ratio_parameterization', False)
        if hier_gamma or hier_sigma or log_ratio:
            print("Advanced modeling enabled for pulsar noise parameters")
            if hier_gamma and hier_sigma:
                print(f"Both γp and σp use hierarchical priors")
                print(f"Effective dimensionality: 4 hyperparameters + {2*n_pulsars} constrained parameters")
            elif hier_gamma and log_ratio:
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
    rng_key = random.PRNGKey(42)  # Use same seed as JAXNS for reproducibility
    sampler.run(rng_key)
    
    # Print summary
    sampler.print_summary()
    
    # Convert to ArviZ format
    inf_data = az.from_numpyro(sampler)
    
    return inf_data


def display_prior_summary(prior_specs, n_pulsars, logger=None):
    """Display a readable summary of all prior specifications.
    
    Parameters
    ----------
    prior_specs : dict
        Dictionary containing prior specifications from get_prior_model_specs()
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
    if hierarchical_specs and hierarchical_specs.get('hierarchical_sigma_p', False):
        # Hierarchical modeling case
        mean_spec = hierarchical_specs['log10_sigma_p_mean_spec']
        std_spec = hierarchical_specs['log10_sigma_p_std_spec']
        log_or_print(f"log10(σ_p): HIERARCHICAL modeling")
        log_or_print(f"  - Population mean: Uniform({float(mean_spec.low):.1f}, {float(mean_spec.high):.1f})")
        log_or_print(f"  - Population std: Uniform({float(std_spec.low):.1f}, {float(std_spec.high):.1f})")
        log_or_print(f"  - Individual pulsars: Normal(population_mean, population_std)")
    elif hierarchical_specs and hierarchical_specs.get('log_ratio_parameterization', False):
        # Log-ratio parameterization case
        mean_spec = hierarchical_specs['log10_ratio_mean_spec']
        std_spec = hierarchical_specs['log10_ratio_std_spec']
        log_or_print(f"log10(σ_p): LOG-RATIO parameterization")
        log_or_print(f"  - log10(σ_p) = log10(γ_p) + log10(ratio)")
        log_or_print(f"  - Ratio mean: Uniform({float(mean_spec.low):.1f}, {float(mean_spec.high):.1f})")
        log_or_print(f"  - Ratio std: Uniform({float(std_spec.low):.1f}, {float(std_spec.high):.1f})")
        log_or_print(f"  - Individual ratios: Normal(ratio_mean, ratio_std)")
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


def run_inference(kalman_filter, config, n_pulsars, sigma_p_array=None, 
                 gamma_p_array=None, efac_array=None, equad_array=None):
    """Run Bayesian inference using NumPyro NUTS sampling.
    
    Parameters
    ----------
    kalman_filter : object
        JAX Kalman filter with get_likelihood method
    config : configparser.ConfigParser
        Configuration object
    n_pulsars : int
        Number of pulsars
    sigma_p_array : jnp.ndarray, optional
        Pulsar red noise sigma values
    gamma_p_array : jnp.ndarray, optional
        Pulsar red noise gamma values
    efac_array : jnp.ndarray, optional
        EFAC values
    equad_array : jnp.ndarray, optional
        EQUAD values
        
    Returns
    -------
    result : arviz.InferenceData
        ArviZ InferenceData object containing MCMC results
    """
    # Get inference method from config
    method = config.get('Inference', 'method', fallback='numpyro').lower()
    
    if method == 'jaxns':
        raise ValueError("JAXNS nested sampling is no longer supported. Please use 'numpyro' for NUTS sampling.")
    elif method == 'numpyro':
        return run_numpyro_inference(
            kalman_filter, config, n_pulsars,
            sigma_p_array, gamma_p_array, efac_array, equad_array
        )
    else:
        raise ValueError(f"Unknown inference method: {method}. Only 'numpyro' is supported.")