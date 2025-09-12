"""Prior model specifications for Bayesian inference.

This module provides functionality for defining and creating prior distributions
for gravitational wave background and pulsar noise parameters used in 
pulsar timing array analysis.
"""

import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

tfpd = tfp.distributions


def create_hierarchical_priors(config, hierarchical_noise, log_ratio_parameterization):
    """Create hierarchical modeling prior distributions if needed.
    
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


def get_gw_parameter_priors(config):
    """Extract gravitational wave parameter prior distributions from config.
    
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


def get_pulsar_noise_priors(config, n_pulsars, sigma_p_array, gamma_p_array):
    """Extract pulsar red noise parameter prior distributions from config.
    
    Parameters
    ----------
    config : ConfigParser
        Configuration object containing prior model settings
    n_pulsars : int
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
    except Exception:
        # If no spin_injections_path, sample from priors
        log10_gamma_p_fixed = False
        log10_sigma_p_fixed = False
        print("No spin_injections_path provided, sampling red noise parameters from priors")
    
    # Check for hierarchical modeling and log-ratio parameterization
    hierarchical_noise = config.getboolean('PriorModel', 'hierarchical_noise', fallback=False)
    log_ratio_parameterization = config.getboolean('PriorModel', 'log_ratio_parameterization', fallback=False)
    hierarchical_specs = create_hierarchical_priors(config, hierarchical_noise, log_ratio_parameterization)
    
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
                log10_gamma_p_spec = jnp.full(n_pulsars, gamma_p_fixed_value)
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
            low=jnp.full(n_pulsars, config.getfloat('PriorModel', 'log10_gamma_p_min')),
            high=jnp.full(n_pulsars, config.getfloat('PriorModel', 'log10_gamma_p_max'))
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
                log10_sigma_p_spec = jnp.full(n_pulsars, sigma_p_fixed_value)
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
            low=jnp.full(n_pulsars, config.getfloat('PriorModel', 'log10_sigma_p_min')),
            high=jnp.full(n_pulsars, config.getfloat('PriorModel', 'log10_sigma_p_max'))
        )

    return {
        'log10_gamma_p_spec': log10_gamma_p_spec,
        'log10_sigma_p_spec': log10_sigma_p_spec,
        'hierarchical_specs': hierarchical_specs
    }


def get_measurement_noise_priors(config, efac_array, equad_array):
    """Extract measurement noise parameter prior distributions from config.
    
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
    except Exception:
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


def get_prior_model_specs(config, n_pulsars, sigma_p_array, gamma_p_array, efac_array, equad_array):
    """Create prior model distributions based on config settings.
    
    Parameters
    ----------
    config : ConfigParser
        Configuration object containing prior model settings
    n_pulsars : int
        Number of pulsars
    sigma_p_array : array
        Array of pulsar red noise sigma values, only used if psr_noise_fixed=true in config
    gamma_p_array : array
        Array of pulsar red noise gamma values, only used if psr_noise_fixed=true in config
    efac_array : array
        Array of EFAC values, only used if efac_equad_fixed=true in config
    equad_array : array
        Array of EQUAD values, only used if efac_equad_fixed=true in config
        
    Returns
    -------
    dict
        Dictionary containing all prior distributions with the following keys:
        - log10_ha_spec: Prior distribution for log10 of GW amplitude (fixed value or Uniform)
        - log10_gamma_a_spec: Prior distribution for log10(GW spectral index) (fixed value or Uniform)
        - log10_gamma_p_spec: Prior distribution for log10 of pulsar red noise gamma (Uniform)
        - log10_sigma_p_spec: Prior distribution for log10 of pulsar red noise sigma (Uniform)
        - efac_spec: Prior distribution for EFAC (fixed array or Uniform)
        - equad_spec: Prior distribution for EQUAD (fixed array or Uniform)
        
    Notes
    -----
    For each parameter, the prior type (fixed or Uniform) is determined by the
    corresponding *_fixed setting in the config file. If fixed=true, the *_value
    is used; if fixed=false, a Uniform distribution is created using *_min and *_max.
    For EFAC and EQUAD, if efac_equad_fixed=true, the provided arrays are used;
    otherwise, Uniform distributions are created using efac_min/max and equad_min/max.
    The same logic applies to the pulsar red noise parameters.
    """
    print("Getting prior model specs...")
    
    # Get parameter prior distributions from specialized functions
    gw_specs = get_gw_parameter_priors(config)
    pulsar_noise_specs = get_pulsar_noise_priors(config, n_pulsars, sigma_p_array, gamma_p_array)
    measurement_noise_specs = get_measurement_noise_priors(config, efac_array, equad_array)

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