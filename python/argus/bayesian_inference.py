"""Bayesian inference module for pulsar timing array analysis.

This module provides functionality for performing Bayesian parameter estimation
on pulsar timing array data using nested sampling. It includes:

- Prior definitions for gravitational wave background and pulsar noise parameters
- Likelihood calculations using a Kalman filter implementation
- Nested sampling routines using JAXNS

The module is designed to work with the JAX framework for automatic differentiation
and GPU acceleration. It handles parameters like:

- Gravitational wave background amplitude (ha) and spectral index (γa)
- Pulsar-specific red noise parameters (γp, σp)
- White noise parameters (EFAC, EQUAD)

The implementation uses the Hellings-Downs correlation pattern for the 
gravitational wave background and models pulsar red noise as an 
Ornstein-Uhlenbeck process.

Uses JAXNS for JAX-native nested sampling with automatic differentiation.
"""

#Jax
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

#Jaxns
from jaxns import Prior, Model, NestedSampler,TerminationCondition # Import necessary components

#Tensorflow - for setting up the prior distributions
import tensorflow_probability.substrates.jax as tfp
tfpd = tfp.distributions

#NumPyro - for NUTS sampling
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

#Arviz - for NumPyro result handling
import arviz as az

#Flax
from flax import struct



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



def configurable_prior_model(
    Npsr: int, # Number of pulsars, needed for array shapes if defaults are used
    # --- Specifications for each parameter ---
    # Each *_spec can be a TFP distribution object or a fixed jnp.ndarray/float.
    # If None, a default distribution will be used for some, or an error raised for others.
    log10_ha_spec = tfpd.Uniform(-17.0, -14.0),
    gamma_a_spec = 1e-9, # Typically fixed
    log10_gamma_p_spec = None,
    log10_sigma_p_spec = None,
    efac_spec = None,
    equad_spec = None
):
    """
    Defines prior distributions for the parameters.
    Each parameter's specification (e.g., `log10_ha_spec`) can be:
    - A TFP distribution object (e.g., tfpd.Uniform(...)) for sampling.
    - A scalar or jnp.ndarray for a fixed value.

    If a value (e.g. -15.0), Prior will use it as fixed.
    If (e.g.) tfpd.Uniform(...), Prior will sample from it.
    """

    # GW parameters

    log10_ha = yield Prior(log10_ha_spec, name='log10_ha')
    γa = yield Prior(gamma_a_spec, name='γa') 

    # PSR vector parameters: γp and σp.
    log10_γp = yield Prior(log10_gamma_p_spec   , name='log10_γp')
    log10_σp = yield Prior(log10_sigma_p_spec, name='log10_σp')
    
    # Measurement noise parameters: EFAC and EQUAD.
    efac = yield Prior(efac_spec, name='efac')
    equad = yield Prior(equad_spec, name='equad')

    return log10_ha, γa, log10_γp, log10_σp, efac, equad    



# JAXNS model
def jaxns_log_likelihood(KF, log10_ha, γa, log10_γp, log10_σp, efac, equad):
    ha = 10.0 ** log10_ha
    γp = 10.0 ** log10_γp
    σp = 10.0 ** log10_σp

    params = Parameters(
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

def get_prior_model_specs(config, Npsr,sigma_p_array,gamma_p_array, efac_array, equad_array):
    """
    Create prior model specifications based on config settings.
    
    Args:
        config: ConfigParser object containing prior model settings
        Npsr: Number of pulsars
        sigma_p_array: Array of pulsar red noise sigma values, only used if psr_noise_fixed=true in config
        gamma_p_array: Array of pulsar red noise gamma values, only used if psr_noise_fixed=true in config
        efac_array: Array of EFAC values, only used if efac_equad_fixed=true in config
        equad_array: Array of EQUAD values, only used if efac_equad_fixed=true in config
        
    Returns:
        dict: Dictionary containing all prior specifications with the following keys:
            - log10_ha_spec: Prior for log10 of GW amplitude (fixed value or Uniform)
            - gamma_a_spec: Prior for GW spectral index (fixed value or Uniform)
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
    log10_ha_spec = get_prior_spec('log10_ha')
    gamma_a_spec = get_prior_spec('gamma_a')


    #Pulsar red noise parameters
    psr_noise_fixed = config.getboolean('PriorModel', 'psr_noise_fixed')
    print("The psr_noise_fixed is:")
    print(psr_noise_fixed)
    if psr_noise_fixed:
        log10_gamma_p_spec = jnp.log10(gamma_p_array)
        log10_sigma_p_spec = jnp.log10(sigma_p_array)
    else:
        log10_gamma_p_spec = tfpd.Uniform(
            low=jnp.full(Npsr, config.getfloat('PriorModel', 'log10_gamma_p_min')),
            high=jnp.full(Npsr, config.getfloat('PriorModel', 'log10_gamma_p_max'))
        )
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
        equad_spec = tfpd.Uniform(
            low=jnp.full_like(equad_array, config.getfloat('PriorModel', 'equad_min')),
            high=jnp.full_like(equad_array, config.getfloat('PriorModel', 'equad_max'))
        )

    return {
        'log10_ha_spec': log10_ha_spec,
        'gamma_a_spec': gamma_a_spec,
        'log10_gamma_p_spec': log10_gamma_p_spec,
        'log10_sigma_p_spec': log10_sigma_p_spec,
        'efac_spec': efac_spec,
        'equad_spec': equad_spec
    }




def run_jaxns_inference(kalman_filter, config, n_pulsars, sigma_p_array, gamma_p_array, 
                       efac_array, equad_array):
    """Run JAXNS nested sampling inference.
    
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
    results : object
        JAXNS results object
    """
    import jax.random as random
    
    # Get prior model specifications
    prior_specs = get_prior_model_specs(
        config, n_pulsars, sigma_p_array, gamma_p_array, efac_array, equad_array
    )
    
    # Create prior model
    def prior_model():
        return configurable_prior_model(
            Npsr=n_pulsars,
            **prior_specs
        )
    
    # Create likelihood function
    def log_likelihood(*args):
        return jaxns_log_likelihood(kalman_filter, *args)
    
    # Create JAXNS model
    jax_model = Model(prior_model=prior_model, log_likelihood=log_likelihood)
    
    # Get sampling parameters
    num_live_points = config.getint('NestedSampling', 'num_live_points', fallback=1000)
    dlogZ = config.getfloat('NestedSampling', 'dlogZ', fallback=0.1)
    
    # Initialize sampler
    ns = NestedSampler(
        model=jax_model,
        num_live_points=num_live_points,
        verbose=True
    )
    
    # Run sampling
    term_cond = TerminationCondition(dlogZ=dlogZ)
    termination_reason, state = jax.jit(ns)(
        key=random.PRNGKey(42),
        term_cond=term_cond
    )
    
    # Convert results
    results = ns.to_results(termination_reason=termination_reason, state=state)
    
    return results




def numpyro_model(kalman_filter, prior_specs, n_pulsars):
    """NumPyro model definition for Bayesian inference.
    
    This function defines the NumPyro probabilistic model using the same
    prior specifications as the JAXNS model to ensure consistency.
    
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
    if isinstance(prior_specs['log10_ha_spec'], tfpd.Distribution):
        log10_ha = numpyro.sample("log10_ha", 
            dist.Uniform(prior_specs['log10_ha_spec'].low, prior_specs['log10_ha_spec'].high))
    else:
        log10_ha = numpyro.deterministic("log10_ha", prior_specs['log10_ha_spec'])
    
    if isinstance(prior_specs['gamma_a_spec'], tfpd.Distribution):
        γa = numpyro.sample("γa", 
            dist.Uniform(prior_specs['gamma_a_spec'].low, prior_specs['gamma_a_spec'].high))
    else:
        γa = numpyro.deterministic("γa", prior_specs['gamma_a_spec'])
    
    # Pulsar red noise parameters
    if isinstance(prior_specs['log10_gamma_p_spec'], tfpd.Distribution):
        log10_γp = numpyro.sample("log10_γp", 
            dist.Uniform(prior_specs['log10_gamma_p_spec'].low, prior_specs['log10_gamma_p_spec'].high))
    else:
        log10_γp = numpyro.deterministic("log10_γp", prior_specs['log10_gamma_p_spec'])
    
    if isinstance(prior_specs['log10_sigma_p_spec'], tfpd.Distribution):
        log10_σp = numpyro.sample("log10_σp", 
            dist.Uniform(prior_specs['log10_sigma_p_spec'].low, prior_specs['log10_sigma_p_spec'].high))
    else:
        log10_σp = numpyro.deterministic("log10_σp", prior_specs['log10_sigma_p_spec'])
    
    # Measurement noise parameters
    if isinstance(prior_specs['efac_spec'], tfpd.Distribution):
        efac = numpyro.sample("efac", 
            dist.Uniform(prior_specs['efac_spec'].low, prior_specs['efac_spec'].high))
    else:
        efac = numpyro.deterministic("efac", prior_specs['efac_spec'])
    
    if isinstance(prior_specs['equad_spec'], tfpd.Distribution):
        equad = numpyro.sample("equad", 
            dist.Uniform(prior_specs['equad_spec'].low, prior_specs['equad_spec'].high))
    else:
        equad = numpyro.deterministic("equad", prior_specs['equad_spec'])
    
    # Calculate log likelihood using the same function as JAXNS
    log_likelihood = jaxns_log_likelihood(kalman_filter, log10_ha, γa, log10_γp, log10_σp, efac, equad)
    
    # Add likelihood to the model
    numpyro.factor("likelihood", log_likelihood)


def run_numpyro_inference(kalman_filter, config, n_pulsars, sigma_p_array, gamma_p_array, 
                         efac_array, equad_array):
    """Run NumPyro NUTS inference.
    
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
    
    # Get NUTS parameters from config
    num_samples = config.getint('NUTS', 'num_samples', fallback=2000)
    num_warmup = config.getint('NUTS', 'num_warmup', fallback=2000)
    num_chains = config.getint('NUTS', 'num_chains', fallback=2)
    target_accept_prob = config.getfloat('NUTS', 'target_accept_prob', fallback=0.8)
    
    print(f"Running NumPyro NUTS inference with {n_pulsars} pulsars...")
    print(f"NUTS parameters: {num_samples} samples, {num_warmup} warmup, {num_chains} chains")
    
    # Set up NUTS kernel
    kernel = NUTS(
        lambda: numpyro_model(kalman_filter, prior_specs, n_pulsars),
        target_accept_prob=target_accept_prob
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


def run_inference(kalman_filter, config, n_pulsars, sigma_p_array=None, 
                 gamma_p_array=None, efac_array=None, equad_array=None):
    """Run Bayesian inference using either JAXNS or NumPyro based on config.
    
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
    result : object
        Either JAXNS results object or ArviZ InferenceData object
    """
    
    # Get inference method from config
    method = config.get('Inference', 'method', fallback='jaxns').lower()
    
    if method == 'jaxns':
        print(f"Running JAXNS inference with {n_pulsars} pulsars...")
        return run_jaxns_inference(
            kalman_filter, config, n_pulsars, 
            sigma_p_array, gamma_p_array, efac_array, equad_array
        )
    elif method == 'numpyro':
        return run_numpyro_inference(
            kalman_filter, config, n_pulsars,
            sigma_p_array, gamma_p_array, efac_array, equad_array
        )
    else:
        raise ValueError(f"Unknown inference method: {method}. Choose 'jaxns' or 'numpyro'.")