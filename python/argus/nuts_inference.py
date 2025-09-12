"""NUTS inference functionality for Bayesian parameter estimation.

This module provides NumPyro NUTS sampling routines and related functionality
for performing Bayesian inference on pulsar timing array data.
"""

import jax.numpy as jnp
import jax.random as random
import numpyro
import arviz as az
from numpyro.infer import MCMC, NUTS
import time

from .parameter_sampling import (
    sample_gw_parameters, 
    sample_pulsar_noise_parameters, 
    sample_measurement_noise_parameters,
    count_free_parameters
)


def log_likelihood_fn(kalman_filter, log10_ha, log10_gamma_a, log10_γp, log10_σp, efac, equad):
    """Calculate log likelihood for NumPyro sampling.
    
    Parameters
    ----------
    kalman_filter : object
        Kalman filter instance with get_likelihood method
    log10_ha : float
        Log10 of GW amplitude
    log10_gamma_a : float
        Log10 of GW spectral index
    log10_γp : jax.Array
        Log10 of pulsar gamma values
    log10_σp : jax.Array
        Log10 of pulsar sigma values
    efac : jax.Array
        EFAC values
    equad : jax.Array
        EQUAD values
        
    Returns
    -------
    float
        Log likelihood value
    """
    from .bayesian_inference import Parameters  # Import here to avoid circular imports
    
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

    return kalman_filter.get_likelihood(params)


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
    log10_ha, log10_gamma_a, γa = sample_gw_parameters(prior_specs)
    log10_γp, log10_σp = sample_pulsar_noise_parameters(prior_specs, n_pulsars)
    efac, equad = sample_measurement_noise_parameters(prior_specs, n_pulsars)
    
    # Calculate log likelihood
    log_likelihood = log_likelihood_fn(kalman_filter, log10_ha, log10_gamma_a, log10_γp, log10_σp, efac, equad)
    
    # Add likelihood to the model
    numpyro.factor("likelihood", log_likelihood)


def setup_nuts_kernel(prior_specs, n_pulsars, config):
    """Set up NUTS kernel with optimized parameters.
    
    Parameters
    ----------
    prior_specs : dict
        Prior specifications dictionary
    n_pulsars : int
        Number of pulsars
    config : configparser.ConfigParser
        Configuration object
        
    Returns
    -------
    tuple
        (nuts_kernel, nuts_info) where nuts_info contains diagnostic information
    """
    # Get NUTS parameters from config with optimized defaults for high-dimensional sampling
    target_accept_prob = config.getfloat('NUTS', 'target_accept_prob', fallback=0.95)  # More conservative for high-dim
    max_tree_depth = config.getint('NUTS', 'max_tree_depth', fallback=10)
    dense_mass = config.getboolean('NUTS', 'dense_mass', fallback=False)
    
    # Handle step_size - only set if explicitly provided in config
    nuts_kwargs = {
        'target_accept_prob': target_accept_prob,
        'max_tree_depth': max_tree_depth,
        'adapt_step_size': True,
        'adapt_mass_matrix': True,
        'dense_mass': dense_mass
    }
    
    # Only add step_size if explicitly set in config
    if config.has_option('NUTS', 'step_size'):
        step_size = config.getfloat('NUTS', 'step_size')
        nuts_kwargs['step_size'] = step_size
        print(f"Using custom step size: {step_size}")
    
    # Count total number of free parameters for diagnostics
    total_params = count_free_parameters(prior_specs, n_pulsars)
    
    nuts_info = {
        'total_params': total_params,
        'target_accept_prob': target_accept_prob,
        'max_tree_depth': max_tree_depth,
        'dense_mass': dense_mass
    }
    
    # Set up NUTS kernel with optimizations
    def model_fn():
        return numpyro_model(None, prior_specs, n_pulsars)  # Will be bound later
    
    kernel = NUTS(model_fn, **nuts_kwargs)
    
    return kernel, nuts_info


def print_nuts_diagnostics(prior_specs, nuts_info, config):
    """Print NUTS sampling diagnostics and parameter information.
    
    Parameters
    ----------
    prior_specs : dict
        Prior specifications
    nuts_info : dict
        NUTS diagnostic information
    config : configparser.ConfigParser
        Configuration object
    """
    n_pulsars = len([spec for spec in prior_specs.keys() if 'pulsar' in spec])
    num_samples = config.getint('NUTS', 'num_samples', fallback=2000)
    num_warmup = config.getint('NUTS', 'num_warmup', fallback=2000)
    num_chains = config.getint('NUTS', 'num_chains', fallback=2)
    
    print("Running NumPyro NUTS inference...")
    print(f"NUTS parameters: {num_samples} samples, {num_warmup} warmup, {num_chains} chains")
    print(f"Target accept prob: {nuts_info['target_accept_prob']} (optimized for high-dimensional sampling)")
    print(f"Dense mass matrix: {nuts_info['dense_mass']}")
    print(f"Max tree depth: {nuts_info['max_tree_depth']}")
    print(f"Total free parameters: {nuts_info['total_params']}")
    
    # Check if hierarchical modeling is enabled
    hierarchical_specs = prior_specs.get('hierarchical_specs')
    if hierarchical_specs:
        hier_gamma = hierarchical_specs.get('hierarchical_noise', False)
        log_ratio = hierarchical_specs.get('log_ratio_parameterization', False)
        if hier_gamma or log_ratio:
            print("Advanced modeling enabled for pulsar noise parameters")
            if hier_gamma and log_ratio:
                print("γp hierarchical + σp via log-ratio parameterization")
                print(f"Effective dimensionality: 4 hyperparameters + {2*n_pulsars} constrained parameters")
                print("σp = γp + ratio (reduces parameter correlations)")
            elif hier_gamma:
                print("γp uses hierarchical priors, σp fixed")
                print(f"Effective dimensionality: 2 hyperparameters + {n_pulsars} constrained parameters")
            elif log_ratio:
                print("σp via log-ratio parameterization, γp independent")
                print(f"Effective dimensionality: 2 hyperparameters + {2*n_pulsars} parameters")
    
    if nuts_info['total_params'] > 10:
        print("High-dimensional parameter space detected - using aggressive NUTS tuning")


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
    arviz.InferenceData
        ArviZ InferenceData object containing MCMC results
    """
    from .prior_models import get_prior_model_specs  # Import here to avoid circular imports
    
    # Get prior model distributions
    prior_specs = get_prior_model_specs(
        config, n_pulsars, sigma_p_array, gamma_p_array, efac_array, equad_array
    )
    
    # Get NUTS parameters from config
    num_samples = config.getint('NUTS', 'num_samples', fallback=2000)
    num_warmup = config.getint('NUTS', 'num_warmup', fallback=2000)
    num_chains = config.getint('NUTS', 'num_chains', fallback=2)
    
    # Set up NUTS kernel
    kernel, nuts_info = setup_nuts_kernel(prior_specs, n_pulsars, config)
    
    # Print diagnostics
    print_nuts_diagnostics(prior_specs, nuts_info, config)
    
    # Create the actual model function bound to the Kalman filter
    def bound_model():
        return numpyro_model(kalman_filter, prior_specs, n_pulsars)
    
    # Create NUTS kernel with bound model
    kernel = NUTS(bound_model, **{
        'target_accept_prob': nuts_info.get('target_accept_prob', 0.95),
        'max_tree_depth': nuts_info.get('max_tree_depth', 10),
        'adapt_step_size': True,
        'adapt_mass_matrix': True,
        'dense_mass': nuts_info.get('dense_mass', False)
    })
    
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


def test_likelihood_performance(kalman_filter, config, logger):
    """Test likelihood evaluation performance using known parameter values.
    
    This function runs a single likelihood evaluation using the same parameter
    values as in test_likelihood_value to provide users with timing and
    likelihood value information before running the full inference.
    
    Parameters
    ----------
    kalman_filter : object
        Kalman filter object
    config : configparser.ConfigParser
        Configuration object
    logger : logging.Logger
        Logger object
        
    Returns
    -------
    float
        The computed log likelihood value
    """
    from .bayesian_inference import Parameters  # Import here to avoid circular imports
    from argus.utils import get_noise_parameters
    
    logger.info("=== Likelihood Performance Test ===")
    logger.info("Testing likelihood evaluation with known parameter values...")
    
    # Get noise parameters using the common function
    efac_array, equad_array, sigma_p_array, gamma_p_array = get_noise_parameters(config)
    
    # Set test parameter values
    γa_test = 1e-9 
    ha_test = 1e-15
    
    # Create parameter object
    test_params = Parameters(
        log10_gamma_a=jnp.log10(γa_test),
        γa=γa_test,
        ha=ha_test,
        γp=gamma_p_array,
        σp=sigma_p_array,
        EFAC=efac_array,
        EQUAD=equad_array
    )
    
    logger.info(f"Test parameters: γa={γa_test}, ha={ha_test}")
    logger.info(f"Number of pulsars: {len(gamma_p_array)}")
    
    # Time the likelihood evaluation (first time)
    logger.info("Performing for the first time a likelihood evaluation...")
    start_time = time.perf_counter()
    
    log_likelihood = kalman_filter.get_likelihood(test_params)
    # Ensure computation is complete before stopping timer
    log_likelihood.block_until_ready()
    
    end_time = time.perf_counter()
    duration1 = end_time - start_time

    # Time the likelihood evaluation (second time)
    logger.info("Performing timed for the second time a likelihood evaluation...")
    start_time = time.perf_counter()
    
    log_likelihood = kalman_filter.get_likelihood(test_params)
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