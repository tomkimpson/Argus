"""Inference execution and performance testing utilities for the argus package."""

import time
from jax import random
import jax

from jaxns import NestedSampler, TerminationCondition
from argus import bayesian_inference, utils


def test_likelihood_performance(KF, config, logger):
    """Test likelihood evaluation performance using known parameter values.
    
    This function runs a single likelihood evaluation using the same parameter
    values as in test_likelihood_value to provide users with timing and
    likelihood value information before running the full nested sampling.
    
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
    from argus.workflow import get_noise_parameters
    efac_array, equad_array, sigma_p_array, gamma_p_array = get_noise_parameters(config)
    
    # Set test parameter values (same as test_likelihood_value)
    γa_test = 1e-9 
    ha_test = 1e-15
    

    # Create parameter object
    test_params = bayesian_inference.Parameters(
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


def run_nested_sampling(config, jax_model, logger):
    """Run the nested sampling algorithm.
    
    Args:
        config: Configuration object
        jax_model: JAX model object
        logger: Logger object
    
    Returns
    -------
        tuple: (termination_reason, state, ns)
    """
    logger.info("Initializing nested sampling...")
    ns = NestedSampler(
        model=jax_model,
        num_live_points=config.getint('NestedSampling', 'num_live_points', fallback=100),
        verbose=True
    )

    logger.info("Running nested sampling...")
    term_cond = TerminationCondition(
        dlogZ=config.getfloat('NestedSampling', 'dlogZ', fallback=0.1)
    )
    termination_reason, state = jax.jit(ns)(
        key=random.PRNGKey(432345987),
        term_cond=term_cond
    )
    
    return termination_reason, state, ns


def run_jaxns_inference(config, jax_model, param_names, output_dir, output_id, logger):
    """Run JAXNS nested sampling inference pipeline.
    
    Args:
        config: Configuration object
        jax_model: JAX model object
        param_names: List of parameter names
        output_dir: Output directory path
        output_id: Output identifier
        logger: Logger object
        
    Returns
    -------
        dict: Results dictionary
    """
    # Sample from prior and evaluate likelihood for testing (JAXNS only)
    u = jax_model.sample_U(key=random.PRNGKey(432345987))
    θ = jax_model.transform(u)
    
    params = [θ[name] for name in param_names]
    log_likelihood = jax_model.log_likelihood(*params)
    logger.info("\nLog likelihood for parameters sampled from prior:")
    logger.info(str(log_likelihood))
    
    # Run nested sampling
    if config.getboolean('NestedSampling', 'run_sampling', fallback=True):
        termination_reason, state, ns = run_nested_sampling(config, jax_model, logger)
        
        # Save results and create plots
        from argus.io_manager import save_jaxns_results
        results_path = save_jaxns_results(ns, termination_reason, state, output_dir, output_id, logger)
        
        # Create corner plot
        logger.info("Loading results and creating corner plot...")
        plot_path = utils.corner_plot(results_path, output_dir)
        if plot_path:
            logger.info(f"Corner plot saved to {plot_path}")
        
        return ns.to_results(termination_reason=termination_reason, state=state)
    else:
        logger.info("Nested sampling is not being run")
        return None


def run_numpyro_inference(config, KF, pulsar_data, output_dir, output_id, logger):
    """Run NumPyro NUTS inference pipeline.
    
    Args:
        config: Configuration object
        KF: Kalman filter object
        pulsar_data: Processed pulsar data
        output_dir: Output directory path
        output_id: Output identifier
        logger: Logger object
        
    Returns
    -------
        str: Path to saved results file
    """
    # Get noise parameters
    from argus.workflow import get_noise_parameters
    efac_array, equad_array, sigma_p_array, gamma_p_array = get_noise_parameters(config)
    
    # Run inference using the dispatcher function
    logger.info("Running NUMPYRO inference...")
    results = bayesian_inference.run_inference(
        KF, config, len(pulsar_data['metadata']), 
        sigma_p_array, gamma_p_array, efac_array, equad_array
    )
    
    # Save results
    from argus.io_manager import save_numpyro_results
    results_path = save_numpyro_results(results, output_dir, output_id, logger)
    
    # Create plots and diagnostics for NUTS
    logger.info("Creating corner plot and diagnostics for NUTS results...")
    
    # Create corner plot
    try:
        plot_path = utils.corner_plot(results_path, output_dir)
        if plot_path:
            logger.info(f"Corner plot saved to {plot_path}")
        
    except Exception as e:
        logger.error(f"Error creating corner plot: {e}")
    
    # Run diagnostics
    try:
        logger.info("Running MCMC diagnostics...")
        utils.diagnostics(results_path, output_dir)
        logger.info("MCMC diagnostics completed")
    except Exception as e:
        logger.error(f"Error running diagnostics: {e}")
    
    return results_path