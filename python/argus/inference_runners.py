"""Inference execution and performance testing utilities for the argus package."""

import time
from jax import random
import jax

from argus import bayesian_inference, utils


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
    from argus.workflow import get_noise_parameters
    efac_array, equad_array, sigma_p_array, gamma_p_array = get_noise_parameters(config)
    
    # Set test parameter values (same as test_likelihood_value)
    γa_test = 1e-9 
    ha_test = 1e-15
    

    # Create parameter object
    test_params = bayesian_inference.Parameters(
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






def estimate_runtime(config, likelihood_time, logger, n_free_params=None):
    """Estimate NUTS runtime based on sampling parameters and likelihood timing.
    
    Args:
        config: Configuration object
        likelihood_time: Time for single likelihood evaluation (seconds)
        logger: Logger object
        n_free_params: Number of free parameters (for optimization factor estimation)
    """
    # Get NUTS parameters from config
    num_samples = config.getint('NUTS', 'num_samples', fallback=2000)
    num_warmup = config.getint('NUTS', 'num_warmup', fallback=2000) 
    num_chains = config.getint('NUTS', 'num_chains', fallback=2)
    target_accept_prob = config.getfloat('NUTS', 'target_accept_prob', fallback=0.8)
    max_tree_depth = config.getint('NUTS', 'max_tree_depth', fallback=10)
    
    # Calculate total likelihood evaluations with optimizations factored in
    # Base leapfrog multiplier varies with dimensionality and target acceptance probability
    if n_free_params is not None and n_free_params > 10:
        # High-dimensional case - more conservative estimate with optimizations
        base_multiplier = 15 + (n_free_params - 10) * 0.5  # Scales with dimensionality
        
        # Optimizations reduce the effective multiplier
        optimization_factor = 1.0
        
        # Higher target acceptance prob reduces step size but improves efficiency
        if target_accept_prob > 0.9:
            optimization_factor *= 0.7  # 30% improvement from conservative stepping
        
        # Diagonal mass matrix improves efficiency
        optimization_factor *= 0.8  # 20% improvement from better preconditioning
        
        # Parameter standardization improves efficiency
        optimization_factor *= 0.75  # 25% improvement from standardized parameters
        
        # Lower tree depth reduces computational cost per sample
        if max_tree_depth < 10:
            optimization_factor *= 0.9  # 10% improvement from reduced tree depth
            
        leapfrog_multiplier = max(5, int(base_multiplier * optimization_factor))
    else:
        # Low-dimensional case - use standard estimate
        leapfrog_multiplier = 10
        
    total_evals = (num_warmup + num_samples) * num_chains * leapfrog_multiplier
    
    # Estimate total runtime
    estimated_seconds = total_evals * likelihood_time
    
    # Convert to human-readable format
    hours = int(estimated_seconds // 3600)
    minutes = int((estimated_seconds % 3600) // 60)
    seconds = int(estimated_seconds % 60)
    
    # Use logger for proper logging
    from argus.io_manager import get_argus_logger
    logger = get_argus_logger()
    
    # Note: Runtime estimation disabled for hierarchical models due to inaccuracy
    # High-dimensional NUTS sampling with hierarchical priors can have highly variable
    # trajectory lengths that make simple estimates unreliable
    
    logger.info("\n" + "="*60)
    logger.info("NUTS CONFIGURATION")
    logger.info("="*60)
    logger.info(f"NUTS Configuration:")
    logger.info(f"  - Number of samples: {num_samples}")
    logger.info(f"  - Warmup samples: {num_warmup}")
    logger.info(f"  - Number of chains: {num_chains}")
    logger.info(f"  - Target accept prob: {target_accept_prob}")
    logger.info(f"  - Max tree depth: {max_tree_depth}")
    if n_free_params is not None:
        logger.info(f"  - Free parameters: {n_free_params}")
    logger.info(f"")
    logger.info(f"Likelihood Performance:")
    logger.info(f"  - Single likelihood evaluation: {likelihood_time:.4f} seconds")
    if n_free_params is not None and n_free_params > 10:
        logger.info(f"  - High-dimensional optimizations applied")
        logger.info(f"  - Hierarchical priors reduce effective dimensionality")
    logger.info(f"")
    logger.info("Note: Runtime estimation disabled for hierarchical models.")
    logger.info("      NUTS trajectory lengths vary significantly with model complexity.")
    logger.info("      Monitor progress through the sampling progress bar.")
    logger.info("="*60)


def calculate_and_display_gradients(KF, test_params, prior_specs, logger):
    """Calculate and display likelihood gradients with respect to non-fixed parameters only.
    
    Args:
        KF: Kalman filter object
        test_params: Parameters object with test values
        prior_specs: Dictionary with prior specifications to determine which parameters are fixed
        logger: Logger object
    """
    import jax
    
    # Use logger for proper logging  
    from argus.io_manager import get_argus_logger
    logger = get_argus_logger()
    
    logger.info("\n" + "="*60)
    logger.info("LIKELIHOOD GRADIENT ANALYSIS")
    logger.info("="*60)
    
    # Create gradient function for the likelihood
    def likelihood_fn(log10_ha, log10_gamma_a, log10_gamma_p, log10_sigma_p, efac, equad):
        return bayesian_inference.log_likelihood_fn(
            KF, log10_ha, log10_gamma_a, log10_gamma_p, log10_sigma_p, efac, equad
        )
    
    # Calculate gradients
    grad_fn = jax.grad(likelihood_fn, argnums=(0, 1, 2, 3, 4, 5))
    
    # Extract test parameter values
    log10_ha_test = jax.numpy.log10(test_params.ha)
    log10_gamma_a_test = jax.numpy.log10(test_params.γa)
    log10_gamma_p_test = jax.numpy.log10(test_params.γp)
    log10_sigma_p_test = jax.numpy.log10(test_params.σp)
    efac_test = test_params.EFAC
    equad_test = test_params.EQUAD
    
    logger.info("Computing gradients at test parameter values...")
    logger.info(f"Test values:")
    logger.info(f"  - log10(h_a): {float(log10_ha_test):.2f}")
    logger.info(f"  - log10(γ_a): {float(log10_gamma_a_test):.2e}")
    logger.info(f"  - log10(γ_p): [{float(jax.numpy.min(log10_gamma_p_test)):.2f}, {float(jax.numpy.max(log10_gamma_p_test)):.2f}] (min, max)")
    logger.info(f"  - log10(σ_p): [{float(jax.numpy.min(log10_sigma_p_test)):.2f}, {float(jax.numpy.max(log10_sigma_p_test)):.2f}] (min, max)")
    logger.info("")
    
    # Compute gradients
    grads = grad_fn(log10_ha_test, log10_gamma_a_test, log10_gamma_p_test, 
                   log10_sigma_p_test, efac_test, equad_test)
    
    grad_log10_ha, grad_log10_gamma_a, grad_log10_gamma_p, grad_log10_sigma_p, grad_efac, grad_equad = grads
    
    # Determine which parameters are not fixed (i.e., being sampled)
    import tensorflow_probability.substrates.jax as tfp
    tfpd = tfp.distributions
    
    # Check if each parameter is being sampled (not fixed)
    log10_ha_sampled = (prior_specs['log10_ha_transform_params'] is not None or 
                       isinstance(prior_specs['log10_ha_spec'], tfpd.Distribution))
    log10_gamma_a_sampled = isinstance(prior_specs['log10_gamma_a_spec'], tfpd.Distribution)
    psr_noise_sampled = isinstance(prior_specs['log10_gamma_p_spec'], tfpd.Distribution)
    efac_equad_sampled = isinstance(prior_specs['efac_spec'], tfpd.Distribution)
    
    logger.info("Gradient Results (Non-Fixed Parameters Only):")
    
    # Only display gradients for parameters that are actually being sampled
    has_scalar_params = False
    has_vector_params = False
    
    # Scalar parameters section
    if log10_ha_sampled or log10_gamma_a_sampled:
        has_scalar_params = True
        logger.info("--- Scalar Parameters ---")
        
        if log10_ha_sampled:
            if prior_specs['log10_ha_transform_params'] is not None:
                # For reparameterized log10_ha, show the gradient w.r.t. log10_ha_prime
                transform_params = prior_specs['log10_ha_transform_params']
                grad_log10_ha_prime = grad_log10_ha * transform_params['std']
                logger.info(f"∂ℒ/∂log10_ha_prime: {float(grad_log10_ha_prime):.2e} (reparameterized)")
                logger.info(f"∂ℒ/∂log10(h_a): {float(grad_log10_ha):.2e} (transformed)")
            else:
                logger.info(f"∂ℒ/∂log10(h_a): {float(grad_log10_ha):.2e}")
        
        if log10_gamma_a_sampled:
            logger.info(f"∂ℒ/∂log10(γ_a): {float(grad_log10_gamma_a):.2e}")
    
    # Vector parameters section  
    if psr_noise_sampled or efac_equad_sampled:
        has_vector_params = True
        if has_scalar_params:
            logger.info("")
        logger.info("--- Vector Parameters ---")
        
        if psr_noise_sampled:
            logger.info(f"∂ℒ/∂log10(γ_p):")
            logger.info(f"  - L2 norm: {float(jax.numpy.linalg.norm(grad_log10_gamma_p)):.2e}")
            logger.info(f"  - Range: [{float(jax.numpy.min(grad_log10_gamma_p)):.2e}, {float(jax.numpy.max(grad_log10_gamma_p)):.2e}]")
            logger.info(f"  - Mean: {float(jax.numpy.mean(grad_log10_gamma_p)):.2e}")
            
            logger.info(f"∂ℒ/∂log10(σ_p):")
            logger.info(f"  - L2 norm: {float(jax.numpy.linalg.norm(grad_log10_sigma_p)):.2e}")
            logger.info(f"  - Range: [{float(jax.numpy.min(grad_log10_sigma_p)):.2e}, {float(jax.numpy.max(grad_log10_sigma_p)):.2e}]")
            logger.info(f"  - Mean: {float(jax.numpy.mean(grad_log10_sigma_p)):.2e}")
        
        if efac_equad_sampled:
            logger.info(f"∂ℒ/∂EFAC:")
            logger.info(f"  - L2 norm: {float(jax.numpy.linalg.norm(grad_efac)):.2e}")
            logger.info(f"  - Range: [{float(jax.numpy.min(grad_efac)):.2e}, {float(jax.numpy.max(grad_efac)):.2e}]")
            logger.info(f"  - Mean: {float(jax.numpy.mean(grad_efac)):.2e}")
            
            logger.info(f"∂ℒ/∂EQUAD:")
            logger.info(f"  - L2 norm: {float(jax.numpy.linalg.norm(grad_equad)):.2e}")
            logger.info(f"  - Range: [{float(jax.numpy.min(grad_equad)):.2e}, {float(jax.numpy.max(grad_equad)):.2e}]")
            logger.info(f"  - Mean: {float(jax.numpy.mean(grad_equad)):.2e}")
    
    # Special message if no parameters are being sampled
    if not (log10_ha_sampled or log10_gamma_a_sampled or psr_noise_sampled or efac_equad_sampled):
        logger.info("All parameters are FIXED - no gradients to display for NUTS sampling")
        logger.info("(Gradients are only relevant for parameters being sampled)")
    
    # Calculate total gradient magnitude (only for sampled parameters)
    total_grad_terms = []
    
    if log10_ha_sampled:
        if prior_specs['log10_ha_transform_params'] is not None:
            # Use the reparameterized gradient for total calculation
            transform_params = prior_specs['log10_ha_transform_params']
            grad_log10_ha_prime = grad_log10_ha * transform_params['std']
            total_grad_terms.append(grad_log10_ha_prime**2)
        else:
            total_grad_terms.append(grad_log10_ha**2)
    
    if log10_gamma_a_sampled:
        total_grad_terms.append(grad_log10_gamma_a**2)
    
    if psr_noise_sampled:
        total_grad_terms.append(jax.numpy.sum(grad_log10_gamma_p**2))
        total_grad_terms.append(jax.numpy.sum(grad_log10_sigma_p**2))
    
    if efac_equad_sampled:
        total_grad_terms.append(jax.numpy.sum(grad_efac**2))
        total_grad_terms.append(jax.numpy.sum(grad_equad**2))
    
    if total_grad_terms:
        total_grad_norm = jax.numpy.sqrt(sum(total_grad_terms))
        
        logger.info(f"\nTotal gradient L2 norm (sampled parameters only): {float(total_grad_norm):.2e}")
        
        # Provide interpretation
        logger.info("\nInterpretation:")
        if float(total_grad_norm) < 1e-6:
            logger.info("  - Very small gradients: May indicate a flat likelihood region")
            logger.info("    or that test parameters are near an optimum")
        elif float(total_grad_norm) > 1e3:
            logger.info("  - Large gradients: Likelihood is changing rapidly")
            logger.info("    Consider parameter scaling or different test values")
        else:
            logger.info("  - Moderate gradients: Normal range for MCMC sampling")
    else:
        logger.info("\nNo sampled parameters - gradient norm not applicable")
    
    logger.info("="*60)


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
    
    # Get prior model specifications and display them
    n_pulsars = len(pulsar_data['metadata'])
    prior_specs = bayesian_inference.get_prior_model_specs(
        config, n_pulsars, sigma_p_array, gamma_p_array, efac_array, equad_array
    )
    
    # Display prior summary
    bayesian_inference.display_prior_summary(prior_specs, n_pulsars, logger)
    
    # Test likelihood performance and get timing
    logger.info("Performing likelihood performance test for runtime estimation...")
    test_likelihood_performance(KF, config, logger)
    
    # Get likelihood timing from the performance test
    # We'll extract this from the test function by modifying it slightly
    # For now, run a quick timing test here
    import time
    γa_test = 1e-9 
    ha_test = 1e-15
    test_params = bayesian_inference.Parameters(
        log10_gamma_a=jax.numpy.log10(γa_test),
        γa=γa_test,
        ha=ha_test,
        γp=gamma_p_array,
        σp=sigma_p_array,
        EFAC=efac_array,
        EQUAD=equad_array
    )
    
    # Quick timing for runtime estimation (JIT-compiled version)
    start_time = time.perf_counter()
    _ = KF.get_likelihood(test_params)
    _.block_until_ready()
    end_time = time.perf_counter()
    likelihood_time = end_time - start_time
    
    # Count free parameters for runtime estimation
    from argus.bayesian_inference import count_free_parameters
    n_free_params = count_free_parameters(prior_specs, len(pulsar_data['metadata']))
    
    # Estimate and display runtime
    estimate_runtime(config, likelihood_time, logger, n_free_params)
    
    # Calculate and display gradients
    calculate_and_display_gradients(KF, test_params, prior_specs, logger)
    
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