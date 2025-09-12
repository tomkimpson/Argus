"""Bayesian inference module for pulsar timing array analysis.

This module provides the main interface for performing Bayesian parameter estimation
on pulsar timing array data. It serves as the orchestration layer that coordinates
prior model specifications, parameter sampling, and NUTS inference.

The module handles parameters like:
- Gravitational wave background amplitude (ha) and spectral index (γa)
- Pulsar-specific red noise parameters (γp, σp)
- White noise parameters (EFAC, EQUAD)

The implementation uses the Hellings-Downs correlation pattern for the 
gravitational wave background and models pulsar red noise as an 
Ornstein-Uhlenbeck process.
"""

import jax
import jax.numpy as jnp
from flax import struct
import tensorflow_probability.substrates.jax as tfp

# Import from the new modular structure

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
        log_or_print("log10(h_a): REPARAMETERIZED for better NUTS sampling")
        log_or_print("  - Sampling: log10_ha_prime ~ N(0, 1)")
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
        log_or_print("log10(γ_p): HIERARCHICAL modeling")
        log_or_print(f"  - Population mean: Uniform({float(mean_spec.low):.1f}, {float(mean_spec.high):.1f})")
        log_or_print(f"  - Population std: Uniform({float(std_spec.low):.1f}, {float(std_spec.high):.1f})")
        log_or_print("  - Individual pulsars: Normal(population_mean, population_std)")
    elif isinstance(gamma_p_spec, tfpd.Distribution):
        log_or_print(f"log10(γ_p): Uniform({float(gamma_p_spec.low[0]):.1f}, {float(gamma_p_spec.high[0]):.1f}) for each pulsar")
    elif gamma_p_spec is not None:
        if hasattr(gamma_p_spec, '__len__') and len(gamma_p_spec) > 1:
            log_or_print(f"log10(γ_p): FIXED at individual values (range: {float(jnp.min(gamma_p_spec)):.2f} to {float(jnp.max(gamma_p_spec)):.2f})")
        else:
            log_or_print(f"log10(γ_p): FIXED at {float(gamma_p_spec):.2f}")
    else:
        log_or_print("log10(γ_p): ERROR - None value encountered")
    
    # log10_sigma_p parameter - check for hierarchical modeling
    sigma_p_spec = prior_specs['log10_sigma_p_spec']
    if hierarchical_specs and hierarchical_specs.get('log_ratio_parameterization', False):
        # Check if the required specs exist before accessing them
        if 'log10_ratio_mean_spec' in hierarchical_specs and 'log10_ratio_std_spec' in hierarchical_specs:
            # Log-ratio parameterization case
            mean_spec = hierarchical_specs['log10_ratio_mean_spec']
            std_spec = hierarchical_specs['log10_ratio_std_spec']
            log_or_print("log10(σ_p): LOG-RATIO parameterization")
            log_or_print("  - log10(σ_p) = log10(γ_p) + log10(ratio)")
            log_or_print(f"  - Ratio mean: Uniform({float(mean_spec.low):.1f}, {float(mean_spec.high):.1f})")
            log_or_print(f"  - Ratio std: Uniform({float(std_spec.low):.1f}, {float(std_spec.high):.1f})")
            log_or_print("  - Individual ratios: Normal(ratio_mean, ratio_std)")
        else:
            # Fallback: hierarchical settings enabled but specs not created (likely due to fixed params)
            log_or_print("log10(σ_p): FIXED (hierarchical settings detected but overridden by fixed parameters)")
    elif isinstance(sigma_p_spec, tfpd.Distribution):
        log_or_print(f"log10(σ_p): Uniform({float(sigma_p_spec.low[0]):.1f}, {float(sigma_p_spec.high[0]):.1f}) for each pulsar")
    elif sigma_p_spec is not None:
        if hasattr(sigma_p_spec, '__len__') and len(sigma_p_spec) > 1:
            log_or_print(f"log10(σ_p): FIXED at individual values (range: {float(jnp.min(sigma_p_spec)):.2f} to {float(jnp.max(sigma_p_spec)):.2f})")
        else:
            log_or_print(f"log10(σ_p): FIXED at {float(sigma_p_spec):.2f}")
    else:
        log_or_print("log10(σ_p): ERROR - None value encountered")
    
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
        log_or_print("EFAC: ERROR - None value encountered")
    
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
        log_or_print("EQUAD: ERROR - None value encountered")
    
    log_or_print("="*60)







