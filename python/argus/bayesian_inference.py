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
import jax.random as random
import numpyro
import arviz as az
from numpyro.infer import MCMC, NUTS
import time

from .parameter_sampling import (
    sample_gw_parameters,
    sample_pulsar_noise_parameters,
    sample_measurement_noise_parameters,
    count_free_parameters,
)

jax.config.update("jax_enable_x64", True)
tfpd = tfp.distributions


@struct.dataclass
class Parameters:
    """Define a struct to store the parameters of the Kalman filter model."""

    # GW parameters
    log10_gamma_a: float  # log10(γa) - log10 of GW spectral index
    γa: float  # s⁻¹ - GW spectral index (derived from log10_gamma_a)
    ha: float  # GWB amplitude

    # Pulsar parameters for the OU process
    γp: jnp.ndarray  # Pulsar-specific gamma values
    σp: jnp.ndarray  # Pulsar-specific sigma values

    # Measurement noise parameters
    EFAC: jnp.ndarray  # Error factors
    EQUAD: jnp.ndarray  # Extra quadrature noise


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

    log_or_print("\n" + "=" * 60)
    log_or_print("PRIOR SPECIFICATIONS SUMMARY")
    log_or_print("=" * 60)

    # GW background parameters
    log_or_print("\n--- Gravitational Wave Background Parameters ---")

    # log10_ha parameter
    ha_spec = prior_specs["log10_ha_spec"]
    ha_transform = prior_specs["log10_ha_transform_params"]

    if ha_transform is not None:
        # Reparameterized case
        log_or_print("log10(h_a): REPARAMETERIZED for better NUTS sampling")
        log_or_print("  - Sampling: log10_ha_prime ~ N(0, 1)")
        log_or_print(
            f"  - Transform: log10_ha = {ha_transform['mean']:.2f} + log10_ha_prime * {ha_transform['std']:.3f}"
        )
        log_or_print(
            f"  - Equivalent to: Uniform({ha_transform['min']:.1f}, {ha_transform['max']:.1f})"
        )
    elif isinstance(ha_spec, tfpd.Distribution):
        # Direct distribution case (backward compatibility)
        if hasattr(ha_spec, "low"):
            log_or_print(
                f"log10(h_a): Uniform({float(ha_spec.low):.1f}, {float(ha_spec.high):.1f})"
            )
        else:
            log_or_print(f"log10(h_a): {type(ha_spec).__name__} distribution")
    else:
        # Fixed value case
        log_or_print(f"log10(h_a): FIXED at {float(ha_spec):.1f}")

    # log10_gamma_a parameter
    log10_gamma_spec = prior_specs["log10_gamma_a_spec"]
    if isinstance(log10_gamma_spec, tfpd.Distribution):
        log_or_print(
            f"log10(γ_a): Uniform({float(log10_gamma_spec.low):.1f}, {float(log10_gamma_spec.high):.1f})"
        )
    else:
        log_or_print(f"log10(γ_a): FIXED at {float(log10_gamma_spec):.1f}")

    # Pulsar red noise parameters
    log_or_print(f"\n--- Pulsar Red Noise Parameters ({n_pulsars} pulsars) ---")

    # log10_gamma_p parameter - check for hierarchical modeling
    gamma_p_spec = prior_specs["log10_gamma_p_spec"]
    hierarchical_specs = prior_specs.get("hierarchical_specs")

    if hierarchical_specs and hierarchical_specs.get("hierarchical_noise", False):
        # Hierarchical modeling case
        mean_spec = hierarchical_specs["log10_gamma_p_mean_spec"]
        std_spec = hierarchical_specs["log10_gamma_p_std_spec"]
        log_or_print("log10(γ_p): HIERARCHICAL modeling")
        log_or_print(
            f"  - Population mean: Uniform({float(mean_spec.low):.1f}, {float(mean_spec.high):.1f})"
        )
        log_or_print(
            f"  - Population std: Uniform({float(std_spec.low):.1f}, {float(std_spec.high):.1f})"
        )
        log_or_print("  - Individual pulsars: Normal(population_mean, population_std)")
    elif isinstance(gamma_p_spec, tfpd.Distribution):
        log_or_print(
            f"log10(γ_p): Uniform({float(gamma_p_spec.low[0]):.1f}, {float(gamma_p_spec.high[0]):.1f}) for each pulsar"
        )
    elif gamma_p_spec is not None:
        if hasattr(gamma_p_spec, "__len__") and len(gamma_p_spec) > 1:
            log_or_print(
                f"log10(γ_p): FIXED at individual values (range: {float(jnp.min(gamma_p_spec)):.2f} to {float(jnp.max(gamma_p_spec)):.2f})"
            )
        else:
            log_or_print(f"log10(γ_p): FIXED at {float(gamma_p_spec):.2f}")
    else:
        log_or_print("log10(γ_p): ERROR - None value encountered")

    # log10_sigma_p parameter - check for hierarchical modeling
    sigma_p_spec = prior_specs["log10_sigma_p_spec"]
    if hierarchical_specs and hierarchical_specs.get(
        "log_ratio_parameterization", False
    ):
        # Check if the required specs exist before accessing them
        if (
            "log10_ratio_mean_spec" in hierarchical_specs
            and "log10_ratio_std_spec" in hierarchical_specs
        ):
            # Log-ratio parameterization case
            mean_spec = hierarchical_specs["log10_ratio_mean_spec"]
            std_spec = hierarchical_specs["log10_ratio_std_spec"]
            log_or_print("log10(σ_p): LOG-RATIO parameterization")
            log_or_print("  - log10(σ_p) = log10(γ_p) + log10(ratio)")
            log_or_print(
                f"  - Ratio mean: Uniform({float(mean_spec.low):.1f}, {float(mean_spec.high):.1f})"
            )
            log_or_print(
                f"  - Ratio std: Uniform({float(std_spec.low):.1f}, {float(std_spec.high):.1f})"
            )
            log_or_print("  - Individual ratios: Normal(ratio_mean, ratio_std)")
        else:
            # Fallback: hierarchical settings enabled but specs not created (likely due to fixed params)
            log_or_print(
                "log10(σ_p): FIXED (hierarchical settings detected but overridden by fixed parameters)"
            )
    elif isinstance(sigma_p_spec, tfpd.Distribution):
        log_or_print(
            f"log10(σ_p): Uniform({float(sigma_p_spec.low[0]):.1f}, {float(sigma_p_spec.high[0]):.1f}) for each pulsar"
        )
    elif sigma_p_spec is not None:
        if hasattr(sigma_p_spec, "__len__") and len(sigma_p_spec) > 1:
            log_or_print(
                f"log10(σ_p): FIXED at individual values (range: {float(jnp.min(sigma_p_spec)):.2f} to {float(jnp.max(sigma_p_spec)):.2f})"
            )
        else:
            log_or_print(f"log10(σ_p): FIXED at {float(sigma_p_spec):.2f}")
    else:
        log_or_print("log10(σ_p): ERROR - None value encountered")

    # Measurement noise parameters
    log_or_print(f"\n--- Measurement Noise Parameters ({n_pulsars} pulsars) ---")

    # EFAC parameter
    efac_spec = prior_specs["efac_spec"]
    if isinstance(efac_spec, tfpd.Distribution):
        log_or_print(
            f"EFAC: Uniform({float(efac_spec.low[0]):.2f}, {float(efac_spec.high[0]):.2f}) for each pulsar"
        )
    elif efac_spec is not None:
        if hasattr(efac_spec, "__len__") and len(efac_spec) > 1:
            log_or_print(
                f"EFAC: FIXED at individual values (range: {float(jnp.min(efac_spec)):.3f} to {float(jnp.max(efac_spec)):.3f})"
            )
        else:
            log_or_print(f"EFAC: FIXED at {float(efac_spec):.3f}")
    else:
        log_or_print("EFAC: ERROR - None value encountered")

    # EQUAD parameter
    equad_spec = prior_specs["equad_spec"]
    if isinstance(equad_spec, dict) and equad_spec.get("use_log10", False):
        # log10(EQUAD) parameterization
        log10_equad_spec = equad_spec["log10_equad_spec"]
        log10_low = float(log10_equad_spec.low[0])
        log10_high = float(log10_equad_spec.high[0])
        log_or_print(
            f"EQUAD: log10(EQUAD) ~ Uniform({log10_low:.1f}, {log10_high:.1f}) for each pulsar"
        )
    elif isinstance(equad_spec, tfpd.Distribution):
        # Regular uniform distribution
        log_or_print(
            f"EQUAD: Uniform({float(equad_spec.low[0]):.2e}, {float(equad_spec.high[0]):.2e}) for each pulsar"
        )
    elif equad_spec is not None:
        if hasattr(equad_spec, "__len__") and len(equad_spec) > 1:
            log_or_print(
                f"EQUAD: FIXED at individual values (range: {float(jnp.min(equad_spec)):.2e} to {float(jnp.max(equad_spec)):.2e})"
            )
        else:
            log_or_print(f"EQUAD: FIXED at {float(equad_spec):.2e}")
    else:
        log_or_print("EQUAD: ERROR - None value encountered")

    log_or_print("=" * 60)


def log_likelihood_fn(
    kalman_filter, log10_ha, log10_gamma_a, log10_γp, log10_σp, efac, equad
):
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
    ha = 10.0**log10_ha
    γa = 10.0**log10_gamma_a
    γp = 10.0**log10_γp
    σp = 10.0**log10_σp

    params = Parameters(
        log10_gamma_a=log10_gamma_a, γa=γa, ha=ha, γp=γp, σp=σp, EFAC=efac, EQUAD=equad
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
    log_likelihood = log_likelihood_fn(
        kalman_filter, log10_ha, log10_gamma_a, log10_γp, log10_σp, efac, equad
    )

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
    target_accept_prob = config.getfloat(
        "NUTS", "target_accept_prob", fallback=0.95
    )  # More conservative for high-dim
    max_tree_depth = config.getint("NUTS", "max_tree_depth", fallback=10)
    dense_mass = config.getboolean("NUTS", "dense_mass", fallback=False)

    # Handle step_size - only set if explicitly provided in config
    nuts_kwargs = {
        "target_accept_prob": target_accept_prob,
        "max_tree_depth": max_tree_depth,
        "adapt_step_size": True,
        "adapt_mass_matrix": True,
        "dense_mass": dense_mass,
    }

    # Only add step_size if explicitly set in config
    if config.has_option("NUTS", "step_size"):
        step_size = config.getfloat("NUTS", "step_size")
        nuts_kwargs["step_size"] = step_size
        print(f"Using custom step size: {step_size}")

    # Count total number of free parameters for diagnostics
    total_params = count_free_parameters(prior_specs, n_pulsars)

    nuts_info = {
        "total_params": total_params,
        "target_accept_prob": target_accept_prob,
        "max_tree_depth": max_tree_depth,
        "dense_mass": dense_mass,
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
    n_pulsars = len([spec for spec in prior_specs.keys() if "pulsar" in spec])
    num_samples = config.getint("NUTS", "num_samples", fallback=2000)
    num_warmup = config.getint("NUTS", "num_warmup", fallback=2000)
    num_chains = config.getint("NUTS", "num_chains", fallback=2)

    print("Running NumPyro NUTS inference...")
    print(
        f"NUTS parameters: {num_samples} samples, {num_warmup} warmup, {num_chains} chains"
    )
    print(
        f"Target accept prob: {nuts_info['target_accept_prob']} (optimized for high-dimensional sampling)"
    )
    print(f"Dense mass matrix: {nuts_info['dense_mass']}")
    print(f"Max tree depth: {nuts_info['max_tree_depth']}")
    print(f"Total free parameters: {nuts_info['total_params']}")

    # Check if hierarchical modeling is enabled
    hierarchical_specs = prior_specs.get("hierarchical_specs")
    if hierarchical_specs:
        hier_gamma = hierarchical_specs.get("hierarchical_noise", False)
        log_ratio = hierarchical_specs.get("log_ratio_parameterization", False)
        if hier_gamma or log_ratio:
            print("Advanced modeling enabled for pulsar noise parameters")
            if hier_gamma and log_ratio:
                print("γp hierarchical + σp via log-ratio parameterization")
                print(
                    f"Effective dimensionality: 4 hyperparameters + {2*n_pulsars} constrained parameters"
                )
                print("σp = γp + ratio (reduces parameter correlations)")
            elif hier_gamma:
                print("γp uses hierarchical priors, σp fixed")
                print(
                    f"Effective dimensionality: 2 hyperparameters + {n_pulsars} constrained parameters"
                )
            elif log_ratio:
                print("σp via log-ratio parameterization, γp independent")
                print(
                    f"Effective dimensionality: 2 hyperparameters + {2*n_pulsars} parameters"
                )

    if nuts_info["total_params"] > 10:
        print(
            "High-dimensional parameter space detected - using aggressive NUTS tuning"
        )


def run_nuts_sampling(
    kalman_filter,
    config,
    n_pulsars,
    sigma_p_array,
    gamma_p_array,
    efac_array,
    equad_array,
):
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
    from .prior_models import get_prior_model_specs

    # Get prior model distributions
    prior_specs = get_prior_model_specs(
        config, n_pulsars, sigma_p_array, gamma_p_array, efac_array, equad_array
    )

    # Get NUTS parameters from config
    num_samples = config.getint("NUTS", "num_samples", fallback=2000)
    num_warmup = config.getint("NUTS", "num_warmup", fallback=2000)
    num_chains = config.getint("NUTS", "num_chains", fallback=2)

    # Set up NUTS kernel
    kernel, nuts_info = setup_nuts_kernel(prior_specs, n_pulsars, config)

    # Print diagnostics
    print_nuts_diagnostics(prior_specs, nuts_info, config)

    # Create the actual model function bound to the Kalman filter
    def bound_model():
        return numpyro_model(kalman_filter, prior_specs, n_pulsars)

    # Create NUTS kernel with bound model
    kernel = NUTS(
        bound_model,
        **{
            "target_accept_prob": nuts_info.get("target_accept_prob", 0.95),
            "max_tree_depth": nuts_info.get("max_tree_depth", 10),
            "adapt_step_size": True,
            "adapt_mass_matrix": True,
            "dense_mass": nuts_info.get("dense_mass", False),
        },
    )

    # Set up MCMC sampler
    sampler = MCMC(
        kernel,
        num_samples=num_samples,
        num_warmup=num_warmup,
        num_chains=num_chains,
        progress_bar=True,
    )

    # Run sampling
    rng_key = random.PRNGKey(42)  # Fixed seed for reproducibility
    sampler.run(rng_key)

    # Print summary
    sampler.print_summary()

    # Convert to ArviZ format
    inf_data = az.from_numpyro(sampler)

    return inf_data


def test_likelihood_performance(kalman_filter, config, n_pulsars, logger):
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
    n_pulsars : int
        Number of pulsars
    logger : logging.Logger
        Logger object

    Returns
    -------
    float
        The computed log likelihood value
    """
    from argus.utils import get_noise_parameters

    logger.info("=== Likelihood Performance Test ===")
    logger.info("Testing likelihood evaluation with known parameter values...")

    # Get noise parameters using the common function
    efac_array, equad_array, sigma_p_array, gamma_p_array = get_noise_parameters(config)

    # Set test parameter values
    γa_test = 1e-9
    ha_test = 1e-15

    # If noise parameters are None, create test arrays with reasonable values
    if gamma_p_array is None:
        gamma_p_array = jnp.full(n_pulsars, 1e-8)  # Default gamma_p test value
    if sigma_p_array is None:
        sigma_p_array = jnp.full(n_pulsars, 1e-15)  # Default sigma_p test value
    if efac_array is None:
        efac_array = jnp.ones(n_pulsars)  # Default EFAC test value
    if equad_array is None:
        equad_array = jnp.full(n_pulsars, 1e-7)  # Default EQUAD test value

    # Create parameter object
    test_params = Parameters(
        log10_gamma_a=jnp.log10(γa_test),
        γa=γa_test,
        ha=ha_test,
        γp=gamma_p_array,
        σp=sigma_p_array,
        EFAC=efac_array,
        EQUAD=equad_array,
    )

    logger.info(f"Test parameters: γa={γa_test}, ha={ha_test}")
    logger.info(f"Number of pulsars: {n_pulsars}")

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
    logger.info(
        f"Likelihood evaluation completed in {duration1:.4f} seconds the first time"
    )
    logger.info(
        f"Likelihood evaluation completed in {duration2:.4f} seconds the second time"
    )
    logger.info(f"Log likelihood value: {float(log_likelihood)}")
    logger.info("=== End Likelihood Performance Test ===")

    return float(log_likelihood)
