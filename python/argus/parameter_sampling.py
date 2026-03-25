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
    # Handle log10_ha: either fixed value or reparameterized sampling
    if prior_specs["log10_ha_transform_params"] is not None:
        # Sample log10_ha_prime ~ N(0,1) and transform to log10_ha for efficient NUTS sampling
        transform_params = prior_specs["log10_ha_transform_params"]
        log10_ha_prime = numpyro.sample("log10_ha_prime", dist.Normal(0.0, 1.0))
        log10_ha = numpyro.deterministic(
            "log10_ha",
            transform_params["mean"] + log10_ha_prime * transform_params["std"],
        )
    else:
        # Fixed value (delta prior)
        log10_ha = numpyro.deterministic("log10_ha", prior_specs["log10_ha_spec"])

    # Handle log10_gamma_a: either fixed value or reparameterized sampling
    if isinstance(prior_specs["log10_gamma_a_spec"], tfpd.Distribution):
        # Use reparameterization for efficient NUTS sampling
        low = prior_specs["log10_gamma_a_spec"].low
        high = prior_specs["log10_gamma_a_spec"].high
        mean = (low + high) / 2.0
        std = (high - low) / 6.0  # 3-sigma rule

        log10_gamma_a_prime = numpyro.sample(
            "log10_gamma_a_prime", dist.Normal(0.0, 1.0)
        )
        log10_gamma_a = numpyro.deterministic(
            "log10_gamma_a", mean + log10_gamma_a_prime * std
        )
        γa = numpyro.deterministic("γa", 10.0**log10_gamma_a)
    else:
        # Fixed value
        log10_gamma_a = numpyro.deterministic(
            "log10_gamma_a", prior_specs["log10_gamma_a_spec"]
        )
        γa = numpyro.deterministic("γa", 10.0**log10_gamma_a)

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
    log10_gamma_p_mean_raw = numpyro.sample(
        "log10_gamma_p_mean_raw", dist.Normal(0.0, 1.0)
    )
    log10_gamma_p_std_raw = numpyro.sample(
        "log10_gamma_p_std_raw", dist.Normal(0.0, 1.0)
    )

    # Transform to appropriate ranges with balanced gradients
    mean_low = hierarchical_specs["log10_gamma_p_mean_spec"].low
    mean_high = hierarchical_specs["log10_gamma_p_mean_spec"].high
    std_low = hierarchical_specs["log10_gamma_p_std_spec"].low
    std_high = hierarchical_specs["log10_gamma_p_std_spec"].high

    # Apply gradient-balanced transforms
    log10_gamma_p_mean = numpyro.deterministic(
        "log10_gamma_p_mean",
        (mean_low + mean_high) / 2.0
        + log10_gamma_p_mean_raw * (mean_high - mean_low) / 6.0,
    )
    log10_gamma_p_std = numpyro.deterministic(
        "log10_gamma_p_std",
        (std_low + std_high) / 2.0 + log10_gamma_p_std_raw * (std_high - std_low) / 6.0,
    )

    # Sample individual pulsar parameters with scaled gradients
    log10_γp_raw = numpyro.sample(
        "log10_γp_raw", dist.Normal(jnp.zeros(n_pulsars), jnp.ones(n_pulsars))
    )
    log10_γp = numpyro.deterministic(
        "log10_γp",
        log10_gamma_p_mean + log10_γp_raw * log10_gamma_p_std / jnp.sqrt(n_pulsars),
    )

    return log10_γp


def sample_reparameterized_parameters(prior_spec, param_name, n_pulsars):
    """Sample parameters using reparameterization for efficient NUTS sampling.

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
    # Reparameterize uniform distribution using Normal(0,1) + affine transformation
    low = prior_spec.low
    high = prior_spec.high
    mean = (low + high) / 2.0
    std = (high - low) / 6.0  # 3-sigma rule

    param_standardized = numpyro.sample(
        f"{param_name}_standardized",
        dist.Normal(jnp.zeros(n_pulsars), jnp.ones(n_pulsars) / jnp.sqrt(n_pulsars)),
    )
    param_values = numpyro.deterministic(param_name, mean + param_standardized * std)

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
    mean_low = hierarchical_specs["log10_ratio_mean_spec"].low
    mean_high = hierarchical_specs["log10_ratio_mean_spec"].high
    std_low = hierarchical_specs["log10_ratio_std_spec"].low
    std_high = hierarchical_specs["log10_ratio_std_spec"].high

    # Apply gradient-balanced transforms
    log10_ratio_mean = numpyro.deterministic(
        "log10_ratio_mean",
        (mean_low + mean_high) / 2.0
        + log10_ratio_mean_raw * (mean_high - mean_low) / 6.0,
    )
    log10_ratio_std = numpyro.deterministic(
        "log10_ratio_std",
        (std_low + std_high) / 2.0 + log10_ratio_std_raw * (std_high - std_low) / 6.0,
    )

    # Sample individual ratio parameters with scaled gradients
    log10_ratio_raw = numpyro.sample(
        "log10_ratio_raw", dist.Normal(jnp.zeros(n_pulsars), jnp.ones(n_pulsars))
    )
    log10_ratio = numpyro.deterministic(
        "log10_ratio",
        log10_ratio_mean + log10_ratio_raw * log10_ratio_std / jnp.sqrt(n_pulsars),
    )

    # Derive log10_σp deterministically from γp + ratio
    log10_σp = numpyro.deterministic("log10_σp", log10_γp + log10_ratio)

    return log10_σp


def sample_pulsar_noise_parameters(prior_specs, n_pulsars):
    """Sample pulsar red noise parameters using hierarchical modeling.

    Always uses hierarchical modeling for gamma_p and log-ratio parameterization
    for sigma_p unless parameters are explicitly fixed via injection.

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
    hierarchical_specs = prior_specs.get("hierarchical_specs")

    # Handle log10_gamma_p - either hierarchical or fixed
    if isinstance(prior_specs["log10_gamma_p_spec"], tfpd.Distribution):
        # Fallback for backwards compatibility - shouldn't occur in new setup
        log10_γp = sample_reparameterized_parameters(
            prior_specs["log10_gamma_p_spec"], "log10_γp", n_pulsars
        )
    elif prior_specs["log10_gamma_p_spec"] is not None:
        # Fixed value (from injections)
        log10_γp = numpyro.deterministic("log10_γp", prior_specs["log10_gamma_p_spec"])
    else:
        # Always use hierarchical modeling
        log10_γp = sample_hierarchical_gamma_parameters(hierarchical_specs, n_pulsars)

    # Handle log10_sigma_p - either log-ratio or fixed
    if isinstance(prior_specs["log10_sigma_p_spec"], tfpd.Distribution):
        # Fallback for backwards compatibility - shouldn't occur in new setup
        log10_σp = sample_reparameterized_parameters(
            prior_specs["log10_sigma_p_spec"], "log10_σp", n_pulsars
        )
    elif prior_specs["log10_sigma_p_spec"] is not None:
        # Fixed value (from injections)
        log10_σp = numpyro.deterministic("log10_σp", prior_specs["log10_sigma_p_spec"])
    else:
        # Always use log-ratio parameterization
        log10_σp = sample_log_ratio_parameters(hierarchical_specs, log10_γp, n_pulsars)

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
    # Handle EFAC: either fixed value or reparameterized sampling
    if isinstance(prior_specs["efac_spec"], tfpd.Distribution):
        # Use reparameterization for efficient NUTS sampling
        low = prior_specs["efac_spec"].low
        high = prior_specs["efac_spec"].high
        mean = (low + high) / 2.0
        std = (high - low) / 6.0  # 3-sigma rule

        efac_standardized = numpyro.sample(
            "efac_standardized", dist.Normal(jnp.zeros(n_pulsars), jnp.ones(n_pulsars))
        )
        efac = numpyro.deterministic("efac", mean + efac_standardized * std)
    else:
        # Fixed value
        efac = numpyro.deterministic("efac", prior_specs["efac_spec"])

    # Handle EQUAD: either fixed value or reparameterized sampling
    if isinstance(prior_specs["equad_spec"], dict) and prior_specs["equad_spec"].get(
        "use_log10", False
    ):
        # log10(EQUAD) parameterization with reparameterization
        log10_equad_spec = prior_specs["equad_spec"]["log10_equad_spec"]
        low = log10_equad_spec.low
        high = log10_equad_spec.high
        mean = (low + high) / 2.0
        std = (high - low) / 6.0  # 3-sigma rule

        log10_equad_prime = numpyro.sample(
            "log10_equad_prime", dist.Normal(jnp.zeros(n_pulsars), jnp.ones(n_pulsars))
        )
        log10_equad = numpyro.deterministic(
            "log10_equad", mean + log10_equad_prime * std
        )
        equad = numpyro.deterministic("equad", 10.0**log10_equad)
    elif isinstance(prior_specs["equad_spec"], tfpd.Distribution):
        # Regular distribution with reparameterization
        low = prior_specs["equad_spec"].low
        high = prior_specs["equad_spec"].high
        mean = (low + high) / 2.0
        std = (high - low) / 6.0  # 3-sigma rule

        equad_standardized = numpyro.sample(
            "equad_standardized", dist.Normal(jnp.zeros(n_pulsars), jnp.ones(n_pulsars))
        )
        equad = numpyro.deterministic("equad", mean + equad_standardized * std)
    else:
        # Fixed value
        equad = numpyro.deterministic("equad", prior_specs["equad_spec"])

    return efac, equad


def sample_cw_parameters(prior_specs):
    """Sample continuous wave source parameters from their priors.

    Parameters
    ----------
    prior_specs : dict
        Prior distributions dictionary containing CW parameter specs.

    Returns
    -------
    tuple
        (log10_h0, alpha_gw, delta_gw, log10_f_gw, cos_iota, psi, Phi0)
    """
    cw_specs = prior_specs["cw_specs"]

    # log10_h0: strain amplitude (reparameterized)
    if cw_specs["log10_h0_transform_params"] is not None:
        tp = cw_specs["log10_h0_transform_params"]
        log10_h0_prime = numpyro.sample("log10_h0_prime", dist.Normal(0.0, 1.0))
        log10_h0 = numpyro.deterministic(
            "log10_h0", tp["mean"] + log10_h0_prime * tp["std"]
        )
    else:
        log10_h0 = numpyro.deterministic("log10_h0", cw_specs["log10_h0_spec"])

    # alpha_gw: source RA (reparameterized)
    if cw_specs["alpha_gw_transform_params"] is not None:
        tp = cw_specs["alpha_gw_transform_params"]
        alpha_gw_prime = numpyro.sample("alpha_gw_prime", dist.Normal(0.0, 1.0))
        alpha_gw = numpyro.deterministic(
            "alpha_gw", tp["mean"] + alpha_gw_prime * tp["std"]
        )
    else:
        alpha_gw = numpyro.deterministic("alpha_gw", cw_specs["alpha_gw_spec"])

    # delta_gw: source DEC via sin(delta) for isotropic sky coverage
    if cw_specs["sin_delta_gw_transform_params"] is not None:
        tp = cw_specs["sin_delta_gw_transform_params"]
        sin_delta_prime = numpyro.sample("sin_delta_gw_prime", dist.Normal(0.0, 1.0))
        sin_delta_gw = numpyro.deterministic(
            "sin_delta_gw", tp["mean"] + sin_delta_prime * tp["std"]
        )
        delta_gw = numpyro.deterministic("delta_gw", jnp.arcsin(sin_delta_gw))
    else:
        delta_gw = numpyro.deterministic("delta_gw", cw_specs["delta_gw_spec"])

    # log10_f_gw: GW frequency (reparameterized)
    if cw_specs["log10_f_gw_transform_params"] is not None:
        tp = cw_specs["log10_f_gw_transform_params"]
        log10_f_gw_prime = numpyro.sample("log10_f_gw_prime", dist.Normal(0.0, 1.0))
        log10_f_gw = numpyro.deterministic(
            "log10_f_gw", tp["mean"] + log10_f_gw_prime * tp["std"]
        )
    else:
        log10_f_gw = numpyro.deterministic("log10_f_gw", cw_specs["log10_f_gw_spec"])

    # cos_iota: inclination (reparameterized)
    if cw_specs["cos_iota_transform_params"] is not None:
        tp = cw_specs["cos_iota_transform_params"]
        cos_iota_prime = numpyro.sample("cos_iota_prime", dist.Normal(0.0, 1.0))
        cos_iota = numpyro.deterministic(
            "cos_iota", tp["mean"] + cos_iota_prime * tp["std"]
        )
    else:
        cos_iota = numpyro.deterministic("cos_iota", cw_specs["cos_iota_spec"])

    # psi: polarization angle (reparameterized)
    if cw_specs["psi_transform_params"] is not None:
        tp = cw_specs["psi_transform_params"]
        psi_prime = numpyro.sample("psi_prime", dist.Normal(0.0, 1.0))
        psi = numpyro.deterministic("psi", tp["mean"] + psi_prime * tp["std"])
    else:
        psi = numpyro.deterministic("psi", cw_specs["psi_spec"])

    # Phi0: initial phase (reparameterized)
    if cw_specs["Phi0_transform_params"] is not None:
        tp = cw_specs["Phi0_transform_params"]
        Phi0_prime = numpyro.sample("Phi0_prime", dist.Normal(0.0, 1.0))
        Phi0 = numpyro.deterministic("Phi0", tp["mean"] + Phi0_prime * tp["std"])
    else:
        Phi0 = numpyro.deterministic("Phi0", cw_specs["Phi0_spec"])

    return log10_h0, alpha_gw, delta_gw, log10_f_gw, cos_iota, psi, Phi0


def sample_chi_parameters(prior_specs, n_pulsars):
    """Sample per-pulsar phase parameters for phase-reparameterized pulsar term.

    When phase parameterization is active, samples chi^(n) ~ Uniform(0, 2pi)
    for each pulsar using Normal(0,1) -> affine reparameterization.
    When inactive, returns zeros (no pulsar term phase contribution).

    Parameters
    ----------
    prior_specs : dict
        Prior distributions dictionary containing CW parameter specs.
    n_pulsars : int
        Number of pulsars.

    Returns
    -------
    jax.Array
        Per-pulsar phase parameters, shape (n_pulsars,).
    """
    cw_specs = prior_specs.get("cw_specs", {})

    if cw_specs.get("chi_transform_params") is not None:
        tp = cw_specs["chi_transform_params"]
        chi_prime = numpyro.sample(
            "chi_prime",
            dist.Normal(jnp.zeros(n_pulsars), jnp.ones(n_pulsars)),
        )
        chi = numpyro.deterministic(
            "chi", tp["mean"] + chi_prime * tp["std"]
        )
    else:
        chi = numpyro.deterministic("chi", jnp.zeros(n_pulsars))

    return chi


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
    count = 0

    # CW parameters (if CW mode)
    cw_specs = prior_specs.get("cw_specs")
    if cw_specs is not None:
        # Count each CW parameter that has transform_params (i.e., is sampled)
        for key in [
            "log10_h0_transform_params",
            "alpha_gw_transform_params",
            "sin_delta_gw_transform_params",
            "log10_f_gw_transform_params",
            "cos_iota_transform_params",
            "psi_transform_params",
            "Phi0_transform_params",
        ]:
            if cw_specs.get(key) is not None:
                count += 1
        # Per-pulsar phase parameters (phase reparameterization)
        if cw_specs.get("chi_transform_params") is not None:
            count += n_pulsars
    else:
        # GWB parameters
        # GW amplitude parameter - free if reparameterization is used
        if prior_specs["log10_ha_transform_params"] is not None:
            count += 1

        # GW spectral index parameter - free if it's a distribution (not fixed)
        if isinstance(prior_specs["log10_gamma_a_spec"], tfpd.Distribution):
            count += 1

    # Pulsar red noise parameters - always hierarchical unless fixed
    prior_specs.get("hierarchical_specs")

    # Count gamma_p parameters
    if isinstance(prior_specs["log10_gamma_p_spec"], tfpd.Distribution):
        count += n_pulsars  # Fallback: one per pulsar
    elif prior_specs["log10_gamma_p_spec"] is None:
        # Hierarchical modeling: 2 hyperparameters + n_pulsars individual parameters
        count += 2  # log10_gamma_p_mean and log10_gamma_p_std
        count += n_pulsars  # Individual pulsar gamma parameters
    # If fixed, no parameters to count

    # Count sigma_p parameters
    if isinstance(prior_specs["log10_sigma_p_spec"], tfpd.Distribution):
        count += n_pulsars  # Fallback: one per pulsar
    elif prior_specs["log10_sigma_p_spec"] is None:
        # Log-ratio parameterization: 2 hyperparameters + n_pulsars ratio parameters
        count += 2  # log10_ratio_mean and log10_ratio_std
        count += n_pulsars  # Individual pulsar ratio parameters (σp derived deterministically)
    # If fixed, no parameters to count

    # Measurement noise parameters
    if isinstance(prior_specs["efac_spec"], tfpd.Distribution):
        count += n_pulsars  # One per pulsar

    # Handle EQUAD - can be either regular distribution or log10 parameterization
    if isinstance(prior_specs["equad_spec"], dict) and prior_specs["equad_spec"].get(
        "use_log10", False
    ):
        count += n_pulsars  # log10(EQUAD) parameters
    elif isinstance(prior_specs["equad_spec"], tfpd.Distribution):
        count += n_pulsars  # Regular EQUAD parameters

    return count


def build_jaxns_cw_prior_model(prior_specs, n_pulsars):
    """Build a jaxns-compatible prior model generator for CW inference.

    Creates a generator function that yields jaxns Prior objects using native
    TFP distributions. Unlike the NUTS path, no Normal(0,1) reparameterization
    is applied — jaxns samples the prior directly via unit hypercube mapping.

    Parameters
    ----------
    prior_specs : dict
        Prior specifications dictionary (same format as used by NUTS path).
    n_pulsars : int
        Number of pulsars.

    Returns
    -------
    callable
        Generator function compatible with jaxns.Model.
    """
    from jaxns.framework.prior import Prior
    import math

    cw_specs = prior_specs["cw_specs"]
    hierarchical_specs = prior_specs.get("hierarchical_specs")

    def prior_model():
        # --- CW source parameters (7 scalars) ---

        # log10_h0
        if cw_specs["log10_h0_transform_params"] is not None:
            tp = cw_specs["log10_h0_transform_params"]
            log10_h0 = yield Prior(
                tfpd.Uniform(low=tp["min"], high=tp["max"]),
                name="log10_h0",
            )
        else:
            log10_h0 = jnp.asarray(cw_specs["log10_h0_spec"])

        # alpha_gw
        if cw_specs["alpha_gw_transform_params"] is not None:
            tp = cw_specs["alpha_gw_transform_params"]
            alpha_gw = yield Prior(
                tfpd.Uniform(low=tp["min"], high=tp["max"]),
                name="alpha_gw",
            )
        else:
            alpha_gw = jnp.asarray(cw_specs["alpha_gw_spec"])

        # delta_gw via sin(delta) for isotropic sky coverage
        if cw_specs["sin_delta_gw_transform_params"] is not None:
            tp = cw_specs["sin_delta_gw_transform_params"]
            sin_delta_gw = yield Prior(
                tfpd.Uniform(low=tp["min"], high=tp["max"]),
                name="sin_delta_gw",
            )
            delta_gw = jnp.arcsin(sin_delta_gw)
        else:
            delta_gw = jnp.asarray(cw_specs["delta_gw_spec"])

        # log10_f_gw
        if cw_specs["log10_f_gw_transform_params"] is not None:
            tp = cw_specs["log10_f_gw_transform_params"]
            log10_f_gw = yield Prior(
                tfpd.Uniform(low=tp["min"], high=tp["max"]),
                name="log10_f_gw",
            )
        else:
            log10_f_gw = jnp.asarray(cw_specs["log10_f_gw_spec"])

        # cos_iota
        if cw_specs["cos_iota_transform_params"] is not None:
            tp = cw_specs["cos_iota_transform_params"]
            cos_iota = yield Prior(
                tfpd.Uniform(low=tp["min"], high=tp["max"]),
                name="cos_iota",
            )
        else:
            cos_iota = jnp.asarray(cw_specs["cos_iota_spec"])

        # psi
        if cw_specs["psi_transform_params"] is not None:
            tp = cw_specs["psi_transform_params"]
            psi = yield Prior(
                tfpd.Uniform(low=tp["min"], high=tp["max"]),
                name="psi",
            )
        else:
            psi = jnp.asarray(cw_specs["psi_spec"])

        # Phi0
        if cw_specs["Phi0_transform_params"] is not None:
            tp = cw_specs["Phi0_transform_params"]
            Phi0 = yield Prior(
                tfpd.Uniform(low=tp["min"], high=tp["max"]),
                name="Phi0",
            )
        else:
            Phi0 = jnp.asarray(cw_specs["Phi0_spec"])

        # --- Per-pulsar chi parameters ---
        if cw_specs.get("chi_transform_params") is not None:
            tp = cw_specs["chi_transform_params"]
            chi = yield Prior(
                tfpd.Uniform(
                    low=jnp.full(n_pulsars, tp["min"]),
                    high=jnp.full(n_pulsars, tp["max"]),
                ),
                name="chi",
            )
        else:
            chi = jnp.zeros(n_pulsars)

        # --- Pulsar noise parameters ---

        # log10_gamma_p: hierarchical or fixed
        if isinstance(prior_specs["log10_gamma_p_spec"], tfpd.Distribution):
            # Fallback: direct uniform prior
            spec = prior_specs["log10_gamma_p_spec"]
            log10_gamma_p = yield Prior(
                tfpd.Uniform(
                    low=jnp.full(n_pulsars, spec.low),
                    high=jnp.full(n_pulsars, spec.high),
                ),
                name="log10_gamma_p",
            )
        elif prior_specs["log10_gamma_p_spec"] is not None:
            # Fixed value (from injections)
            log10_gamma_p = jnp.asarray(prior_specs["log10_gamma_p_spec"])
        else:
            # Hierarchical modeling
            gp_mean_spec = hierarchical_specs["log10_gamma_p_mean_spec"]
            log10_gamma_p_mean = yield Prior(
                tfpd.Uniform(low=gp_mean_spec.low, high=gp_mean_spec.high),
                name="log10_gamma_p_mean",
            )
            gp_std_spec = hierarchical_specs["log10_gamma_p_std_spec"]
            log10_gamma_p_std = yield Prior(
                tfpd.Uniform(low=gp_std_spec.low, high=gp_std_spec.high),
                name="log10_gamma_p_std",
            )
            log10_gamma_p = yield Prior(
                tfpd.Normal(
                    loc=jnp.full(n_pulsars, log10_gamma_p_mean),
                    scale=jnp.full(n_pulsars, log10_gamma_p_std),
                ),
                name="log10_gamma_p",
            )

        # log10_sigma_p: log-ratio or fixed
        if isinstance(prior_specs["log10_sigma_p_spec"], tfpd.Distribution):
            # Fallback: direct uniform prior
            spec = prior_specs["log10_sigma_p_spec"]
            log10_sigma_p = yield Prior(
                tfpd.Uniform(
                    low=jnp.full(n_pulsars, spec.low),
                    high=jnp.full(n_pulsars, spec.high),
                ),
                name="log10_sigma_p",
            )
        elif prior_specs["log10_sigma_p_spec"] is not None:
            # Fixed value (from injections)
            log10_sigma_p = jnp.asarray(prior_specs["log10_sigma_p_spec"])
        else:
            # Log-ratio parameterization
            ratio_mean_spec = hierarchical_specs["log10_ratio_mean_spec"]
            log10_ratio_mean = yield Prior(
                tfpd.Uniform(low=ratio_mean_spec.low, high=ratio_mean_spec.high),
                name="log10_ratio_mean",
            )
            ratio_std_spec = hierarchical_specs["log10_ratio_std_spec"]
            log10_ratio_std = yield Prior(
                tfpd.Uniform(low=ratio_std_spec.low, high=ratio_std_spec.high),
                name="log10_ratio_std",
            )
            log10_ratio = yield Prior(
                tfpd.Normal(
                    loc=jnp.full(n_pulsars, log10_ratio_mean),
                    scale=jnp.full(n_pulsars, log10_ratio_std),
                ),
                name="log10_ratio",
            )
            log10_sigma_p = log10_gamma_p + log10_ratio

        # --- Measurement noise parameters ---

        # EFAC
        if isinstance(prior_specs["efac_spec"], tfpd.Distribution):
            spec = prior_specs["efac_spec"]
            efac = yield Prior(
                tfpd.Uniform(
                    low=jnp.full(n_pulsars, spec.low),
                    high=jnp.full(n_pulsars, spec.high),
                ),
                name="efac",
            )
        else:
            efac = jnp.asarray(prior_specs["efac_spec"])

        # EQUAD
        if isinstance(prior_specs["equad_spec"], dict) and prior_specs[
            "equad_spec"
        ].get("use_log10", False):
            log10_equad_spec = prior_specs["equad_spec"]["log10_equad_spec"]
            log10_equad = yield Prior(
                tfpd.Uniform(
                    low=jnp.full(n_pulsars, log10_equad_spec.low),
                    high=jnp.full(n_pulsars, log10_equad_spec.high),
                ),
                name="log10_equad",
            )
            equad = 10.0**log10_equad
        elif isinstance(prior_specs["equad_spec"], tfpd.Distribution):
            spec = prior_specs["equad_spec"]
            equad = yield Prior(
                tfpd.Uniform(
                    low=jnp.full(n_pulsars, spec.low),
                    high=jnp.full(n_pulsars, spec.high),
                ),
                name="equad",
            )
        else:
            equad = jnp.asarray(prior_specs["equad_spec"])

        return (
            log10_h0, alpha_gw, delta_gw, log10_f_gw,
            cos_iota, psi, Phi0, chi,
            log10_gamma_p, log10_sigma_p, efac, equad,
        )

    return prior_model
