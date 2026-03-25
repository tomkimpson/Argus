"""Prior model specifications for Bayesian inference.

This module provides functionality for defining and creating prior distributions
for gravitational wave background and pulsar noise parameters used in
pulsar timing array analysis.
"""

import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

tfpd = tfp.distributions


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
        is_fixed = config.getboolean("PriorModel", f"{param_name}_fixed")
        if is_fixed:
            return config.getfloat("PriorModel", f"{param_name}_value")
        else:
            min_val = config.getfloat("PriorModel", f"{param_name}_min")
            max_val = config.getfloat("PriorModel", f"{param_name}_max")
            return tfpd.Uniform(min_val, max_val)

    # Handle log10_ha with reparameterization for better NUTS sampling
    log10_ha_fixed = config.getboolean("PriorModel", "log10_ha_fixed")
    if log10_ha_fixed:
        # Fixed value - no reparameterization needed
        log10_ha_spec = config.getfloat("PriorModel", "log10_ha_value")
        log10_ha_transform_params = None
    else:
        # Reparameterize U(a,b) -> N(0,1) for better NUTS sampling
        min_val = config.getfloat("PriorModel", "log10_ha_min")
        max_val = config.getfloat("PriorModel", "log10_ha_max")

        # Calculate improved transformation parameters: log10_ha = mean + log10_ha_prime * std
        # Use 3-sigma rule for better convergence
        mean = (min_val + max_val) / 2.0
        std = (max_val - min_val) / 6.0  # 3-sigma rule: 99.7% of samples within range

        # Use N(0,1) for log10_ha_prime, store transformation parameters
        log10_ha_spec = tfpd.Normal(0.0, 1.0)  # log10_ha_prime ~ N(0,1)
        log10_ha_transform_params = {
            "mean": mean,
            "std": std,
            "min": min_val,
            "max": max_val,
        }

    log10_gamma_a_spec = get_prior_spec("log10_gamma_a")

    return {
        "log10_ha_spec": log10_ha_spec,
        "log10_ha_transform_params": log10_ha_transform_params,
        "log10_gamma_a_spec": log10_gamma_a_spec,
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
        spin_injections_path = config.get("PriorModel", "spin_injections_path")
        # If path is provided and not empty, fix red noise parameters
        log10_gamma_p_fixed = bool(spin_injections_path.strip())
        log10_sigma_p_fixed = bool(spin_injections_path.strip())
        print(
            f"Red noise parameters fixed via spin_injections_path: {log10_gamma_p_fixed}"
        )
    except Exception:
        # If no spin_injections_path, sample from priors
        log10_gamma_p_fixed = False
        log10_sigma_p_fixed = False
        print(
            "No spin_injections_path provided, sampling red noise parameters from priors"
        )

    # Always use hierarchical modeling and log-ratio parameterization
    hierarchical_specs = create_hierarchical_priors(config)

    # Handle gamma_p specification
    if log10_gamma_p_fixed:
        if config.has_option("PriorModel", "log10_gamma_p_value"):
            # Check if value is a string (for 'injected'/'default') or a number
            gamma_p_value_str = config.get("PriorModel", "log10_gamma_p_value")
            if gamma_p_value_str.lower() in ["injected", "default"]:
                # Use injected values
                log10_gamma_p_spec = jnp.log10(gamma_p_array)
                print(f"Using injected gamma_p values: {gamma_p_value_str}")
            else:
                # Use explicit fixed value from config
                gamma_p_fixed_value = config.getfloat(
                    "PriorModel", "log10_gamma_p_value"
                )
                log10_gamma_p_spec = jnp.full(n_pulsars, gamma_p_fixed_value)
                print(f"Using fixed gamma_p value: {gamma_p_fixed_value}")
        else:
            # Use injected values (legacy approach)
            log10_gamma_p_spec = jnp.log10(gamma_p_array)
            print("Using injected gamma_p values (legacy mode)")
    else:
        log10_gamma_p_spec = None  # Will be handled hierarchically

    # Handle sigma_p specification
    if log10_sigma_p_fixed:
        if config.has_option("PriorModel", "log10_sigma_p_value"):
            # Check if value is a string (for 'injected'/'default') or a number
            sigma_p_value_str = config.get("PriorModel", "log10_sigma_p_value")
            if sigma_p_value_str.lower() in ["injected", "default"]:
                # Use injected values
                log10_sigma_p_spec = jnp.log10(sigma_p_array)
                print(f"Using injected sigma_p values: {sigma_p_value_str}")
            else:
                # Use explicit fixed value from config
                sigma_p_fixed_value = config.getfloat(
                    "PriorModel", "log10_sigma_p_value"
                )
                log10_sigma_p_spec = jnp.full(n_pulsars, sigma_p_fixed_value)
                print(f"Using fixed sigma_p value: {sigma_p_fixed_value}")
        else:
            # Use injected values (legacy approach)
            log10_sigma_p_spec = jnp.log10(sigma_p_array)
            print("Using injected sigma_p values (legacy mode)")
    else:
        log10_sigma_p_spec = None  # Will be derived from log-ratio

    return {
        "log10_gamma_p_spec": log10_gamma_p_spec,
        "log10_sigma_p_spec": log10_sigma_p_spec,
        "hierarchical_specs": hierarchical_specs,
    }


def get_measurement_noise_priors(config, n_pulsars, efac_array, equad_array):
    """Extract measurement noise parameter prior distributions from config.

    Parameters
    ----------
    config : ConfigParser
        Configuration object containing prior model settings
    n_pulsars : int
        Number of pulsars
    efac_array : array or None
        Array of EFAC values, or None if not provided
    equad_array : array or None
        Array of EQUAD values, or None if not provided

    Returns
    -------
    dict
        Dictionary containing measurement noise parameter prior distributions:
        - efac_spec: Prior distribution for EFAC
        - equad_spec: Prior distribution for EQUAD
    """
    # Check if noise_params_path is provided to determine if EFAC/EQUAD should be fixed
    try:
        noise_params_path = config.get("PriorModel", "noise_params_path")
        # If path is provided and not empty, fix EFAC/EQUAD parameters
        efac_equad_fixed = bool(noise_params_path.strip())
        print(f"EFAC/EQUAD parameters fixed via noise_params_path: {efac_equad_fixed}")
    except Exception:
        # If no noise_params_path, sample from priors
        efac_equad_fixed = False
        print("No noise_params_path provided, sampling EFAC/EQUAD from priors")

    if efac_equad_fixed and efac_array is not None and equad_array is not None:
        efac_spec = efac_array
        equad_spec = equad_array
    else:
        # Create prior distributions using the number of pulsars to determine shape
        efac_spec = tfpd.Uniform(
            low=jnp.full(n_pulsars, config.getfloat("PriorModel", "efac_min")),
            high=jnp.full(n_pulsars, config.getfloat("PriorModel", "efac_max")),
        )

        # Use log10(EQUAD) uniform prior - transformation handled in numpyro model
        log10_equad_spec = tfpd.Uniform(
            low=jnp.full(n_pulsars, config.getfloat("PriorModel", "log10_equad_min")),
            high=jnp.full(n_pulsars, config.getfloat("PriorModel", "log10_equad_max")),
        )
        equad_spec = {"log10_equad_spec": log10_equad_spec, "use_log10": True}

    return {"efac_spec": efac_spec, "equad_spec": equad_spec}


def create_hierarchical_priors(config):
    """Create hierarchical modeling prior distributions.

    Always creates hierarchical priors for both gamma_p and sigma_p (via log-ratio)
    parameterization to improve MCMC sampling efficiency.

    Parameters
    ----------
    config : ConfigParser
        Configuration object containing hyperprior ranges

    Returns
    -------
    dict
        Hierarchical prior distributions dictionary
    """
    hierarchical_specs = {
        "hierarchical_noise": True,
        "log_ratio_parameterization": True,
        "log10_gamma_p_mean_spec": tfpd.Uniform(
            config.getfloat("PriorModel", "log10_gamma_p_mean_min"),
            config.getfloat("PriorModel", "log10_gamma_p_mean_max"),
        ),
        "log10_gamma_p_std_spec": tfpd.Uniform(
            config.getfloat("PriorModel", "log10_gamma_p_std_min"),
            config.getfloat("PriorModel", "log10_gamma_p_std_max"),
        ),
        "log10_ratio_mean_spec": tfpd.Uniform(
            config.getfloat("PriorModel", "log10_ratio_mean_min"),
            config.getfloat("PriorModel", "log10_ratio_mean_max"),
        ),
        "log10_ratio_std_spec": tfpd.Uniform(
            config.getfloat("PriorModel", "log10_ratio_std_min"),
            config.getfloat("PriorModel", "log10_ratio_std_max"),
        ),
    }

    return hierarchical_specs


def _make_reparameterized_prior(config, section, param_name):
    """Helper to create a reparameterized prior from config settings.

    Returns (spec, transform_params) tuple. If fixed, transform_params is None.
    """
    is_fixed = config.getboolean(section, f"{param_name}_fixed")
    if is_fixed:
        return config.getfloat(section, f"{param_name}_value"), None
    else:
        min_val = config.getfloat(section, f"{param_name}_min")
        max_val = config.getfloat(section, f"{param_name}_max")
        mean = (min_val + max_val) / 2.0
        std = (max_val - min_val) / 6.0
        return None, {"mean": mean, "std": std, "min": min_val, "max": max_val}


def get_cw_parameter_priors(config):
    """Extract CW source parameter prior distributions from config.

    Parameters
    ----------
    config : ConfigParser
        Configuration object containing [CWModel] section.

    Returns
    -------
    dict
        Dictionary containing CW parameter prior specifications.
    """
    section = "CWModel"
    cw_specs = {}

    # log10_h0: strain amplitude
    spec, tp = _make_reparameterized_prior(config, section, "log10_h0")
    cw_specs["log10_h0_spec"] = spec
    cw_specs["log10_h0_transform_params"] = tp

    # alpha_gw: source RA
    spec, tp = _make_reparameterized_prior(config, section, "alpha_gw")
    cw_specs["alpha_gw_spec"] = spec
    cw_specs["alpha_gw_transform_params"] = tp

    # sin_delta_gw: for isotropic sky coverage (sample in sin(delta), convert to delta)
    spec, tp = _make_reparameterized_prior(config, section, "sin_delta_gw")
    cw_specs["sin_delta_gw_spec"] = spec
    cw_specs["sin_delta_gw_transform_params"] = tp
    # Also store delta_gw_spec for fixed case (direct declination)
    if config.getboolean(section, "sin_delta_gw_fixed"):
        cw_specs["delta_gw_spec"] = config.getfloat(section, "delta_gw_value")
    else:
        cw_specs["delta_gw_spec"] = None

    # log10_f_gw: GW frequency
    spec, tp = _make_reparameterized_prior(config, section, "log10_f_gw")
    cw_specs["log10_f_gw_spec"] = spec
    cw_specs["log10_f_gw_transform_params"] = tp

    # cos_iota: inclination
    spec, tp = _make_reparameterized_prior(config, section, "cos_iota")
    cw_specs["cos_iota_spec"] = spec
    cw_specs["cos_iota_transform_params"] = tp

    # psi: polarization angle
    spec, tp = _make_reparameterized_prior(config, section, "psi")
    cw_specs["psi_spec"] = spec
    cw_specs["psi_transform_params"] = tp

    # Phi0: initial phase
    spec, tp = _make_reparameterized_prior(config, section, "Phi0")
    cw_specs["Phi0_spec"] = spec
    cw_specs["Phi0_transform_params"] = tp

    # Per-pulsar phase parameters (phase reparameterization of pulsar term)
    include_pulsar_term = config.getboolean(section, "include_pulsar_term", fallback=False)
    phase_parameterization = config.getboolean(section, "phase_parameterization", fallback=True)

    if include_pulsar_term and phase_parameterization:
        import math
        chi_min = 0.0
        chi_max = 2.0 * math.pi
        mean = (chi_min + chi_max) / 2.0
        std = (chi_max - chi_min) / 6.0
        cw_specs["chi_transform_params"] = {
            "mean": mean, "std": std, "min": chi_min, "max": chi_max,
        }
    else:
        cw_specs["chi_transform_params"] = None

    return cw_specs


def get_prior_model_specs(
    config, n_pulsars, sigma_p_array, gamma_p_array, efac_array, equad_array,
    mode="gwb",
):
    """Create prior model distributions based on config settings.

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
    efac_array : array
        Array of EFAC values
    equad_array : array
        Array of EQUAD values
    mode : str
        Signal model mode: 'gwb' or 'cw'.

    Returns
    -------
    dict
        Dictionary containing all prior distributions.
    """
    print(f"Getting prior model specs (mode={mode})...")

    # Pulsar noise and measurement noise priors are shared between modes
    pulsar_noise_specs = get_pulsar_noise_priors(
        config, n_pulsars, sigma_p_array, gamma_p_array
    )
    measurement_noise_specs = get_measurement_noise_priors(
        config, n_pulsars, efac_array, equad_array
    )

    result = {
        "log10_gamma_p_spec": pulsar_noise_specs["log10_gamma_p_spec"],
        "log10_sigma_p_spec": pulsar_noise_specs["log10_sigma_p_spec"],
        "efac_spec": measurement_noise_specs["efac_spec"],
        "equad_spec": measurement_noise_specs["equad_spec"],
        "hierarchical_specs": pulsar_noise_specs["hierarchical_specs"],
    }

    if mode == "cw":
        # CW-specific priors (no GWB amplitude/spectral index)
        cw_specs = get_cw_parameter_priors(config)
        result["cw_specs"] = cw_specs
        # Dummy GWB entries for backward compatibility with count_free_parameters
        result["log10_ha_spec"] = None
        result["log10_ha_transform_params"] = None
        result["log10_gamma_a_spec"] = None
    else:
        # GWB-specific priors
        gw_specs = get_gw_parameter_priors(config)
        result["log10_ha_spec"] = gw_specs["log10_ha_spec"]
        result["log10_ha_transform_params"] = gw_specs["log10_ha_transform_params"]
        result["log10_gamma_a_spec"] = gw_specs["log10_gamma_a_spec"]

    return result
