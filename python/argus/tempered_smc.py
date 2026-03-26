"""Tempered SMC inference via blackjax for CW gravitational wave source parameter estimation.

This module provides a blackjax-based tempered Sequential Monte Carlo (SMC) sampler
as an alternative to NumPyro NUTS and jaxns nested sampling. Tempered SMC moves
particles from the prior (beta=0) to the posterior (beta=1) through a sequence of
tempered distributions, with resampling and MCMC mutation at each step.

Key components:
- Parameter registry for packing/unpacking flat vectors to named physical parameters
- Log-prior and log-likelihood functions compatible with blackjax
- Tempered SMC execution with NUTS mutation kernels
- ArviZ conversion and SMC-specific diagnostics
"""

import dataclasses
import json
import logging
import os
import time
from functools import partial
from typing import Optional

import arviz as az
import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_probability.substrates.jax as tfp

import blackjax
import blackjax.smc.resampling as resampling

from .bayesian_inference import cw_log_likelihood_fn

jax.config.update("jax_enable_x64", True)
tfpd = tfp.distributions

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter registry
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ParamBlock:
    """Describes one block of parameters in the flat unconstrained vector."""
    name: str               # unconstrained name, e.g. "log10_h0_prime"
    physical_name: str      # physical name, e.g. "log10_h0"
    size: int               # 1 for scalar, n_pulsars for vector
    start_idx: int          # position in flat vector
    transform_mean: jnp.ndarray   # affine transform: physical = mean + prime * std
    transform_std: jnp.ndarray


@dataclasses.dataclass
class ParameterRegistry:
    """Registry mapping flat unconstrained vector to named physical parameters."""
    blocks: list            # list of ParamBlock
    ndim: int               # total flat vector dimension
    n_pulsars: int
    # Fixed parameter values (not in flat vector)
    fixed_values: dict      # {physical_name: value}
    # Hierarchical coupling info
    has_hierarchical_gamma: bool
    has_hierarchical_ratio: bool
    # EQUAD parameterisation
    equad_use_log10: bool


def build_parameter_registry(prior_specs, n_pulsars):
    """Build a registry mapping flat unconstrained vector positions to physical parameters.

    Mirrors the reparameterisation used in parameter_sampling.py for NUTS.
    All unconstrained parameters are sampled from Normal(0,1).

    Parameters
    ----------
    prior_specs : dict
        Prior specifications (same format as used by NUTS/jaxns).
    n_pulsars : int
        Number of pulsars.

    Returns
    -------
    ParameterRegistry
    """
    blocks = []
    fixed_values = {}
    idx = 0

    cw_specs = prior_specs.get("cw_specs", {})

    # --- CW source parameters (7 scalars) ---
    cw_param_keys = [
        ("log10_h0", "log10_h0_transform_params", "log10_h0_spec"),
        ("alpha_gw", "alpha_gw_transform_params", "alpha_gw_spec"),
        ("sin_delta_gw", "sin_delta_gw_transform_params", "sin_delta_gw_spec"),
        ("log10_f_gw", "log10_f_gw_transform_params", "log10_f_gw_spec"),
        ("cos_iota", "cos_iota_transform_params", "cos_iota_spec"),
        ("psi", "psi_transform_params", "psi_spec"),
        ("Phi0", "Phi0_transform_params", "Phi0_spec"),
    ]

    for phys_name, tp_key, fixed_key in cw_param_keys:
        tp = cw_specs.get(tp_key)
        if tp is not None:
            blocks.append(ParamBlock(
                name=f"{phys_name}_prime",
                physical_name=phys_name,
                size=1,
                start_idx=idx,
                transform_mean=jnp.asarray(tp["mean"]),
                transform_std=jnp.asarray(tp["std"]),
            ))
            idx += 1
        else:
            # Fixed value
            val = cw_specs.get(fixed_key)
            if phys_name == "sin_delta_gw":
                # For fixed delta, store delta_gw directly
                fixed_values["delta_gw"] = jnp.asarray(cw_specs.get("delta_gw_spec", 0.0))
                fixed_values["sin_delta_gw"] = jnp.sin(fixed_values["delta_gw"])
            else:
                fixed_values[phys_name] = jnp.asarray(val)

    # --- Per-pulsar chi parameters ---
    has_chi = False
    chi_tp = cw_specs.get("chi_transform_params")
    if chi_tp is not None:
        has_chi = True
        blocks.append(ParamBlock(
            name="chi_prime",
            physical_name="chi",
            size=n_pulsars,
            start_idx=idx,
            transform_mean=jnp.asarray(chi_tp["mean"]),
            transform_std=jnp.asarray(chi_tp["std"]),
        ))
        idx += n_pulsars

    # --- Hierarchical gamma_p parameters ---
    has_hierarchical_gamma = False
    hierarchical_specs = prior_specs.get("hierarchical_specs")

    if isinstance(prior_specs["log10_gamma_p_spec"], tfpd.Distribution):
        # Fallback: reparameterised per-pulsar (not hierarchical)
        low = prior_specs["log10_gamma_p_spec"].low
        high = prior_specs["log10_gamma_p_spec"].high
        mean = (low + high) / 2.0
        std = (high - low) / 6.0
        blocks.append(ParamBlock(
            name="log10_γp_standardized",
            physical_name="log10_γp",
            size=n_pulsars,
            start_idx=idx,
            transform_mean=jnp.asarray(mean),
            transform_std=jnp.asarray(std),
        ))
        idx += n_pulsars
    elif prior_specs["log10_gamma_p_spec"] is not None:
        # Fixed
        fixed_values["log10_γp"] = jnp.asarray(prior_specs["log10_gamma_p_spec"])
    else:
        # Hierarchical: 2 hyperparams + n_pulsars raw
        has_hierarchical_gamma = True
        hs = hierarchical_specs

        # Hyperparameter: log10_gamma_p_mean
        mean_low = hs["log10_gamma_p_mean_spec"].low
        mean_high = hs["log10_gamma_p_mean_spec"].high
        blocks.append(ParamBlock(
            name="log10_gamma_p_mean_raw",
            physical_name="log10_gamma_p_mean",
            size=1,
            start_idx=idx,
            transform_mean=jnp.asarray((mean_low + mean_high) / 2.0),
            transform_std=jnp.asarray((mean_high - mean_low) / 6.0),
        ))
        idx += 1

        # Hyperparameter: log10_gamma_p_std
        std_low = hs["log10_gamma_p_std_spec"].low
        std_high = hs["log10_gamma_p_std_spec"].high
        blocks.append(ParamBlock(
            name="log10_gamma_p_std_raw",
            physical_name="log10_gamma_p_std",
            size=1,
            start_idx=idx,
            transform_mean=jnp.asarray((std_low + std_high) / 2.0),
            transform_std=jnp.asarray((std_high - std_low) / 6.0),
        ))
        idx += 1

        # Per-pulsar raw values (Normal(0,1), coupled transform)
        blocks.append(ParamBlock(
            name="log10_γp_raw",
            physical_name="log10_γp_raw",
            size=n_pulsars,
            start_idx=idx,
            transform_mean=jnp.zeros(n_pulsars),  # placeholder — coupled transform
            transform_std=jnp.ones(n_pulsars),     # placeholder — coupled transform
        ))
        idx += n_pulsars

    # --- Hierarchical log-ratio (sigma_p) parameters ---
    has_hierarchical_ratio = False

    if isinstance(prior_specs["log10_sigma_p_spec"], tfpd.Distribution):
        # Fallback: reparameterised per-pulsar
        low = prior_specs["log10_sigma_p_spec"].low
        high = prior_specs["log10_sigma_p_spec"].high
        mean = (low + high) / 2.0
        std = (high - low) / 6.0
        blocks.append(ParamBlock(
            name="log10_σp_standardized",
            physical_name="log10_σp",
            size=n_pulsars,
            start_idx=idx,
            transform_mean=jnp.asarray(mean),
            transform_std=jnp.asarray(std),
        ))
        idx += n_pulsars
    elif prior_specs["log10_sigma_p_spec"] is not None:
        # Fixed
        fixed_values["log10_σp"] = jnp.asarray(prior_specs["log10_sigma_p_spec"])
    else:
        # Hierarchical log-ratio: 2 hyperparams + n_pulsars raw
        has_hierarchical_ratio = True
        hs = hierarchical_specs

        # Hyperparameter: log10_ratio_mean
        mean_low = hs["log10_ratio_mean_spec"].low
        mean_high = hs["log10_ratio_mean_spec"].high
        blocks.append(ParamBlock(
            name="log10_ratio_mean_raw",
            physical_name="log10_ratio_mean",
            size=1,
            start_idx=idx,
            transform_mean=jnp.asarray((mean_low + mean_high) / 2.0),
            transform_std=jnp.asarray((mean_high - mean_low) / 6.0),
        ))
        idx += 1

        # Hyperparameter: log10_ratio_std
        std_low = hs["log10_ratio_std_spec"].low
        std_high = hs["log10_ratio_std_spec"].high
        blocks.append(ParamBlock(
            name="log10_ratio_std_raw",
            physical_name="log10_ratio_std",
            size=1,
            start_idx=idx,
            transform_mean=jnp.asarray((std_low + std_high) / 2.0),
            transform_std=jnp.asarray((std_high - std_low) / 6.0),
        ))
        idx += 1

        # Per-pulsar raw ratio values
        blocks.append(ParamBlock(
            name="log10_ratio_raw",
            physical_name="log10_ratio_raw",
            size=n_pulsars,
            start_idx=idx,
            transform_mean=jnp.zeros(n_pulsars),
            transform_std=jnp.ones(n_pulsars),
        ))
        idx += n_pulsars

    # --- Measurement noise: EFAC ---
    equad_use_log10 = False

    if isinstance(prior_specs["efac_spec"], tfpd.Distribution):
        low = prior_specs["efac_spec"].low
        high = prior_specs["efac_spec"].high
        mean = (low + high) / 2.0
        std = (high - low) / 6.0
        blocks.append(ParamBlock(
            name="efac_standardized",
            physical_name="efac",
            size=n_pulsars,
            start_idx=idx,
            transform_mean=jnp.asarray(mean),
            transform_std=jnp.asarray(std),
        ))
        idx += n_pulsars
    else:
        fixed_values["efac"] = jnp.asarray(prior_specs["efac_spec"])

    # --- Measurement noise: EQUAD ---
    if isinstance(prior_specs["equad_spec"], dict) and prior_specs["equad_spec"].get(
        "use_log10", False
    ):
        equad_use_log10 = True
        log10_equad_spec = prior_specs["equad_spec"]["log10_equad_spec"]
        low = log10_equad_spec.low
        high = log10_equad_spec.high
        mean = (low + high) / 2.0
        std = (high - low) / 6.0
        blocks.append(ParamBlock(
            name="log10_equad_prime",
            physical_name="log10_equad",
            size=n_pulsars,
            start_idx=idx,
            transform_mean=jnp.asarray(mean),
            transform_std=jnp.asarray(std),
        ))
        idx += n_pulsars
    elif isinstance(prior_specs["equad_spec"], tfpd.Distribution):
        low = prior_specs["equad_spec"].low
        high = prior_specs["equad_spec"].high
        mean = (low + high) / 2.0
        std = (high - low) / 6.0
        blocks.append(ParamBlock(
            name="equad_standardized",
            physical_name="equad",
            size=n_pulsars,
            start_idx=idx,
            transform_mean=jnp.asarray(mean),
            transform_std=jnp.asarray(std),
        ))
        idx += n_pulsars
    else:
        fixed_values["equad"] = jnp.asarray(prior_specs["equad_spec"])

    return ParameterRegistry(
        blocks=blocks,
        ndim=idx,
        n_pulsars=n_pulsars,
        fixed_values=fixed_values,
        has_hierarchical_gamma=has_hierarchical_gamma,
        has_hierarchical_ratio=has_hierarchical_ratio,
        equad_use_log10=equad_use_log10,
    )


# ---------------------------------------------------------------------------
# Unpack flat vector → physical parameters
# ---------------------------------------------------------------------------

def _get_block(registry, name):
    """Find a ParamBlock by name."""
    for b in registry.blocks:
        if b.name == name:
            return b
    return None


def unpack_to_physical(x_flat, registry):
    """Transform a flat unconstrained vector to physical parameter dict.

    Parameters
    ----------
    x_flat : jnp.ndarray
        Flat unconstrained vector of shape (ndim,).
    registry : ParameterRegistry

    Returns
    -------
    dict
        Physical parameter values keyed by name.
    """
    params = {}
    n_pulsars = registry.n_pulsars

    # Simple affine blocks
    for block in registry.blocks:
        raw = x_flat[block.start_idx : block.start_idx + block.size]
        if block.size == 1:
            raw = raw[0]

        # Skip hierarchical raw blocks — handled below
        if block.name in ("log10_γp_raw", "log10_ratio_raw"):
            params[block.name] = raw
            continue

        physical = block.transform_mean + raw * block.transform_std
        params[block.physical_name] = physical

    # --- Hierarchical gamma coupling ---
    if registry.has_hierarchical_gamma:
        gp_mean = params["log10_gamma_p_mean"]
        gp_std = params["log10_gamma_p_std"]
        gp_raw = params["log10_γp_raw"]
        params["log10_γp"] = gp_mean + gp_raw * gp_std / jnp.sqrt(n_pulsars)

    # --- Hierarchical ratio coupling ---
    if registry.has_hierarchical_ratio:
        ratio_mean = params["log10_ratio_mean"]
        ratio_std = params["log10_ratio_std"]
        ratio_raw = params["log10_ratio_raw"]
        log10_ratio = ratio_mean + ratio_raw * ratio_std / jnp.sqrt(n_pulsars)
        params["log10_ratio"] = log10_ratio
        # Derive sigma_p from gamma_p + ratio
        params["log10_σp"] = params["log10_γp"] + log10_ratio

    # --- Derived: delta_gw from sin_delta_gw ---
    if "sin_delta_gw" in params:
        params["delta_gw"] = jnp.arcsin(params["sin_delta_gw"])

    # --- Derived: equad from log10_equad ---
    if registry.equad_use_log10 and "log10_equad" in params:
        params["equad"] = 10.0 ** params["log10_equad"]

    # --- Inject fixed values ---
    for k, v in registry.fixed_values.items():
        params[k] = v

    # --- Default chi to zeros if not present ---
    if "chi" not in params:
        params["chi"] = jnp.zeros(n_pulsars)

    return params


# ---------------------------------------------------------------------------
# Log-probability functions for blackjax
# ---------------------------------------------------------------------------

def build_logprior_fn(registry):
    """Build log-prior function in unconstrained space.

    All unconstrained parameters are independently Normal(0,1), so the
    log-prior is simply -0.5 * sum(x^2) (up to a constant).

    Parameters
    ----------
    registry : ParameterRegistry

    Returns
    -------
    Callable
        log_prior(x_flat) -> scalar
    """
    def logprior_fn(x_flat):
        return -0.5 * jnp.sum(x_flat ** 2)

    return logprior_fn


def build_loglikelihood_fn(cw_kf, registry, n_pulsars):
    """Build log-likelihood function that unpacks flat vector and calls the CW Kalman filter.

    Parameters
    ----------
    cw_kf : CWKalmanFilter
        CW Kalman filter instance.
    registry : ParameterRegistry
    n_pulsars : int

    Returns
    -------
    Callable
        log_likelihood(x_flat) -> scalar
    """
    def loglikelihood_fn(x_flat):
        params = unpack_to_physical(x_flat, registry)
        return cw_log_likelihood_fn(
            cw_kf,
            params["log10_h0"],
            params.get("alpha_gw", registry.fixed_values.get("alpha_gw")),
            params.get("delta_gw", registry.fixed_values.get("delta_gw")),
            params.get("log10_f_gw", registry.fixed_values.get("log10_f_gw")),
            params.get("cos_iota", registry.fixed_values.get("cos_iota")),
            params.get("psi", registry.fixed_values.get("psi")),
            params.get("Phi0", registry.fixed_values.get("Phi0")),
            params["chi"],
            params["log10_γp"],
            params["log10_σp"],
            params["efac"],
            params["equad"],
        )

    return loglikelihood_fn


# ---------------------------------------------------------------------------
# Temperature schedule
# ---------------------------------------------------------------------------

def build_temperature_schedule(num_temps, spacing="geometric"):
    """Build a temperature schedule (lambda values from 0 to 1).

    Parameters
    ----------
    num_temps : int
        Number of temperature rungs (excluding the initial prior step).
    spacing : str
        "geometric" or "linear".

    Returns
    -------
    jnp.ndarray
        Array of lambda values in (0, 1], ending at 1.0.
    """
    if spacing == "linear":
        return jnp.linspace(1.0 / num_temps, 1.0, num_temps)
    else:
        # Geometric: denser near 0, sparser near 1
        return jnp.geomspace(1.0 / num_temps, 1.0, num_temps)


# ---------------------------------------------------------------------------
# Tempered SMC execution
# ---------------------------------------------------------------------------

def run_tempered_smc(
    logprior_fn,
    loglikelihood_fn,
    ndim,
    num_particles,
    num_mcmc_steps,
    adaptive,
    target_ess,
    num_temps,
    temp_spacing,
    step_size,
    inverse_mass_matrix,
    seed,
):
    """Run blackjax tempered SMC.

    Parameters
    ----------
    logprior_fn : Callable
        Log-prior function.
    loglikelihood_fn : Callable
        Log-likelihood function.
    ndim : int
        Dimensionality of the parameter space.
    num_particles : int
        Number of SMC particles.
    num_mcmc_steps : int
        Number of MCMC mutation steps per temperature rung.
    adaptive : bool
        If True, use adaptive temperature scheduling (ESS-based).
    target_ess : float
        Target ESS ratio for adaptive scheduling (0 < target_ess < 1).
    num_temps : int
        Number of temperature rungs (used if not adaptive).
    temp_spacing : str
        Temperature spacing ("geometric" or "linear").
    step_size : float
        NUTS step size.
    inverse_mass_matrix : jnp.ndarray
        Diagonal inverse mass matrix, shape (ndim,).
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with keys: "particles", "weights", "intermediate_states",
        "log_evidence", "temperature_schedule", "wall_time".
    """
    rng_key = random.PRNGKey(seed)

    # Draw initial particles from the prior (Normal(0,1) in unconstrained space)
    rng_key, init_key = random.split(rng_key)
    initial_particles = random.normal(init_key, shape=(num_particles, ndim))

    # Initialise tempered SMC state
    if adaptive:
        import blackjax.smc.adaptive_tempered as adaptive_tempered
        smc_module = adaptive_tempered
    else:
        import blackjax.smc.tempered as tempered
        smc_module = tempered

    state = smc_module.init(initial_particles)

    # Build NUTS mutation kernel
    nuts_kernel = blackjax.nuts.build_kernel()
    nuts_init = blackjax.nuts.init

    # MCMC parameters (shared across particles — leading dim = 1)
    mcmc_parameters = {
        "step_size": jnp.array([step_size]),
        "inverse_mass_matrix": jnp.expand_dims(inverse_mass_matrix, axis=0),
    }

    # Build the SMC kernel
    if adaptive:
        kernel = smc_module.build_kernel(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            mcmc_step_fn=nuts_kernel,
            mcmc_init_fn=nuts_init,
            resampling_fn=resampling.systematic,
            target_ess=target_ess,
        )
    else:
        kernel = smc_module.build_kernel(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            mcmc_step_fn=nuts_kernel,
            mcmc_init_fn=nuts_init,
            resampling_fn=resampling.systematic,
        )

    # Run the tempered SMC loop
    intermediate_states = []
    log_evidence = 0.0
    temperature_schedule = [0.0]

    if adaptive:
        # Adaptive: step until tempering_param reaches 1.0
        logger.info("Running adaptive tempered SMC...")
        t_start = time.time()
        step_count = 0

        while state.tempering_param < 1.0:
            rng_key, step_key = random.split(rng_key)
            state, info = kernel(
                step_key,
                state,
                num_mcmc_steps=num_mcmc_steps,
                mcmc_parameters=mcmc_parameters,
            )
            log_evidence += info.log_likelihood_increment
            temperature_schedule.append(float(state.tempering_param))
            intermediate_states.append({
                "tempering_param": float(state.tempering_param),
                "log_likelihood_increment": float(info.log_likelihood_increment),
                "log_evidence_cumulative": float(log_evidence),
            })
            step_count += 1
            if step_count % 5 == 0:
                logger.info(
                    f"  Step {step_count}: lambda={state.tempering_param:.4f}, "
                    f"log_Z={log_evidence:.2f}"
                )

        wall_time = time.time() - t_start
        logger.info(
            f"Adaptive tempered SMC completed: {step_count} steps, "
            f"log_Z={log_evidence:.2f}, wall_time={wall_time:.1f}s"
        )

    else:
        # Fixed schedule
        schedule = build_temperature_schedule(num_temps, temp_spacing)
        logger.info(f"Running tempered SMC with {num_temps} temperature rungs...")
        t_start = time.time()

        for i, lam in enumerate(schedule):
            rng_key, step_key = random.split(rng_key)
            state, info = kernel(
                step_key,
                state,
                num_mcmc_steps=num_mcmc_steps,
                tempering_param=float(lam),
                mcmc_parameters=mcmc_parameters,
            )
            log_evidence += info.log_likelihood_increment
            temperature_schedule.append(float(lam))
            intermediate_states.append({
                "tempering_param": float(lam),
                "log_likelihood_increment": float(info.log_likelihood_increment),
                "log_evidence_cumulative": float(log_evidence),
            })
            if (i + 1) % 5 == 0 or i == len(schedule) - 1:
                logger.info(
                    f"  Rung {i+1}/{num_temps}: lambda={lam:.4f}, "
                    f"log_Z={log_evidence:.2f}"
                )

        wall_time = time.time() - t_start
        logger.info(
            f"Tempered SMC completed: {num_temps} rungs, "
            f"log_Z={log_evidence:.2f}, wall_time={wall_time:.1f}s"
        )

    return {
        "particles": state.particles,
        "weights": state.weights,
        "intermediate_states": intermediate_states,
        "log_evidence": float(log_evidence),
        "temperature_schedule": temperature_schedule,
        "wall_time": wall_time,
    }


# ---------------------------------------------------------------------------
# ArviZ conversion
# ---------------------------------------------------------------------------

def smc_results_to_arviz(particles, registry, n_pulsars):
    """Convert SMC particles to ArviZ InferenceData.

    Parameters
    ----------
    particles : jnp.ndarray
        Final particles, shape (num_particles, ndim), in unconstrained space.
    registry : ParameterRegistry
    n_pulsars : int

    Returns
    -------
    arviz.InferenceData
    """
    # Transform all particles to physical space
    physical_samples = jax.vmap(lambda x: unpack_to_physical(x, registry))(particles)

    # Build posterior dict with shape (1, num_particles, ...)
    # physical_samples is a dict of arrays with leading dim = num_particles
    posterior_dict = {}

    # Standard CW parameters
    for name in ["log10_h0", "alpha_gw", "sin_delta_gw", "delta_gw",
                 "log10_f_gw", "cos_iota", "psi", "Phi0"]:
        if name in physical_samples:
            vals = physical_samples[name]
            posterior_dict[name] = np.asarray(jnp.expand_dims(vals, axis=0))

    # Chi parameters
    if "chi" in physical_samples:
        vals = physical_samples["chi"]
        if vals.ndim == 2 and vals.shape[-1] == n_pulsars:
            posterior_dict["chi"] = np.asarray(jnp.expand_dims(vals, axis=0))

    # Noise parameters
    for name in ["log10_γp", "log10_σp", "efac", "equad",
                 "log10_gamma_p_mean", "log10_gamma_p_std",
                 "log10_ratio_mean", "log10_ratio_std",
                 "log10_equad"]:
        if name in physical_samples:
            vals = physical_samples[name]
            posterior_dict[name] = np.asarray(jnp.expand_dims(vals, axis=0))

    inf_data = az.from_dict(posterior=posterior_dict)
    return inf_data


# ---------------------------------------------------------------------------
# SMC diagnostics
# ---------------------------------------------------------------------------

def save_smc_diagnostics(smc_results, output_dir, output_id):
    """Save SMC-specific diagnostic data.

    Parameters
    ----------
    smc_results : dict
        Output from run_tempered_smc.
    output_dir : str
        Output directory.
    output_id : str
        Output identifier.

    Returns
    -------
    str
        Path to saved diagnostics JSON.
    """
    diagnostics = {
        "log_evidence": smc_results["log_evidence"],
        "temperature_schedule": smc_results["temperature_schedule"],
        "wall_time_seconds": smc_results["wall_time"],
        "num_rungs": len(smc_results["intermediate_states"]),
        "intermediate_states": smc_results["intermediate_states"],
    }

    path = os.path.join(output_dir, f"{output_id}_smc_diagnostics.json")
    with open(path, "w") as f:
        json.dump(diagnostics, f, indent=2)

    logger.info(f"SMC diagnostics saved to {path}")
    return path


def plot_smc_diagnostics(smc_results, output_dir, output_id):
    """Create SMC diagnostic plots.

    Parameters
    ----------
    smc_results : dict
        Output from run_tempered_smc.
    output_dir : str
        Output directory.
    output_id : str
        Output identifier.

    Returns
    -------
    str
        Path to saved plot.
    """
    states = smc_results["intermediate_states"]
    lambdas = [s["tempering_param"] for s in states]
    log_z_cumulative = [s["log_evidence_cumulative"] for s in states]
    increments = [s["log_likelihood_increment"] for s in states]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Evidence accumulation
    axes[0].plot(lambdas, log_z_cumulative, "o-", markersize=3)
    axes[0].set_xlabel(r"Tempering parameter $\lambda$")
    axes[0].set_ylabel(r"Cumulative $\log Z$")
    axes[0].set_title("Evidence accumulation")

    # Per-rung log-likelihood increments
    axes[1].plot(lambdas, increments, "o-", markersize=3)
    axes[1].set_xlabel(r"Tempering parameter $\lambda$")
    axes[1].set_ylabel(r"$\log$ likelihood increment")
    axes[1].set_title("Per-rung increments")

    plt.tight_layout()
    path = os.path.join(output_dir, f"{output_id}_smc_diagnostics.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"SMC diagnostic plot saved to {path}")
    return path
