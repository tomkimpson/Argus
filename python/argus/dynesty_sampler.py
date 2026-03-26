"""Dynesty nested sampling for CW gravitational wave source parameter estimation.

Dynesty is a pure Python nested sampler that calls the likelihood as a black box,
avoiding the JIT compilation overhead of JAX-native samplers (jaxns, blackjax).
The JAX Kalman filter likelihood is JIT-compiled once for a single evaluation
(seconds, not hours) and then called repeatedly by dynesty.
"""

import logging
import time
from functools import partial

import arviz as az
import dynesty
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import ndtri

import tensorflow_probability.substrates.jax as tfp

from .bayesian_inference import cw_log_likelihood_fn

jax.config.update("jax_enable_x64", True)
tfpd = tfp.distributions

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter layout: maps prior_specs → flat array structure for dynesty
# ---------------------------------------------------------------------------

def build_param_layout(prior_specs, n_pulsars):
    """Build a parameter layout describing the flat array structure for dynesty.

    Returns the parameter names, bounds, indices, and fixed values needed
    to construct the prior_transform and likelihood wrapper.

    Parameters
    ----------
    prior_specs : dict
        Prior specifications (same format as used by all samplers).
    n_pulsars : int

    Returns
    -------
    dict with keys:
        "names": list of (name, size) tuples for free params
        "ndim": total number of free parameters
        "fixed_values": dict of fixed parameter values
        "transforms": list of transform descriptors for prior_transform
        "has_hierarchical_gamma": bool
        "has_hierarchical_ratio": bool
        "equad_use_log10": bool
    """
    transforms = []  # list of dicts describing each param block
    fixed_values = {}
    idx = 0

    cw_specs = prior_specs.get("cw_specs", {})
    hierarchical_specs = prior_specs.get("hierarchical_specs")

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
            transforms.append({
                "name": phys_name,
                "type": "uniform",
                "start": idx,
                "size": 1,
                "low": float(tp["min"]),
                "high": float(tp["max"]),
            })
            idx += 1
        else:
            if phys_name == "sin_delta_gw":
                fixed_values["delta_gw"] = float(cw_specs.get("delta_gw_spec", 0.0))
                fixed_values["sin_delta_gw"] = np.sin(fixed_values["delta_gw"])
            else:
                fixed_values[phys_name] = float(cw_specs.get(fixed_key, 0.0))

    # --- Per-pulsar chi ---
    chi_tp = cw_specs.get("chi_transform_params")
    if chi_tp is not None:
        transforms.append({
            "name": "chi",
            "type": "uniform",
            "start": idx,
            "size": n_pulsars,
            "low": float(chi_tp["min"]),
            "high": float(chi_tp["max"]),
        })
        idx += n_pulsars

    # --- Hierarchical gamma_p ---
    has_hierarchical_gamma = False
    if isinstance(prior_specs["log10_gamma_p_spec"], tfpd.Distribution):
        # Fallback: per-pulsar uniform
        spec = prior_specs["log10_gamma_p_spec"]
        transforms.append({
            "name": "log10_gamma_p",
            "type": "uniform",
            "start": idx,
            "size": n_pulsars,
            "low": float(np.asarray(spec.low).flat[0]),
            "high": float(np.asarray(spec.high).flat[0]),
        })
        idx += n_pulsars
    elif prior_specs["log10_gamma_p_spec"] is not None:
        fixed_values["log10_gamma_p"] = np.asarray(prior_specs["log10_gamma_p_spec"])
    else:
        has_hierarchical_gamma = True
        hs = hierarchical_specs
        # Hyperparameter: mean
        transforms.append({
            "name": "log10_gamma_p_mean",
            "type": "uniform",
            "start": idx,
            "size": 1,
            "low": float(hs["log10_gamma_p_mean_spec"].low),
            "high": float(hs["log10_gamma_p_mean_spec"].high),
        })
        gp_mean_idx = idx
        idx += 1
        # Hyperparameter: std
        transforms.append({
            "name": "log10_gamma_p_std",
            "type": "uniform",
            "start": idx,
            "size": 1,
            "low": float(hs["log10_gamma_p_std_spec"].low),
            "high": float(hs["log10_gamma_p_std_spec"].high),
        })
        gp_std_idx = idx
        idx += 1
        # Per-pulsar: Normal(mean, std)
        transforms.append({
            "name": "log10_gamma_p",
            "type": "hierarchical_normal",
            "start": idx,
            "size": n_pulsars,
            "mean_idx": gp_mean_idx,
            "std_idx": gp_std_idx,
        })
        idx += n_pulsars

    # --- Hierarchical ratio (sigma_p) ---
    has_hierarchical_ratio = False
    if isinstance(prior_specs["log10_sigma_p_spec"], tfpd.Distribution):
        spec = prior_specs["log10_sigma_p_spec"]
        transforms.append({
            "name": "log10_sigma_p",
            "type": "uniform",
            "start": idx,
            "size": n_pulsars,
            "low": float(spec.low),
            "high": float(spec.high),
        })
        idx += n_pulsars
    elif prior_specs["log10_sigma_p_spec"] is not None:
        fixed_values["log10_sigma_p"] = np.asarray(prior_specs["log10_sigma_p_spec"])
    else:
        has_hierarchical_ratio = True
        hs = hierarchical_specs
        transforms.append({
            "name": "log10_ratio_mean",
            "type": "uniform",
            "start": idx,
            "size": 1,
            "low": float(hs["log10_ratio_mean_spec"].low),
            "high": float(hs["log10_ratio_mean_spec"].high),
        })
        ratio_mean_idx = idx
        idx += 1
        transforms.append({
            "name": "log10_ratio_std",
            "type": "uniform",
            "start": idx,
            "size": 1,
            "low": float(hs["log10_ratio_std_spec"].low),
            "high": float(hs["log10_ratio_std_spec"].high),
        })
        ratio_std_idx = idx
        idx += 1
        transforms.append({
            "name": "log10_ratio",
            "type": "hierarchical_normal",
            "start": idx,
            "size": n_pulsars,
            "mean_idx": ratio_mean_idx,
            "std_idx": ratio_std_idx,
        })
        idx += n_pulsars

    # --- EFAC ---
    equad_use_log10 = False
    if isinstance(prior_specs["efac_spec"], tfpd.Distribution):
        low = np.asarray(prior_specs["efac_spec"].low).flat[0]
        high = np.asarray(prior_specs["efac_spec"].high).flat[0]
        transforms.append({
            "name": "efac",
            "type": "uniform",
            "start": idx,
            "size": n_pulsars,
            "low": float(low),
            "high": float(high),
        })
        idx += n_pulsars
    else:
        fixed_values["efac"] = np.asarray(prior_specs["efac_spec"])

    # --- EQUAD ---
    if isinstance(prior_specs["equad_spec"], dict) and prior_specs["equad_spec"].get(
        "use_log10", False
    ):
        equad_use_log10 = True
        log10_equad_spec = prior_specs["equad_spec"]["log10_equad_spec"]
        low = np.asarray(log10_equad_spec.low).flat[0]
        high = np.asarray(log10_equad_spec.high).flat[0]
        transforms.append({
            "name": "log10_equad",
            "type": "uniform",
            "start": idx,
            "size": n_pulsars,
            "low": float(low),
            "high": float(high),
        })
        idx += n_pulsars
    elif isinstance(prior_specs["equad_spec"], tfpd.Distribution):
        low = np.asarray(prior_specs["equad_spec"].low).flat[0]
        high = np.asarray(prior_specs["equad_spec"].high).flat[0]
        transforms.append({
            "name": "equad",
            "type": "uniform",
            "start": idx,
            "size": n_pulsars,
            "low": float(low),
            "high": float(high),
        })
        idx += n_pulsars
    else:
        fixed_values["equad"] = np.asarray(prior_specs["equad_spec"])

    return {
        "ndim": idx,
        "transforms": transforms,
        "fixed_values": fixed_values,
        "has_hierarchical_gamma": has_hierarchical_gamma,
        "has_hierarchical_ratio": has_hierarchical_ratio,
        "equad_use_log10": equad_use_log10,
        "n_pulsars": n_pulsars,
    }


# ---------------------------------------------------------------------------
# Prior transform: unit hypercube → physical parameters
# ---------------------------------------------------------------------------

def build_prior_transform(layout):
    """Build dynesty prior_transform function from parameter layout.

    Parameters
    ----------
    layout : dict
        Output from build_param_layout.

    Returns
    -------
    callable
        prior_transform(u) -> numpy array of physical parameters
    """
    transforms = layout["transforms"]
    ndim = layout["ndim"]

    def prior_transform(u):
        theta = np.empty(ndim)
        for t in transforms:
            s = t["start"]
            n = t["size"]
            if t["type"] == "uniform":
                theta[s:s+n] = t["low"] + u[s:s+n] * (t["high"] - t["low"])
            elif t["type"] == "hierarchical_normal":
                mean_val = theta[t["mean_idx"]]  # already transformed
                std_val = theta[t["std_idx"]]
                # Clip to avoid ±inf from ndtri
                u_clipped = np.clip(u[s:s+n], 1e-10, 1.0 - 1e-10)
                theta[s:s+n] = mean_val + std_val * ndtri(u_clipped)
        return theta

    return prior_transform


# ---------------------------------------------------------------------------
# Likelihood wrapper: flat physical array → JAX Kalman filter
# ---------------------------------------------------------------------------

def build_likelihood_fn(cw_kf, layout):
    """Build dynesty log-likelihood function.

    JIT-compiles the CW Kalman filter likelihood for single evaluation,
    then wraps it to unpack the flat parameter array.

    Parameters
    ----------
    cw_kf : CWKalmanFilter
        CW Kalman filter instance.
    layout : dict
        Output from build_param_layout.

    Returns
    -------
    callable
        log_likelihood(theta) -> float
    """
    transforms = layout["transforms"]
    fixed_values = layout["fixed_values"]
    n_pulsars = layout["n_pulsars"]
    equad_use_log10 = layout["equad_use_log10"]
    has_hierarchical_gamma = layout["has_hierarchical_gamma"]
    has_hierarchical_ratio = layout["has_hierarchical_ratio"]

    # Build index lookup
    param_indices = {}
    for t in transforms:
        param_indices[t["name"]] = (t["start"], t["size"])

    # JIT-compile the core likelihood
    jit_ll = jax.jit(partial(cw_log_likelihood_fn, cw_kf))

    def log_likelihood(theta):
        # Helper to extract param from theta or fixed_values
        def get(name, default_size=1):
            if name in param_indices:
                s, n = param_indices[name]
                val = theta[s:s+n]
                return val[0] if n == 1 else val
            return fixed_values.get(name)

        # CW source params
        log10_h0 = get("log10_h0")
        alpha_gw = get("alpha_gw")
        log10_f_gw = get("log10_f_gw")
        cos_iota = get("cos_iota")
        psi = get("psi")
        Phi0 = get("Phi0")

        # delta_gw: derived from sin_delta_gw or fixed
        if "sin_delta_gw" in param_indices:
            sin_delta_gw = get("sin_delta_gw")
            delta_gw = np.arcsin(sin_delta_gw)
        else:
            delta_gw = fixed_values.get("delta_gw", 0.0)

        # Chi
        if "chi" in param_indices:
            chi = get("chi")
        else:
            chi = np.zeros(n_pulsars)

        # Noise: gamma_p
        if "log10_gamma_p" in param_indices:
            log10_gp = get("log10_gamma_p")
        else:
            log10_gp = fixed_values["log10_gamma_p"]

        # Noise: sigma_p
        if "log10_sigma_p" in param_indices:
            log10_sp = get("log10_sigma_p")
        elif has_hierarchical_ratio:
            log10_ratio = get("log10_ratio")
            log10_sp = log10_gp + log10_ratio
        else:
            log10_sp = fixed_values["log10_sigma_p"]

        # EFAC
        efac = get("efac") if "efac" in param_indices else fixed_values["efac"]

        # EQUAD
        if equad_use_log10 and "log10_equad" in param_indices:
            log10_equad = get("log10_equad")
            equad = 10.0 ** log10_equad
        elif "equad" in param_indices:
            equad = get("equad")
        else:
            equad = fixed_values["equad"]

        # Convert to JAX arrays and call
        ll = jit_ll(
            jnp.float64(log10_h0),
            jnp.float64(alpha_gw),
            jnp.float64(delta_gw),
            jnp.float64(log10_f_gw),
            jnp.float64(cos_iota),
            jnp.float64(psi),
            jnp.float64(Phi0),
            jnp.asarray(chi, dtype=jnp.float64),
            jnp.asarray(log10_gp, dtype=jnp.float64),
            jnp.asarray(log10_sp, dtype=jnp.float64),
            jnp.asarray(efac, dtype=jnp.float64),
            jnp.asarray(equad, dtype=jnp.float64),
        )
        return float(ll)

    return log_likelihood


# ---------------------------------------------------------------------------
# Run dynesty
# ---------------------------------------------------------------------------

def run_dynesty_sampling(log_likelihood, prior_transform, ndim, config):
    """Run dynesty nested sampling.

    Parameters
    ----------
    log_likelihood : callable
    prior_transform : callable
    ndim : int
    config : configparser.ConfigParser

    Returns
    -------
    dynesty.results.Results
    """
    nlive = config.getint("Dynesty", "nlive", fallback=500)
    dlogz = config.getfloat("Dynesty", "dlogz", fallback=0.5)
    bound = config.get("Dynesty", "bound", fallback="multi")
    sample = config.get("Dynesty", "sample", fallback="auto")
    seed = config.getint("Dynesty", "seed", fallback=42)
    dynamic = config.getboolean("Dynesty", "dynamic", fallback=False)

    rng = np.random.default_rng(seed)

    print(f"Running dynesty nested sampling...")
    print(f"  ndim: {ndim}")
    print(f"  nlive: {nlive}")
    print(f"  dlogz: {dlogz}")
    print(f"  bound: {bound}")
    print(f"  sample: {sample}")
    print(f"  dynamic: {dynamic}")

    t_start = time.time()

    if dynamic:
        sampler = dynesty.DynamicNestedSampler(
            log_likelihood,
            prior_transform,
            ndim,
            bound=bound,
            sample=sample,
            rstate=rng,
        )
        sampler.run_nested(dlogz_init=dlogz, print_progress=True)
    else:
        sampler = dynesty.NestedSampler(
            log_likelihood,
            prior_transform,
            ndim,
            nlive=nlive,
            bound=bound,
            sample=sample,
            rstate=rng,
        )
        sampler.run_nested(dlogz=dlogz, print_progress=True)

    wall_time = time.time() - t_start
    results = sampler.results

    logger.info(
        f"Dynesty completed: logz={results.logz[-1]:.2f} +/- {results.logzerr[-1]:.2f}, "
        f"ncall={results.ncall}, wall_time={wall_time:.1f}s"
    )

    return results


# ---------------------------------------------------------------------------
# ArviZ conversion
# ---------------------------------------------------------------------------

def dynesty_results_to_arviz(results, layout, num_posterior_samples=10000):
    """Convert dynesty results to ArviZ InferenceData.

    Parameters
    ----------
    results : dynesty.results.Results
    layout : dict
        Output from build_param_layout.
    num_posterior_samples : int

    Returns
    -------
    arviz.InferenceData
    """
    from dynesty.utils import resample_equal

    # Resample to equal-weight posterior samples
    weights = np.exp(results.logwt - results.logz[-1])
    samples = resample_equal(results.samples, weights)

    # Subsample if needed
    if len(samples) > num_posterior_samples:
        rng = np.random.default_rng(0)
        indices = rng.choice(len(samples), size=num_posterior_samples, replace=False)
        samples = samples[indices]

    # Build posterior dict
    transforms = layout["transforms"]
    n_pulsars = layout["n_pulsars"]
    has_hierarchical_ratio = layout["has_hierarchical_ratio"]
    equad_use_log10 = layout["equad_use_log10"]

    param_indices = {}
    for t in transforms:
        param_indices[t["name"]] = (t["start"], t["size"])

    posterior_dict = {}

    for t in transforms:
        s, n = t["start"], t["size"]
        vals = samples[:, s:s+n]
        if n == 1:
            vals = vals[:, 0]
        # Shape: (1, num_samples, ...) — 1 chain
        posterior_dict[t["name"]] = np.expand_dims(vals, axis=0)

    # Derived: delta_gw from sin_delta_gw
    if "sin_delta_gw" in posterior_dict:
        posterior_dict["delta_gw"] = np.arcsin(posterior_dict["sin_delta_gw"])

    # Derived: log10_sigma_p from gamma_p + ratio
    if has_hierarchical_ratio and "log10_ratio" in posterior_dict:
        if "log10_gamma_p" in posterior_dict:
            posterior_dict["log10_sigma_p"] = (
                posterior_dict["log10_gamma_p"] + posterior_dict["log10_ratio"]
            )

    # Derived: equad from log10_equad
    if equad_use_log10 and "log10_equad" in posterior_dict:
        posterior_dict["equad"] = 10.0 ** posterior_dict["log10_equad"]

    inf_data = az.from_dict(posterior=posterior_dict)
    return inf_data
