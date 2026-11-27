#!/usr/bin/env python
"""Reproduce / regression-test the NS near-singular-covariance pathology.

Nested sampling explores the full prior, reaching latent-tail parameter values where the
Kalman innovation covariance goes near-singular and `_log_likelihood` returns a spurious
*huge positive* value (slogdet -> -inf, quadratic stays finite). This probe scans the free
latents over a box (default |z| <= 8, i.e. beyond the 6-sigma NS guard) and reports the max
log-likelihood encountered. A sane likelihood is O(1e4) here; anything >> that is the
pathology.

Usage (CPU, fast for the D=2 fixed-noise configs):
    JAX_PLATFORMS=cpu PYTHONPATH=<repo>/python python ns_pathology_probe.py \
        --config <derived_configs/ns_scal_fixed_N04_...ini> --span 8 --grid 41

Exit 0 if max|logL| stays within --sane-cap (pathology absent/guarded), else 1.
"""

import argparse
import itertools
import os

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import configparser  # noqa: E402
from numpyro.handlers import trace, seed  # noqa: E402
from numpyro.infer.util import log_density  # noqa: E402

from argus import workflow, utils, prior_models, io_manager  # noqa: E402
from argus.bayesian_inference import numpyro_model  # noqa: E402


def build_loglik(config_path):
    cp = configparser.ConfigParser(interpolation=None)
    cp.optionxform = str
    cp.read(config_path)
    logger = io_manager.setup_single_logger(
        cp, output_dir=None, enable_file_logging=False
    )
    pulsar_data, KF = workflow.setup_data_and_kalman_filter(
        cp, logger, use_gw=True, signal_model="gwb"
    )
    n_pulsars = len(pulsar_data["metadata"])
    efac, equad, sigma_p, gamma_p = utils.get_noise_parameters(cp)
    prior_specs = prior_models.get_prior_model_specs(
        cp, n_pulsars, sigma_p, gamma_p, efac, equad, mode="gwb"
    )

    def model():
        numpyro_model(KF, prior_specs, n_pulsars)

    tr = trace(seed(model, jax.random.PRNGKey(0))).get_trace()
    names, shapes = [], []
    for name, site in tr.items():
        if site["type"] == "sample" and not site.get("is_observed", False):
            names.append(name)
            shapes.append(jnp.shape(site["value"]))
    sizes = [int(np.prod(s)) if s else 1 for s in shapes]
    ndim = int(sum(sizes))

    def unpack(z):
        out, off = {}, 0
        for nm, sh, sz in zip(names, shapes, sizes):
            out[nm] = z[off] if sh == () else z[off : off + sz].reshape(sh)
            off += sz
        return out

    def loglik(z):
        lp, _ = log_density(model, (), {}, unpack(z))
        return lp  # log joint; prior is unit-normal (constant scale), fine for magnitude probe

    return jax.jit(loglik), ndim, n_pulsars


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--span", type=float, default=8.0, help="scan each latent in [-span, span]"
    )
    ap.add_argument("--grid", type=int, default=41)
    ap.add_argument("--sane-cap", type=float, default=1e6)
    args = ap.parse_args()

    loglik, ndim, N = build_loglik(args.config)
    if ndim > 2:
        # Only scan the first 2 dims on a grid; hold the rest at 0 (enough to expose the mode).
        print(f"[warn] ndim={ndim} > 2; scanning first 2 latents, rest fixed at 0")
    axis = np.linspace(-args.span, args.span, args.grid)
    worst, worst_z = -np.inf, None
    for zi, zj in itertools.product(axis, axis):
        z = np.zeros(ndim)
        z[0] = zi
        if ndim > 1:
            z[1] = zj
        v = float(loglik(jnp.asarray(z)))
        if np.isfinite(v) and v > worst:
            worst, worst_z = v, (zi, zj)
    print(
        f"config N={N} ndim={ndim}: max logL over |z|<= {args.span} grid = {worst:.4g} "
        f"at z[:2]={worst_z}"
    )
    ok = abs(worst) < args.sane_cap
    print(
        f"PATHOLOGY {'ABSENT (bounded)' if ok else 'PRESENT (spurious huge logL)'} "
        f"[sane cap {args.sane_cap:g}]"
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
