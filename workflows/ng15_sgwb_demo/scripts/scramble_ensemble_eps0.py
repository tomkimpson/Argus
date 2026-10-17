#!/usr/bin/env python
"""Sky-scramble ENSEMBLE for the path-sampling integrand at eps = 0 (CURN end).

Why this exists
---------------
One sky scramble on MDC2 1b gave lnB = +0.35 against the true ORF's +0.69. With HD
evidence this weak, one scramble cannot say whether the true geometry is special: the
scramble-to-scramble scatter is unknown. Running a full 5-rung ladder per scramble is
~30 GPU-hours each, so this script builds the ensemble from the eps = 0 rung alone.

Why eps = 0 is enough to rank geometries
----------------------------------------
At eps = 0 the effective ORF is C(0) = I for EVERY candidate ORF, so the eps = 0
posterior is the CURN posterior and does not depend on which ORF is being tested. The
integrand is

    d lnL / d eps |_{eps=0} = sum_ij  G_ij (Γ - I)_ij ,   G = ∂ lnL / ∂C  at C = I,

by the chain rule through C(eps) = (1 - eps) I + eps Γ. That is EXACTLY linear in Γ.
So one reverse-mode gradient per draw (with respect to the whole correlation matrix)
gives G, and every scramble is then scored by a dot product. The scramble count is
free; the cost is one integrand evaluation of the eps000 rung.

This is a proxy: it ranks geometries by the slope of lnZ(eps) at CURN, not by lnB.
The one full scrambled ladder had a null integrand ~2x below the signal's at every
rung, which is why the slope at eps = 0 is expected to track lnB -- but a percentile
here is not an lnB percentile.

Validation (built in, run before the ensemble is trusted)
---------------------------------------------------------
The contraction is an algebraic identity, so on a subset of draws it must agree per
draw (to ~1e-8, not just in the mean) with the stock estimator's scalar
d lnL/d eps at eps = 0, for the true ORF and for a reference scramble.

    JAX_PLATFORMS=cpu python scripts/scramble_ensemble_eps0.py \
        --rung outputs/mdc2_d1_flat_diffuse_eps000/mdc2_d1_flat_diffuse_eps000_results.nc \
        --config configs/mdc2_d1_flat_diffuse_eps000.ini \
        --scrambles data/scrambles/mdc2_d1_scrambles_ens1000.npz \
        --reference-scramble data/scrambles/mdc2_d1_scrambles.npz \
        --out outputs/scramble_ensemble_eps0_mdc2_d1_flat_diffuse
"""

import argparse
import configparser
import copy
import importlib.util
import json
import logging
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PHYSICAL_SITES = ("log10_ha", "log10_gamma_a", "log10_γp", "log10_σp", "efac", "equad")
VALIDATION_RTOL = 1e-6


def _load(name):
    """Import a sibling script by path (they are scripts, not package modules)."""
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(HERE, name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_filter(config_path):
    """The Kalman filter exactly as the path-sampling readout builds it (true ORF)."""
    from argus import io_manager, utils, workflow

    config = configparser.ConfigParser()
    if not config.read(config_path):
        raise ValueError(f"Could not read config {config_path}")
    config = utils.resolve_config_paths(config, config_path)
    if not config.has_option("PriorModel", "timing_prior"):
        raise ValueError(f"{config_path} does not set [PriorModel] timing_prior.")
    io_manager.setup_single_logger(config, enable_file_logging=False)
    _, kalman_filter = workflow.setup_data_and_kalman_filter(
        config, logging.getLogger("scramble_ensemble_eps0"), use_gw=True
    )
    return kalman_filter


def load_draws(nc_path):
    """Flatten the rung's physical sites to (n_draws, ...) and check eps = 0."""
    import arviz as az

    post = az.from_netcdf(nc_path).posterior
    eps = np.unique(np.round(np.asarray(post["orf_epsilon"].values, dtype=float), 12))
    if eps.size != 1 or eps.item() != 0.0:
        raise ValueError(f"{nc_path} is not an eps = 0 rung (eps values {eps}).")
    n_chain, n_draw = post.sizes["chain"], post.sizes["draw"]
    flat = {}
    for site in PHYSICAL_SITES:
        arr = np.asarray(post[site].values)
        flat[site] = arr.reshape(n_chain * n_draw, *arr.shape[2:])
    return flat, n_chain, n_draw


def _with_orf(kalman_filter, orf):
    """A shallow copy of the filter carrying a different correlation matrix."""
    swapped = copy.copy(kalman_filter)
    swapped.hellings_downs_matrix = orf
    return swapped


def make_gradient_fns(kalman_filter):
    """Two per-draw functions: G = ∂lnL/∂C at C = I, and the stock eps-derivative."""
    import jax
    import jax.numpy as jnp

    from argus import bayesian_inference

    n_psr = kalman_filter.hellings_downs_matrix.shape[0]
    identity = jnp.eye(n_psr, dtype=jnp.float64)

    def grad_wrt_orf(log10_ha, log10_gamma_a, log10_gp, log10_sp, efac, equad):
        # orf_epsilon=None: the matrix is used as-is, so differentiating with respect
        # to it at I is ∂lnL/∂C at C(0). (Passing eps = 0 instead would multiply the
        # input by zero and return an identically zero gradient.)
        def loglik(orf):
            return bayesian_inference.log_likelihood_fn(
                _with_orf(kalman_filter, orf),
                log10_ha,
                log10_gamma_a,
                log10_gp,
                log10_sp,
                efac,
                equad,
                orf_epsilon=None,
            )

        return jax.grad(loglik)(identity)

    def eps_derivative(orf, log10_ha, log10_gamma_a, log10_gp, log10_sp, efac, equad):
        # The stock estimator's integrand (lnb_path_sampling.evaluate_integrand), with
        # the ORF passed in so a scramble can be checked the same way.
        def loglik(e):
            return bayesian_inference.log_likelihood_fn(
                _with_orf(kalman_filter, orf),
                log10_ha,
                log10_gamma_a,
                log10_gp,
                log10_sp,
                efac,
                equad,
                orf_epsilon=e,
            )

        return jax.grad(loglik)(0.0)

    grad_batched = jax.jit(jax.vmap(grad_wrt_orf))
    eps_batched = jax.jit(jax.vmap(eps_derivative, in_axes=(None, 0, 0, 0, 0, 0, 0)))
    return grad_batched, eps_batched


def batched_apply(fn, flat, n_draws, batch_size, *leading):
    pieces = []
    for start in range(0, n_draws, batch_size):
        stop = min(start + batch_size, n_draws)
        args = [flat[site][start:stop] for site in PHYSICAL_SITES]
        pieces.append(np.asarray(fn(*leading, *args), dtype=float))
        print(f"    draws {stop}/{n_draws}", flush=True)
    return np.concatenate(pieces)


def score(gradients, orfs):
    """Per-draw integrand for each ORF: sum over the FULL matrix of G ∘ (Γ - I)."""
    n_psr = gradients.shape[-1]
    offsets = np.asarray(orfs, dtype=float) - np.eye(n_psr)
    return gradients.reshape(gradients.shape[0], -1) @ offsets.reshape(len(orfs), -1).T


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rung", required=True, help="The eps000 rung's results .nc.")
    parser.add_argument("--config", required=True, help="That rung's config.")
    parser.add_argument("--scrambles", required=True, help="Ensemble .npz to score.")
    parser.add_argument(
        "--reference-scramble",
        default=None,
        help="Optional .npz whose scramble 0 is also validated per draw (the one "
        "the full null ladder used).",
    )
    parser.add_argument("--n-validate", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--out", required=True, help="Output prefix (.npz and .json).")
    args = parser.parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)

    path_sampling = _load("lnb_path_sampling")
    sky = _load("sky_scrambles")

    kalman_filter = build_filter(args.config)
    true_orf = np.asarray(kalman_filter.hellings_downs_matrix, dtype=float)
    flat, n_chain, n_draw = load_draws(args.rung)
    n_draws = n_chain * n_draw
    print(
        f"rung: {args.rung}\n  {n_chain} chains x {n_draw} draws, {true_orf.shape[0]} pulsars"
    )

    with np.load(args.scrambles) as archive:
        scramble_orfs = np.asarray(archive["orfs"], dtype=float)
        archive_true = np.asarray(archive["true_orf"], dtype=float)
    if not np.allclose(archive_true, true_orf, rtol=0, atol=1e-12):
        raise ValueError("Ensemble archive's true_orf differs from the config's ORF.")
    matches = np.array([sky.orf_match(o, true_orf) for o in scramble_orfs])
    if matches.max() >= sky.DEFAULT_MATCH_THRESHOLD:
        raise ValueError(
            f"Ensemble contains a scramble with match {matches.max():.3f}."
        )

    grad_batched, eps_batched = make_gradient_fns(kalman_filter)

    # --- validation: contraction == stock eps-derivative, per draw ---------------
    check = {"true_orf": true_orf}
    if args.reference_scramble:
        with np.load(args.reference_scramble) as archive:
            check["reference_scramble_0"] = np.asarray(archive["orfs"][0], dtype=float)
    n_val = min(args.n_validate, n_draws)
    val_flat = {site: flat[site][:n_val] for site in PHYSICAL_SITES}
    print(f"\nvalidating on {n_val} draws")
    val_grads = batched_apply(grad_batched, val_flat, n_val, args.batch_size)
    validation = {}
    for name, orf in check.items():
        direct = batched_apply(eps_batched, val_flat, n_val, args.batch_size, orf)
        contracted = score(val_grads, [orf])[:, 0]
        max_abs = float(np.max(np.abs(contracted - direct)))
        max_rel = float(max_abs / max(np.max(np.abs(direct)), 1e-300))
        validation[name] = {"max_abs_diff": max_abs, "max_rel_diff": max_rel}
        print(
            f"  {name}: max |contracted - direct| = {max_abs:.3e} (rel {max_rel:.3e})"
        )
        if max_rel > VALIDATION_RTOL:
            raise SystemExit(f"VALIDATION FAILED for {name}; not scoring the ensemble.")

    # --- the gradient matrix for every draw --------------------------------------
    print(f"\ncomputing ∂lnL/∂C at C = I for all {n_draws} draws")
    gradients = batched_apply(grad_batched, flat, n_draws, args.batch_size)

    all_orfs = [true_orf] + list(scramble_orfs)
    if "reference_scramble_0" in check:
        all_orfs.append(check["reference_scramble_0"])
    per_draw = score(gradients, all_orfs)  # (n_draws, n_orfs)

    def summary(column):
        x = per_draw[:, column].reshape(n_chain, n_draw)
        ess = path_sampling.effective_sample_size(x)
        return float(x.mean()), float(x.std(ddof=1) / np.sqrt(max(ess, 1.0))), ess

    true_mean, true_err, true_ess = summary(0)
    n_scr = len(scramble_orfs)
    scr = np.array([summary(1 + k) for k in range(n_scr)])
    scr_means = scr[:, 0]
    n_ge = int(np.sum(scr_means >= true_mean))
    p_value = (n_ge + 1) / (n_scr + 1)
    percentile = float(100.0 * np.mean(scr_means < true_mean))
    z = float((true_mean - scr_means.mean()) / scr_means.std(ddof=1))

    report = {
        "rung": args.rung,
        "config": args.config,
        "scrambles": args.scrambles,
        "n_scrambles": n_scr,
        "n_draws": n_draws,
        "validation": validation,
        "true_orf": {"mean": true_mean, "mc_uncert": true_err, "ess": true_ess},
        "scrambles_summary": {
            "mean": float(scr_means.mean()),
            "std": float(scr_means.std(ddof=1)),
            "median": float(np.median(scr_means)),
            "min": float(scr_means.min()),
            "max": float(scr_means.max()),
            "q05_q95": [float(q) for q in np.quantile(scr_means, [0.05, 0.95])],
            "typical_mc_uncert": float(np.median(scr[:, 1])),
            "match_mean": float(matches.mean()),
        },
        "n_scrambles_ge_true": n_ge,
        "p_value_one_sided": p_value,
        "true_percentile": percentile,
        "z_vs_ensemble": z,
    }
    if "reference_scramble_0" in check:
        ref_mean, ref_err, _ = summary(len(all_orfs) - 1)
        report["reference_scramble_0"] = {"mean": ref_mean, "mc_uncert": ref_err}

    print("\n=== eps = 0 integrand: true ORF vs sky-scramble ensemble ===")
    print(f"  true ORF        : {true_mean:.4f} +/- {true_err:.4f}")
    if "reference_scramble_0" in report:
        print(f"  ref scramble 0  : {report['reference_scramble_0']['mean']:.4f}")
    s = report["scrambles_summary"]
    print(
        f"  {n_scr} scrambles : mean {s['mean']:.4f}, std {s['std']:.4f}, "
        f"5-95% [{s['q05_q95'][0]:.4f}, {s['q05_q95'][1]:.4f}], max {s['max']:.4f}"
    )
    print(
        f"  true ORF percentile {percentile:.1f}%, {n_ge} scrambles >= true, "
        f"one-sided p = {p_value:.4f}, z = {z:.2f}"
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(
        args.out + ".npz",
        gradients=gradients,
        scramble_means=scr_means,
        scramble_mc_uncert=scr[:, 1],
        matches=matches,
        true_orf=true_orf,
    )
    with open(args.out + ".json", "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"\n  wrote {args.out}.npz and .json")


if __name__ == "__main__":
    sys.exit(main())
