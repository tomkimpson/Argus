#!/usr/bin/env python
"""Profile the Kalman likelihood along the GW pivot log-PSD, at stored posterior draws.

Why this exists
---------------
The 1b eps=0 (CURN) posterior on the pivot log-PSD s is bimodal relative to its
prior: there is a peak near the injected value (s ~ -7), a deficit around s ~ -9 and
an EXCESS at low amplitude (s < -10). The low-amplitude excess is what makes
``lnb_gw_vs_noise.py`` return ln B(CURN/noise) < 0. It means the marginal likelihood
is non-monotone in s: adding a small common process makes the fit worse than none at
all. This script asks whether that is true at fixed noise parameters, or only after
the per-pulsar red noise re-adjusts.

For each selected draw, the GW pivot log-PSD is swept over a grid while gamma_a, the
red noise and the white noise stay at the draw's values. log10_ha is derived from
(s, gamma_a) exactly as the ridge model does. Each curve is referenced to the same
draw's noise-only likelihood (log10_ha = -30), so

    dlogL(s) = ln L(s, rest) - ln L_noise(rest)

is 0 when the GW is irrelevant, positive where the data want a common process.
Draws are split by where they sit in s (low mode, middle, high mode) so the two
modes' noise configurations can be compared. The same sweep is repeated with gamma_a
replaced by fixed values, to separate the spectral shape from the amplitude.

Run on a GPU node (one A100 is plenty):
    python scripts/diag_gw_amplitude_profile.py \
        --run outputs/mdc2_d1_flat_uprior_eps000 \
        --out outputs/diag_gw_amplitude_profile_mdc2_d1_flat_uprior_eps000
"""

import argparse
import configparser
import glob
import json
import logging
import math
import os

import numpy as np

SITES = ("log10_gamma_a", "log10_γp", "log10_σp", "efac", "equad")
NOISE_ONLY_LOG10_HA = -30.0
DEFAULT_GRID = np.round(np.arange(-15.0, -4.99, 0.25), 3)
DEFAULT_FIXED_GAMMA = (-10.0, -8.5, -7.0)
GROUPS = {"low": (-np.inf, -10.0), "mid": (-10.0, -8.0), "high": (-8.0, np.inf)}


def ridge_log10_ha(s, log10_gamma_a, w):
    """Invert the two-sided OU pivot PSD for log10_ha (as parameter_sampling does)."""
    ga = 10.0**log10_gamma_a
    return 0.5 * (
        math.log10(12.0) + s + 2.0 * math.log10(w)
        + np.log10(ga**2 + w**2) - log10_gamma_a
    )


def select_draws(post, per_group, seed):
    """Indices (chain, draw) spread across chains, ``per_group`` from each s band."""
    s = np.asarray(post["log10_pivot_psd"].values)
    rng = np.random.default_rng(seed)
    out = {}
    for name, (lo, hi) in GROUPS.items():
        idx = np.argwhere((s >= lo) & (s < hi))
        take = min(per_group, len(idx))
        out[name] = idx[rng.choice(len(idx), size=take, replace=False)] if take else idx
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", required=True, help="Run output dir (.nc + .ini).")
    p.add_argument("--out", required=True, help="Output prefix (.npz/.json/.png).")
    p.add_argument("--per-group", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--fixed-gamma",
        type=float,
        nargs="*",
        default=list(DEFAULT_FIXED_GAMMA),
        help="Extra sweeps with log10_gamma_a fixed at these values.",
    )
    args = p.parse_args()

    import arviz as az
    import jax
    import jax.numpy as jnp

    from argus import bayesian_inference, io_manager, utils, workflow

    run_dir = args.run.rstrip("/")
    ini = sorted(glob.glob(os.path.join(run_dir, "*.ini")))
    nc = sorted(glob.glob(os.path.join(run_dir, "*_results.nc")))
    if len(ini) != 1 or len(nc) != 1:
        raise SystemExit(f"{run_dir}: need exactly one .ini and one *_results.nc")
    config = configparser.ConfigParser()
    config.read(ini[0])
    if config.get("PriorModel", "gw_parameterization", fallback="direct") != "ridge":
        raise SystemExit("needs a ridge-parameterised run")
    # The copied .ini sits one level below the original config; data paths are
    # relative to the workflow directory, so resolve against a sibling of configs/.
    config = utils.resolve_config_paths(
        config, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                             "configs", os.path.basename(ini[0]))
    )
    f_piv = config.getfloat("PriorModel", "gw_pivot_freq_hz")
    w = 2.0 * math.pi * f_piv

    io_manager.setup_single_logger(config, enable_file_logging=False)
    logger = logging.getLogger("diag_gw_amplitude_profile")
    _, kalman_filter = workflow.setup_data_and_kalman_filter(config, logger, use_gw=True)

    post = az.from_netcdf(nc[0]).posterior
    eps = float(np.unique(np.round(post["orf_epsilon"].values, 12)).item())
    picks = select_draws(post, args.per_group, args.seed)

    def loglik(log10_ha, log10_gamma_a, lgp, lsp, efac, equad):
        return bayesian_inference.log_likelihood_fn(
            kalman_filter, log10_ha, log10_gamma_a, lgp, lsp, efac, equad,
            orf_epsilon=eps,
        )

    # vmap over the grid of (log10_ha, log10_gamma_a) at one draw's noise.
    sweep = jax.jit(jax.vmap(loglik, in_axes=(0, 0, None, None, None, None)))

    grid = np.asarray(DEFAULT_GRID, dtype=float)
    variants = ["own"] + [f"{g:+.1f}" for g in args.fixed_gamma]
    results = {}
    for group, idx in picks.items():
        curves = np.full((len(idx), len(variants), grid.size), np.nan)
        draw_s = []
        for i, (c, d) in enumerate(idx):
            vals = {site: np.asarray(post[site].values[c, d]) for site in SITES}
            draw_s.append(float(post["log10_pivot_psd"].values[c, d]))
            noise = [jnp.asarray(vals[k]) for k in SITES[1:]]
            gammas = [float(vals["log10_gamma_a"])] + list(args.fixed_gamma)
            for j, lga in enumerate(gammas):
                lha = ridge_log10_ha(grid, lga, w)
                lga_vec = np.full(grid.size + 1, lga)
                lha_vec = np.concatenate([[NOISE_ONLY_LOG10_HA], lha])
                ll = np.asarray(sweep(jnp.asarray(lha_vec), jnp.asarray(lga_vec), *noise))
                curves[i, j] = ll[1:] - ll[0]
            print(f"  {group} draw {i + 1}/{len(idx)} (s = {draw_s[-1]:.2f}): "
                  f"max dlogL(own) = {np.nanmax(curves[i, 0]):+.2f} at "
                  f"s = {grid[np.nanargmax(curves[i, 0])]:.2f}", flush=True)
        results[group] = {"curves": curves, "draw_s": np.asarray(draw_s)}

    np.savez(
        f"{args.out}.npz",
        grid=grid,
        variants=np.asarray(variants),
        **{f"{g}_curves": r["curves"] for g, r in results.items()},
        **{f"{g}_draw_s": r["draw_s"] for g, r in results.items()},
    )

    summary = {"run": os.path.basename(run_dir), "orf_epsilon": eps,
               "grid": grid.tolist(), "variants": variants, "groups": {}}
    for g, r in results.items():
        cv = r["curves"]
        if cv.size == 0:
            continue
        med = np.nanmedian(cv, axis=0)
        summary["groups"][g] = {
            "n_draws": int(cv.shape[0]),
            "median_dlogl": {v: med[j].tolist() for j, v in enumerate(variants)},
            "frac_with_dip_below_noise": {
                v: float(np.mean(np.nanmin(cv[:, j], axis=1) < -0.5))
                for j, v in enumerate(variants)
            },
            "median_peak_dlogl": {
                v: float(np.nanmedian(np.nanmax(cv[:, j], axis=1)))
                for j, v in enumerate(variants)
            },
        }
    with open(f"{args.out}.json", "w") as f:
        json.dump(summary, f, indent=2)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(variants), figsize=(4.2 * len(variants), 4),
                             sharey=True)
    colours = {"low": "C0", "mid": "C1", "high": "C3"}
    for j, v in enumerate(variants):
        ax = axes[j]
        for g, r in results.items():
            for k, curve in enumerate(r["curves"][:, j]):
                ax.plot(grid, curve, color=colours[g], alpha=0.25, lw=0.8,
                        label=f"{g}-s draws" if k == 0 else None)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"gamma_a: {v}")
        ax.set_xlabel("log10 pivot PSD (two-sided)")
    axes[0].set_ylabel("ln L(s) - ln L(noise-only), noise at draw")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=130)
    print(f"wrote {args.out}.npz/.json/.png")


if __name__ == "__main__":
    main()
