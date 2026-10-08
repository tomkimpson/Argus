#!/usr/bin/env python
"""Kalman vs static-Gaussian GW likelihood gain at the same posterior draws (MDC2 1b).

Why this exists
---------------
``diag_curn_static_lnb.py`` integrates Argus's own model (OU GW under the NUTS prior, OU
red noise, white noise at truth) as a static Gaussian, with the timing model projected
out, and gets lnB(CURN / noise) ~ +4.6 on MDC2 1b. Argus's sampled estimate is -0.47. So
either the Kalman likelihood holds less GW information than the static Gaussian, or the
likelihood is fine and the loss is in the sampling or the lnB estimator.

This compares the two likelihoods directly. At each selected posterior draw (low,
middle and high pivot modes, as in ``diag_gw_amplitude_profile.py``) it sweeps the GW
pivot log-PSD s with everything else at the draw's values and records

    dlogL(s) = ln L(s, rest) - ln L(GW off, rest)

from the Kalman filter (``bayesian_inference.log_likelihood_fn``, GW off = log10_ha -30)
and from the static Gaussian (per-pulsar G^T r ~ N(0, W + RN_OU + 10^s GW_OU), GW off =
no GW term). Constants, including the Kalman timing-model prior, cancel in each
difference. If the curves agree, the likelihood is fine. If the Kalman curve is lower,
the filter is losing GW information. Both use CURN, with eps fixed at the run's value.

Run on a GPU node:
    python scripts/diag_kalman_vs_static.py --run outputs/mdc2_d1_flat_uprior_eps000 \
        --out outputs/diag_kalman_vs_static_mdc2_d1_flat_uprior_eps000
"""

import argparse
import configparser
import glob
import json
import logging
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_spectral_prior_volume as spv  # noqa: E402
from diag_curn_static_lnb import ou_gw_unit  # noqa: E402
from diag_gw_amplitude_profile import (  # noqa: E402
    NOISE_ONLY_LOG10_HA,
    SITES,
    ridge_log10_ha,
    select_draws,
)

GRID = np.round(np.arange(-13.0, -4.99, 0.25), 3)


def static_curve(psrs, grid, lga, lgp, lsp, efac, equad):
    """Static-Gaussian dlogL(s) summed over pulsars, at one draw's noise."""
    total = np.zeros(grid.size)
    amps = np.concatenate([[0.0], 10.0**grid])
    for i, p in enumerate(psrs):
        G, t = p["G"], p["toas"]
        white = (efac[i] * p["sig"]) ** 2 + equad[i] ** 2
        rn = 10.0 ** (2 * lsp[i]) * spv.ou_covariance(t, lgp[i], p["f0"])
        base = G.T @ (np.diag(white) + rn) @ G
        gw = G.T @ ou_gw_unit(t, lga) @ G
        ll, _ = spv.amplitude_line(base, gw, amps, G.T @ p["residuals"], None)
        total += ll[1:] - ll[0]
    return total


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True, help="Output prefix (.json/.npz/.png).")
    p.add_argument("--per-group", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timing-prior", choices=("informative", "diffuse"), default=None,
                   help="Override the run's Kalman timing prior. 'diffuse' is the flat-prior "
                        "limit, which projects the timing model out as the static side does.")
    args = p.parse_args()

    import arviz as az
    import jax
    import jax.numpy as jnp
    import pyarrow.feather as pf

    from argus import bayesian_inference, io_manager, utils, workflow

    run_dir = args.run.rstrip("/")
    ini = sorted(glob.glob(os.path.join(run_dir, "*.ini")))
    nc = sorted(glob.glob(os.path.join(run_dir, "*_results.nc")))
    config = configparser.ConfigParser()
    config.read(ini[0])
    config = utils.resolve_config_paths(
        config, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                             "configs", os.path.basename(ini[0])))
    w = 2.0 * math.pi * config.getfloat("PriorModel", "gw_pivot_freq_hz")
    if args.timing_prior:
        config.set("PriorModel", "timing_prior", args.timing_prior)
    timing_prior = config.get("PriorModel", "timing_prior", fallback="informative")
    print(f"Kalman timing prior: {timing_prior}")
    io_manager.setup_single_logger(config, enable_file_logging=False)
    logger = logging.getLogger("diag_kalman_vs_static")
    _, kalman_filter = workflow.setup_data_and_kalman_filter(config, logger, use_gw=True)

    # Static side: same feathers, same (sorted) order as the loader.
    data_dir = config.get("Data", "data_path")
    psrs = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.feather"))):
        t = pf.read_table(f)
        meta = json.loads(t.schema.metadata[b"json"])
        mcols = [c for c in t.column_names if c.startswith("Mmat_")]
        mmat = np.column_stack([t.column(c).to_numpy() for c in mcols]).astype(float)
        psrs.append({
            "name": meta["name"], "f0": float(meta["F0"]),
            "toas": t.column("toas").to_numpy().astype(float),
            "sig": t.column("toaerrs").to_numpy().astype(float),
            "residuals": t.column("residuals").to_numpy().astype(float),
            "G": spv.projector(mmat / np.linalg.norm(mmat, axis=0)),
        })

    post = az.from_netcdf(nc[0]).posterior
    eps = float(np.unique(np.round(post["orf_epsilon"].values, 12)).item())
    picks = select_draws(post, args.per_group, args.seed)

    def loglik(log10_ha, log10_gamma_a, lgp, lsp, efac, equad):
        return bayesian_inference.log_likelihood_fn(
            kalman_filter, log10_ha, log10_gamma_a, lgp, lsp, efac, equad,
            orf_epsilon=eps)

    sweep = jax.jit(jax.vmap(loglik, in_axes=(0, 0, None, None, None, None)))

    rows = []
    curves = {"kalman": [], "static": []}
    for group, idx in picks.items():
        for c, d in idx:
            v = {s: np.asarray(post[s].values[c, d]) for s in SITES}
            lga = float(v["log10_gamma_a"])
            lha = np.concatenate([[NOISE_ONLY_LOG10_HA], ridge_log10_ha(GRID, lga, w)])
            ll = np.asarray(sweep(jnp.asarray(lha), jnp.asarray(np.full(lha.size, lga)),
                                  *[jnp.asarray(v[k]) for k in SITES[1:]]))
            kal = ll[1:] - ll[0]
            sta = static_curve(psrs, GRID, lga, v["log10_γp"], v["log10_σp"],
                               v["efac"], v["equad"])
            curves["kalman"].append(kal)
            curves["static"].append(sta)
            row = {
                "group": group, "chain": int(c), "draw": int(d),
                "draw_s": float(post["log10_pivot_psd"].values[c, d]), "lga": lga,
                "kalman_max": float(kal.max()), "kalman_argmax_s": float(GRID[kal.argmax()]),
                "static_max": float(sta.max()), "static_argmax_s": float(GRID[sta.argmax()]),
                "max_abs_diff": float(np.max(np.abs(kal - sta))),
            }
            rows.append(row)
            print(f"  {group:4s} s={row['draw_s']:6.2f} lga={lga:6.2f}  "
                  f"max dlogL Kalman {row['kalman_max']:+7.2f} @ {row['kalman_argmax_s']:.2f}"
                  f"  static {row['static_max']:+7.2f} @ {row['static_argmax_s']:.2f}"
                  f"  max|diff| {row['max_abs_diff']:.2f}", flush=True)

    kal, sta = np.asarray(curves["kalman"]), np.asarray(curves["static"])
    summary = {"run": os.path.basename(run_dir), "orf_epsilon": eps, "grid": GRID.tolist(),
               "timing_prior": timing_prior,
               "median_kalman_minus_static_at_peak": float(np.median(
                   [r["kalman_max"] - r["static_max"] for r in rows])),
               "draws": rows}
    with open(f"{args.out}.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.savez(f"{args.out}.npz", grid=GRID, kalman=kal, static=sta,
             groups=np.asarray([r["group"] for r in rows]))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)
    for ax, g in zip(axes, ("low", "mid", "high")):
        sel = [i for i, r in enumerate(rows) if r["group"] == g]
        for k, i in enumerate(sel):
            ax.plot(GRID, kal[i], "C0", alpha=0.7, label="Kalman" if k == 0 else None)
            ax.plot(GRID, sta[i], "C1--", alpha=0.7, label="static" if k == 0 else None)
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(-7.21, color="grey", ls=":", lw=0.8)
        ax.set_title(f"{g}-mode draws (Kalman timing prior: {timing_prior})")
        ax.set_xlabel("GW pivot log-PSD s (two-sided)")
    axes[0].set_ylabel("dlogL(s) vs GW off")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=110)
    print(f"wrote {args.out}.json/.npz/.png")


if __name__ == "__main__":
    main()
