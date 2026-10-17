#!/usr/bin/env python
"""C1: compressed-filter vs dense continuous-time oracle on real pulsars (task 2.5).

    python workflows/ng15_nb/scripts/c1_compression.py --pulsar J1909-3744 \
        [--deltas none 7 14 30] [--scalars 2 1] [--n-points 20] [--seed 2501] \
        [--out workflows/ng15_nb/results/c1/J1909-3744.json]

For each grid width Δ and each layout (`--scalars`: 2 = position and velocity per cell,
the D2 default since the 2026-10-18 amendment; 1 = position only) the pulsar is
compressed (task 2.2) and the Argus marginal filter
(with accum0, task 2.3) is compared with the dense oracle (task 2.4) at the same parameter
points. Reported per Δ:

- spread of ΔlnL = lnL_filter + const − lnL_oracle across the points (max − min), which
  C1 requires to be ≤ 0.01 nat;
- the largest gradient error, max_i ‖g_f − g_o‖∞ / ‖g_o‖∞, which C1 requires ≤ 1e-4.

Two point sets, each `--n-points` draws:
- "posterior" (the C1 gate): random draws of (log10 ha, log10 γa, log10 σp, log10 γp)
  for this pulsar from the Phase 1 Discovery-OU ε = 0 posterior (OU14);
- "prior" (stress test, not gating): uniform draws from the ou.ini prior box, with
  log10 ha drawn to span the same range as the posterior's ±3 dex.

Run on one A100: the oracle's dense n×n Cholesky (n = TOA count) and its gradient need
~5 n² doubles.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "workflows" / "ng15_nb" / "compression"))
import compress  # noqa: E402
import ingest  # noqa: E402
import oracle  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

POSTERIOR = REPO / "workflows/ng15_nb/outputs/ng15nb_ou14_eps000/samples.nc"
PRIOR_BOX = {  # ou.ini [PriorModel]
    "log10_gamma_a": (-11.0, -6.0),
    "log10_sigma_p": (-22.0, -11.5),
    "log10_gamma_p": (-12.0, -6.0),
}


def posterior_points(psr, n, rng):
    import arviz as az

    manifest = json.loads((REPO / "data/ng15_nb/MANIFEST.json").read_text())
    i = [e["pulsar"] for e in manifest["feathers"]].index(psr)
    post = az.from_netcdf(POSTERIOR).posterior
    flat = lambda v: np.asarray(v).reshape(-1, *np.asarray(v).shape[2:])  # noqa: E731
    cols = np.column_stack(
        [
            flat(post["gw_log10_ha"]),
            flat(post["gw_log10_gamma_a"]),
            flat(post["log10_sigma_p"])[:, i],
            flat(post["log10_gamma_p"])[:, i],
        ]
    )
    return cols[rng.choice(cols.shape[0], n, replace=False)]


def prior_points(post_pts, n, rng):
    lo_ha, hi_ha = post_pts[:, 0].min() - 3.0, post_pts[:, 0].max() + 3.0
    u = rng.uniform(size=(n, 4))
    box = [(lo_ha, hi_ha)] + [PRIOR_BOX[k] for k in PRIOR_BOX]
    return np.column_stack([lo + (hi - lo) * u[:, j] for j, (lo, hi) in enumerate(box)])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pulsar", required=True)
    ap.add_argument("--deltas", nargs="*", default=["none", "7", "14", "30"])
    ap.add_argument("--scalars", nargs="*", type=int, default=[2, 1], choices=[1, 2])
    ap.add_argument("--n-points", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2501)
    ap.add_argument("--ingest-dir", default=str(REPO / "workflows/ng15_nb/data/ingest"))
    ap.add_argument("--out")
    args = ap.parse_args()
    out = Path(args.out or REPO / f"workflows/ng15_nb/results/c1/{args.pulsar}.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    cache = ingest.load(Path(args.ingest_dir) / f"{args.pulsar}.npz")
    rng = np.random.default_rng(args.seed)
    sets = {"posterior": posterior_points(args.pulsar, args.n_points, rng)}
    sets["prior"] = prior_points(sets["posterior"], args.n_points, rng)

    # Oracle: one jit per pulsar, in the Δ = DMX-window basis. The filter's basis at
    # another Δ spans the same column space, so any difference is a parameter-independent
    # constant, which the spread removes.
    comp0 = compress.compress_pulsar(cache)
    oi = oracle.oracle_inputs(cache, comp0)
    o_vg = jax.jit(jax.value_and_grad(oracle.oracle_loglike))
    t = time.perf_counter()
    o_vals = {}
    for name, pts in sets.items():
        vg = [o_vg(jnp.asarray(p), oi) for p in pts]
        o_vals[name] = (
            np.array([float(v) for v, _ in vg]),
            np.array([np.asarray(g) for _, g in vg]),
        )
    oracle_s = (time.perf_counter() - t) / sum(len(p) for p in sets.values())

    result = {
        "pulsar": args.pulsar,
        "n_toa": int(cache["toas"].size),
        "seed": args.seed,
        "posterior_source": str(POSTERIOR.relative_to(REPO)),
        "oracle_s_per_eval": oracle_s,
        "device": str(jax.devices()[0]),
        "points": {k: v.tolist() for k, v in sets.items()},
        "deltas": {},
    }
    f_vg = jax.jit(jax.value_and_grad(oracle.filter_loglike))
    for d, nsc in [(d, n) for n in args.scalars for d in args.deltas]:
        delta = None if d == "none" else float(d)
        comp = compress.compress_pulsar(cache, delta_days=delta, velocity=nsc == 2)
        fi = oracle.filter_inputs(comp)
        row = {
            "scalars": nsc,
            "n_cells": int(comp["times"].size),
            "n_velocity": int(comp["has_velocity"].sum()),
            "max_cell_span_days": float(
                max(
                    np.ptp(cache["toas"][comp["cell"] == c])
                    for c in range(comp["cell"].max() + 1)
                )
                / 86400.0
            ),
        }
        for name, pts in sets.items():
            ov, og = o_vals[name]
            vg = [f_vg(jnp.asarray(p), fi) for p in pts]
            fv = np.array([float(v) for v, _ in vg]) + comp["const"]
            fg = np.array([np.asarray(g) for _, g in vg])
            dl = fv - ov
            gerr = np.max(np.abs(fg - og), axis=1) / np.max(np.abs(og), axis=1)
            row[name] = {
                "dlnL": dl.tolist(),
                "dlnL_spread": float(np.ptp(dl)),
                "grad_rel_err": gerr.tolist(),
                "grad_rel_err_max": float(gerr.max()),
                "oracle_lnL": ov.tolist(),
                "oracle_grad": og.tolist(),
                "filter_grad": fg.tolist(),
            }
        row["passes_c1"] = bool(
            row["posterior"]["dlnL_spread"] <= 0.01
            and row["posterior"]["grad_rel_err_max"] <= 1e-4
        )
        result["deltas"][f"{d}/{nsc}"] = row
        print(
            args.pulsar,
            f"{d}/{nsc}",
            row["n_cells"],
            "posterior spread %.2e gerr %.1e | prior spread %.2e gerr %.1e"
            % (
                row["posterior"]["dlnL_spread"],
                row["posterior"]["grad_rel_err_max"],
                row["prior"]["dlnL_spread"],
                row["prior"]["grad_rel_err_max"],
            ),
            flush=True,
        )
    out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
