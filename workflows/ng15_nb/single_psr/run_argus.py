#!/usr/bin/env python
"""Single-pulsar Argus noise run on NG15 narrowband (spec task 3.1).

The likelihood is the production one for Phases 3-6, restricted to one pulsar: the
pulsar's D2 compression at the adopted layout (two scalars per DMX window, or epoch cells
for the pulsars `results/c1/epoch_rule.json` selects), fed to the sparse marginal filter
(`union.array_loglike_sparse`, diffuse timing prior, fixed white noise and ECORR).

The model is spin red noise only. The filter always carries the GW block, so it is
switched off by its amplitude: h_a = 1e-30 (GW_OFF) is ~15 dex below NG15's background
in amplitude, 30 dex in power. `--check` confirms that is negligible, and that the
likelihood equals the dense oracle on the raw TOAs, before any sampling.

NUTS samples (log10_gamma_p, log10_sigma_p) under exact uniform priors on
`common.BOX` (the boxes of reference/configs/ou.ini), with the frozen procedure's
sampler block. The draws go to outputs/single_psr/argus/<psr>.nc (gitignored) and the
summary to results/single_psr/argus/<psr>.json (`common.summarise`).

    conda activate Argus
    export PYTHONPATH=$PWD/python:$PWD
    python workflows/ng15_nb/single_psr/run_argus.py --pulsar B1855+09 [--check]
    python workflows/ng15_nb/single_psr/run_argus.py --index 0   # the SLURM array index
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "compression"))
import common  # noqa: E402
import compress  # noqa: E402
import ingest  # noqa: E402
import union  # noqa: E402

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
from numpyro.infer import MCMC, NUTS  # noqa: E402

# The ingest caches are gitignored, so a pinned worktree reads the main checkout's.
INGEST = Path(
    os.environ.get(
        "NG15_INGEST_DIR", common.REPO_ROOT / "workflows/ng15_nb/data/ingest"
    )
)
DISCOVERY_DIR = os.environ.get(
    "NG15_DISCOVERY_DIR", "/fred/oz022/tkimpson/external/discovery"
)
GW_OFF = (-30.0, -8.0)  # (log10 h_a, log10 γa): no common process


def build(name):
    cache = ingest.load(INGEST / f"{name}.npz")
    epoch = common.uses_epoch_cells(name)
    comp = compress.compress_pulsar(cache, per_epoch=epoch)
    inp = union.build_sparse_inputs([comp], np.ones((1, 1)))
    return cache, comp, inp, epoch


def make_loglike(inp):
    """lnL(log10 γp, log10 σp) on the sparse path, constant included."""

    def loglike(log10_gamma_p, log10_sigma_p):
        p = jnp.array([GW_OFF[0], GW_OFF[1], log10_sigma_p, log10_gamma_p])
        return union.array_loglike_sparse(p, inp) + inp["const"]

    return loglike


def check(name, n_points=5, seed=0, points_nc=None, origin_shift_days=1000.0):
    """Sparse path vs the dense oracle on the raw TOAs, and GW_OFF negligibility.

    The oracle is the C1 one (`oracle.oracle_loglike`, continuous-time OU covariance);
    it builds n_toa² matrices, so run this on a GPU for the large pulsars. Points are
    drawn uniformly over the prior box, or, with ``points_nc``, from a run's posterior
    (C1's own point set is posterior draws; the box corners stress the oracle too).

    The oracle's own precision (spec C1) is measured by moving its time origin by
    ``origin_shift_days``: exact in theory, since the origin only adds a constant in the
    span of the offset column, but conditioned differently. The spread of that shift is
    the oracle's noise floor at these points.
    """
    import oracle

    cache, comp, inp, epoch = build(name)
    loglike = jax.jit(make_loglike(inp))
    o_inp = oracle.oracle_inputs(cache, comp)
    o_inp_shift = dict(o_inp, t=o_inp["t"] + origin_shift_days * 86400.0)
    o_fn = jax.jit(oracle.oracle_loglike)
    rng = np.random.default_rng(seed)
    if points_nc is None:
        pts = np.column_stack(
            [rng.uniform(*common.BOX[s], n_points) for s in common.SITES]
        )
    else:
        import arviz as az

        post = az.from_netcdf(points_nc).posterior
        draws = np.column_stack([np.asarray(post[s]).ravel() for s in common.SITES])
        pts = draws[rng.choice(draws.shape[0], n_points, replace=False)]
    rows = []
    for gp, sp in pts:
        v = float(loglike(gp, sp))
        p_off = jnp.array([GW_OFF[0], GW_OFF[1], sp, gp])
        o = float(o_fn(p_off, o_inp))
        o_shift = float(o_fn(p_off, o_inp_shift))
        p_on = jnp.array([GW_OFF[0] + 5.0, GW_OFF[1], sp, gp])
        v_shift = float(union.array_loglike_sparse(p_on, inp) + inp["const"])
        rows.append(
            {
                "point": [gp, sp],
                "lnL": v,
                "oracle": o,
                "oracle_shifted_origin": o_shift,
                "gw_x1e5": v_shift,
            }
        )
    d = np.array([r["lnL"] - r["oracle"] for r in rows])
    gw = np.array([r["gw_x1e5"] - r["lnL"] for r in rows])
    floor = np.array([r["oracle_shifted_origin"] - r["oracle"] for r in rows])
    out = {
        "pulsar": name,
        "point_set": "prior box" if points_nc is None else "posterior",
        "epoch_cells": epoch,
        "n_toa": int(cache["toas"].size),
        "n_obs": inp["n_obs"],
        "points": rows,
        "dlnL_vs_oracle_spread": float(np.ptp(d)),
        "dlnL_vs_oracle_max_abs": float(np.max(np.abs(d))),
        "gw_off_max_abs_shift": float(np.max(np.abs(gw))),
        "origin_shift_days": origin_shift_days,
        "oracle_origin_shift_spread": float(np.ptp(floor)),
        "git_sha": common.git_sha(),
    }
    print(json.dumps(out, indent=2), flush=True)
    return out


def run(name, num_warmup, num_samples):
    _, comp, inp, epoch = build(name)
    loglike = make_loglike(inp)

    def model():
        x = {s: numpyro.sample(s, dist.Uniform(*common.BOX[s])) for s in common.SITES}
        numpyro.factor("loglik", loglike(x["log10_gamma_p"], x["log10_sigma_p"]))

    nuts = common.NUTS
    mcmc = MCMC(
        NUTS(
            model,
            target_accept_prob=nuts["target_accept_prob"],
            max_tree_depth=nuts["max_tree_depth"],
        ),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=nuts["num_chains"],
        chain_method="vectorized",
        progress_bar=False,
    )
    t0 = time.perf_counter()
    mcmc.run(
        jax.random.PRNGKey(common.seed_of(name)),
        extra_fields=("diverging", "num_steps"),
    )
    wall = time.perf_counter() - t0
    samples = {
        k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()
    }
    extra = mcmc.get_extra_fields(group_by_chain=True)

    import arviz as az

    out_dir = common.OUTPUTS / "argus"
    out_dir.mkdir(parents=True, exist_ok=True)
    az.from_numpyro(mcmc).to_netcdf(out_dir / f"{name}.nc")

    rec = common.summarise(
        name,
        "argus",
        samples,
        np.asarray(extra["diverging"]),
        common.array_span_s(DISCOVERY_DIR),
        extra={
            "layout": "epoch cells" if epoch else "two scalars per DMX window",
            "n_obs": inp["n_obs"],
            "m_cols": int(comp["m"].shape[1]),
            "num_warmup": num_warmup,
            "num_samples": num_samples,
            "mean_num_steps": float(np.mean(extra["num_steps"])),
            "sample_wall_s": wall,
            "device": str(jax.devices()[0]),
        },
    )
    path = common.write(rec)
    print(json.dumps(rec, indent=2), flush=True)
    print(f"wrote {path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--pulsar")
    who.add_argument("--index", type=int, help="manifest index (SLURM array)")
    ap.add_argument("--check", action="store_true", help="oracle check, no sampling")
    ap.add_argument("--points-nc", help="--check at draws from this run's posterior")
    ap.add_argument("--n-points", type=int, default=5)
    ap.add_argument("--num-warmup", type=int, default=common.NUTS["num_warmup"])
    ap.add_argument("--num-samples", type=int, default=common.NUTS["num_samples"])
    args = ap.parse_args()
    name = args.pulsar or common.pulsars()[args.index]
    print(f"{name}: device {jax.devices()[0]}, commit {common.git_sha()}", flush=True)
    if args.check:
        out = check(name, n_points=args.n_points, points_nc=args.points_nc)
        tag = "" if args.points_nc is None else "_posterior"
        dest = common.RESULTS / f"argus_check{tag}" / f"{name}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=2) + "\n")
        return
    run(name, args.num_warmup, args.num_samples)


if __name__ == "__main__":
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    main()
