#!/usr/bin/env python
"""NS cost-scaling study — Stage 1b (revised): Kalman likelihood cost vs pulsar count N.

Replaces running full nested sampling on MDC2 subsets (which is contaminated by the
near-singular-covariance pathology — see notes/ns_numerical_hygiene.md) with a *direct
microbenchmark* of the likelihood NS actually evaluates. We time
``numpyro.infer.util.log_density(model)`` — i.e. one full Kalman-filter evaluation — at SANE
parameters (all reparameterized latents = 0, the prior centre), so we never enter the
pathological tails. This isolates the pure per-evaluation Kalman cost c(N), which is the
quantity the combined cost model needs, cleanly and in seconds.

We evaluate both a single draw and a batch of ``--batch`` draws (via ``jax.vmap``, mimicking
the NS inner kernel's vmap over ``num_delete`` particles), reporting per-single and per-batch
wall times. Fits c(N) ~ a * N^b.

Run on GPU (the real venue) via SLURM, or on CPU for a quick check of small N:
    JAX_PLATFORMS=cpu PYTHONPATH=<repo>/python python ns_likelihood_microbench.py --N 2 4 8
"""

import argparse
import csv
import logging
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import configparser  # noqa: E402
from numpyro.handlers import trace, seed  # noqa: E402
from numpyro.infer.util import log_density  # noqa: E402

from argus import workflow, utils, prior_models, io_manager  # noqa: E402
from argus.bayesian_inference import numpyro_model  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WORKFLOW = os.path.dirname(HERE)
DERIVED = os.path.join(WORKFLOW, "outputs", "derived_configs")
DEFAULT_OUT = os.path.join(WORKFLOW, "outputs", "scaling", "likelihood_cost_vs_N.csv")


def _load_cfg(path):
    cp = configparser.ConfigParser(interpolation=None)
    cp.optionxform = str
    cp.read(path)
    return cp


def bench_one(config_path, batch=25, repeats=20, warmup=3):
    """Time one full-model log-density (Kalman) eval for the pulsar set in `config_path`."""
    cp = _load_cfg(config_path)
    # JaxKalmanFilter uses argus's global logger, which must be initialised first.
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

    # Latent sites (all reparameterized Normal(0,1)); evaluate at z=0 = prior centre (sane).
    key = jax.random.PRNGKey(0)
    tr = trace(seed(model, key)).get_trace()
    latents = {
        name: jnp.zeros(jnp.shape(site["value"]))
        for name, site in tr.items()
        if site["type"] == "sample" and not site.get("is_observed", False)
    }
    ndim = int(
        sum(int(np.prod(jnp.shape(v))) if jnp.shape(v) else 1 for v in latents.values())
    )

    def loglik(params):
        lp, _ = log_density(model, (), {}, params)
        return lp

    f1 = jax.jit(loglik)
    fB = jax.jit(jax.vmap(loglik))
    batch_params = {
        k: jnp.broadcast_to(v, (batch,) + jnp.shape(v)) for k, v in latents.items()
    }

    # single-eval timing
    v = f1(latents)
    v.block_until_ready()
    for _ in range(warmup):
        f1(latents).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(repeats):
        f1(latents).block_until_ready()
    t_single = (time.perf_counter() - t0) / repeats

    # batched-eval timing (vmap over `batch` particles, like NS's num_delete)
    vb = fB(batch_params)
    vb.block_until_ready()
    for _ in range(warmup):
        fB(batch_params).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fB(batch_params).block_until_ready()
    t_batch = (time.perf_counter() - t0) / repeats

    return {
        "N": n_pulsars,
        "ndim": ndim,
        "loglik_at_center": float(v),
        "t_single_s": t_single,
        "t_batch_s": t_batch,
        "batch": batch,
        "t_per_particle_s": t_batch / batch,
    }


def _fit_powerlaw(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = (x > 0) & (y > 0)
    if ok.sum() < 2:
        return float("nan"), float("nan")
    b, loga = np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)
    return float(np.exp(loga)), float(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("=" * 84)
    print(
        f"Kalman likelihood microbenchmark (D=2 fixed-noise configs), batch={args.batch}"
    )
    print("=" * 84)
    print(
        f"{'N':>4} {'ndim':>5} {'loglik@0':>14} {'t_single(ms)':>13} "
        f"{'t_batch(ms)':>12} {'t/particle(ms)':>15}"
    )

    rows = []
    for N in args.N:
        cfg = os.path.join(DERIVED, f"ns_scal_fixed_N{N:02d}_D002_nl500_nd25_s42.ini")
        if not os.path.exists(cfg):
            print(
                f"  (missing config for N={N}: {cfg}) -- run gen_scaling_configs.py --stage 1b"
            )
            continue
        r = bench_one(cfg, batch=args.batch, repeats=args.repeats)
        rows.append(r)
        print(
            f"{r['N']:>4} {r['ndim']:>5} {r['loglik_at_center']:>14.2f} "
            f"{r['t_single_s']*1e3:>13.2f} {r['t_batch_s']*1e3:>12.2f} "
            f"{r['t_per_particle_s']*1e3:>15.3f}"
        )

    if rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        a1, b1 = _fit_powerlaw([r["N"] for r in rows], [r["t_single_s"] for r in rows])
        aB, bB = _fit_powerlaw([r["N"] for r in rows], [r["t_batch_s"] for r in rows])
        print("=" * 84)
        print(
            f"fit: t_single(N) ~ {a1:.3g}*N^{b1:.2f} s   "
            f"t_batch(N) ~ {aB:.3g}*N^{bB:.2f} s"
        )
        print(f"CSV -> {args.out}")


if __name__ == "__main__":
    main()
