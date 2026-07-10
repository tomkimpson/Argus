#!/usr/bin/env python
"""NS cost-scaling study — Stage 1a: pure-dimension scaling (likelihood-free).

Measures how blackjax nested slice sampling scales with problem dimension ``d`` on an
analytically-known target, decoupled entirely from the Kalman likelihood cost. This is the
study's *early kill probe*: if the sampler's step-count / wall-clock blows up in pure
dimension (or its logZ drifts from the closed form) we learn the full-PTA verdict here, on
CPU, before spending any A100 time on Kalman runs.

Target (identical to the T2.6 GATE-2 check, ``blackjax_ns_analytic_check.py``): an isotropic
unit-Gaussian likelihood ``L(x)=N(x;0,I_d)`` under a uniform prior on the box ``[-B,B]^d``,
so ``logZ_true = -d*log(2B)`` is known exactly. This mirrors ``run_dim`` there but exposes the
``num_delete`` / ``num_live`` / ``num_inner_steps`` knobs (the analytic check hard-codes
num_delete=1), so we can characterise the *same* engine core (``_blackjax_ns_evidence``) under
the settings we would actually run at scale.

We measure, per dimension:
  - logZ accuracy: |logZ_est - logZ_true| vs the stochastic logZ_err  (evidence *correctness*)
  - n_steps       : number of sequential NS iterations to reach dlogz  (the D-scaling cost)
  - wall_s        : sampling wall-clock

and fit power laws n_steps(d), wall_s(d). Writes a CSV for ns_scaling_analyze.py.

Run (CPU, fast):
    JAX_PLATFORMS=cpu PYTHONPATH=<repo>/python \
        python workflows/ng15_sgwb_demo/scripts/ns_scaling_dimension.py \
        --dims 2 5 15 30 60 --num-delete 25

Exit code 0 iff every dimension's recovered logZ agrees with the closed form within
tolerance (|logZ_est - logZ_true| < max(3*logZ_err, ATOL)); otherwise 1 (evidence became
unreliable at some dimension — itself a study finding).
"""
import argparse
import csv
import math
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from argus.bayesian_inference import _blackjax_ns_evidence  # noqa: E402

DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs",
    "scaling",
)


def run_dim(d, half_width=10.0, num_live=500, num_delete=25, num_inner_steps=None,
            inner_mult=None, seed=0, dlogz=-5.0, max_steps=200000, kernel="nss"):
    """Run NS on the d-dim analytic Gaussian; return (logZ_true, res).

    Mirrors ``blackjax_ns_analytic_check.run_dim`` (same target, same known logZ) but with
    the batch-deletion / live-point knobs exposed for the scaling sweep. ``inner_mult`` sets
    num_inner_steps = inner_mult*d (overrides ``num_inner_steps``); the Stage-1a diagnostic
    found the engine default of 2*d biases logZ at high d, while ~6*d recovers accuracy.
    """
    lo, hi = -half_width, half_width
    log_box_volume = d * math.log(hi - lo)
    logZ_true = -log_box_volume  # Gaussian integrates to ~1 over the wide box

    def logprior_fn(x):
        inside = jnp.all((x >= lo) & (x <= hi))
        return jnp.where(inside, -log_box_volume, -jnp.inf)

    def loglikelihood_fn(x):
        return -0.5 * jnp.sum(x**2) - 0.5 * d * math.log(2.0 * math.pi)

    if inner_mult is not None:
        num_inner_steps = max(5, int(inner_mult * d))
    elif num_inner_steps is None:
        num_inner_steps = max(5, 2 * d)  # the engine's default; scales with d

    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    init_particles = jax.random.uniform(subkey, (num_live, d), minval=lo, maxval=hi)

    t0 = time.perf_counter()
    res = _blackjax_ns_evidence(
        logprior_fn,
        loglikelihood_fn,
        init_particles,
        num_delete=num_delete,
        num_inner_steps=num_inner_steps,
        dlogz=dlogz,
        max_steps=max_steps,
        rng_key=key,
        kernel=kernel,
    )
    res["wall_s"] = time.perf_counter() - t0
    res["num_inner_steps"] = num_inner_steps
    return logZ_true, res


def _fit_powerlaw(x, y):
    """Fit y ~ a * x^b in log-log; return (a, b) or (nan, nan) if degenerate."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = (x > 0) & (y > 0)
    if ok.sum() < 2:
        return float("nan"), float("nan")
    b, loga = np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)
    return float(np.exp(loga)), float(b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=int, nargs="+",
                        default=[2, 5, 15, 30, 60, 120, 270])
    parser.add_argument("--num-live", type=int, default=500)
    parser.add_argument("--num-delete", type=int, default=25)
    parser.add_argument("--num-inner-steps", type=int, default=None,
                        help="Fixed inner-step count; default engine's max(5, 2*d).")
    parser.add_argument("--inner-mult", type=float, default=None,
                        help="Set num_inner_steps = inner_mult*d (overrides --num-inner-steps). "
                             "Stage-1a diagnostic: ~6 recovers logZ accuracy at high d.")
    parser.add_argument("--dlogz", type=float, default=-5.0)
    parser.add_argument("--atol", type=float, default=0.5)
    parser.add_argument("--out", default=os.path.join(DEFAULT_OUT, "dimension_scaling.csv"))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("=" * 92)
    print("NS scaling Stage 1a: pure-dimension scaling (analytic Gaussian, likelihood-free)")
    print(f"num_live={args.num_live}  num_delete={args.num_delete}  "
          f"num_inner_steps={args.num_inner_steps or 'max(5,2d)'}  dlogz={args.dlogz}")
    print("=" * 92)
    header = (f"{'d':>5} {'logZ_true':>12} {'logZ_est':>12} {'logZ_err':>10} "
              f"{'|err|':>8} {'inner':>6} {'steps':>8} {'wall_s':>9}  verdict")
    print(header)

    rows = []
    all_ok = True
    for d in args.dims:
        logZ_true, res = run_dim(
            d,
            num_live=args.num_live,
            num_delete=args.num_delete,
            num_inner_steps=args.num_inner_steps,
            inner_mult=args.inner_mult,
            dlogz=args.dlogz,
        )
        est = res["logZ"]
        err = res["logZ_err"]
        abs_err = abs(est - logZ_true)
        ok = abs_err < max(3.0 * err, args.atol)
        all_ok &= ok
        rows.append({
            "d": d,
            "logZ_true": logZ_true,
            "logZ_est": est,
            "logZ_err": err,
            "abs_err": abs_err,
            "num_inner_steps": res["num_inner_steps"],
            "n_steps": int(res["n_steps"]),
            "wall_s": res["wall_s"],
            "num_live": args.num_live,
            "num_delete": args.num_delete,
            "ok": int(ok),
        })
        print(f"{d:>5} {logZ_true:>12.3f} {est:>12.3f} {err:>10.3f} "
              f"{abs_err:>8.3f} {res['num_inner_steps']:>6d} {res['n_steps']:>8d} "
              f"{res['wall_s']:>9.1f}  {'PASS' if ok else 'FAIL'}")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    dims = [r["d"] for r in rows]
    a_s, b_s = _fit_powerlaw(dims, [r["n_steps"] for r in rows])
    a_w, b_w = _fit_powerlaw(dims, [r["wall_s"] for r in rows])
    print("=" * 92)
    print(f"Power-law fits:  n_steps ~ {a_s:.3g} * d^{b_s:.2f}   "
          f"wall_s ~ {a_w:.3g} * d^{b_w:.2f}")
    print(f"CSV written: {args.out}")
    print(f"Stage 1a evidence-accuracy gate: {'PASS' if all_ok else 'FAIL'} "
          f"(exponent b_steps={b_s:.2f}, b_wall={b_w:.2f})")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
