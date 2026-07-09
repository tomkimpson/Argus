#!/usr/bin/env python
"""T2.6 GATE 2 — validate the blackjax nested-sampling evidence engine on a problem
with an analytically known logZ, before trusting it on the GWB likelihood.

This calls the *same* library engine core (``argus.bayesian_inference._blackjax_ns_evidence``)
that :func:`run_blackjax_nested_sampling` uses on real data, so a pass here certifies the
exact code path used for inference — not a bespoke re-implementation.

Problem: isotropic unit-Gaussian likelihood ``L(x) = N(x; 0, I_d)`` under a uniform prior
on the box ``[-B, B]^d`` (with B large enough to contain essentially all the Gaussian mass).
The evidence is then

    Z = int L(x) p(x) dx = (1 / (2B)^d) * int N(x;0,I) dx  ~=  (2B)^{-d}
    logZ_true = -d * log(2B).

Run (CPU):
    JAX_PLATFORMS=cpu PYTHONPATH=<repo>/python \
        python workflows/ng15_sgwb_demo/scripts/blackjax_ns_analytic_check.py

Exit code 0 iff every dimension's recovered logZ agrees with the closed form within
tolerance (``|logZ_est - logZ_true| < max(3*logZ_err, ATOL)``).
"""
import argparse
import math
import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from argus.bayesian_inference import _blackjax_ns_evidence  # noqa: E402


def run_dim(d, half_width=10.0, num_live=500, seed=0, dlogz=-5.0):
    """Run NS on the d-dimensional analytic Gaussian; return (logZ_true, result)."""
    lo, hi = -half_width, half_width
    log_box_volume = d * math.log(hi - lo)
    logZ_true = -log_box_volume  # Gaussian integrates to ~1 over the wide box

    def logprior_fn(x):
        inside = jnp.all((x >= lo) & (x <= hi))
        return jnp.where(inside, -log_box_volume, -jnp.inf)

    def loglikelihood_fn(x):
        return -0.5 * jnp.sum(x**2) - 0.5 * d * math.log(2.0 * math.pi)

    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    init_particles = jax.random.uniform(
        subkey, (num_live, d), minval=lo, maxval=hi
    )

    t0 = time.time()
    res = _blackjax_ns_evidence(
        logprior_fn,
        loglikelihood_fn,
        init_particles,
        num_delete=1,
        num_inner_steps=max(5, 2 * d),
        dlogz=dlogz,
        rng_key=key,
    )
    res["wall_s"] = time.time() - t0
    return logZ_true, res


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 5, 15])
    parser.add_argument("--num-live", type=int, default=500)
    parser.add_argument("--atol", type=float, default=0.5)
    args = parser.parse_args()

    print("=" * 78)
    print("T2.6 GATE 2: blackjax NS evidence vs analytic Gaussian logZ")
    print("=" * 78)
    print(f"{'d':>4} {'logZ_true':>12} {'logZ_est':>12} {'logZ_err':>10} "
          f"{'|err|':>8} {'steps':>8} {'wall_s':>8}  verdict")

    all_ok = True
    for d in args.dims:
        logZ_true, res = run_dim(d, num_live=args.num_live)
        est = res["logZ"]
        err = res["logZ_err"]
        abs_err = abs(est - logZ_true)
        ok = abs_err < max(3.0 * err, args.atol)
        all_ok &= ok
        print(f"{d:>4} {logZ_true:>12.3f} {est:>12.3f} {err:>10.3f} "
              f"{abs_err:>8.3f} {res['n_steps']:>8d} {res['wall_s']:>8.1f}  "
              f"{'PASS' if ok else 'FAIL'}")

    print("=" * 78)
    print(f"GATE 2: {'PASS' if all_ok else 'FAIL'}")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
