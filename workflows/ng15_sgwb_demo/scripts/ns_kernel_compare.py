#!/usr/bin/env python
"""NS cost-scaling study — inner-kernel comparison: monolithic slice vs slice-within-Gibbs.

Tests the hypothesis (Yallup 2026, arXiv:2602.17414) that wiring hierarchical *structure*
into the NS inner kernel — an axis-aligned slice-within-Gibbs (SwiG) coordinate sweep
(``blackjax.nsswig``) — mixes better per likelihood evaluation than the default monolithic
hit-and-run slice (``blackjax.nss``), and so needs fewer inner steps / live points for the
same evidence accuracy at high dimension. If so, it directly lowers the full-PTA runtime.

We compare on the likelihood-free analytic Gaussian (known logZ = -d*log(2B)) so the only
difference is the kernel. For each dimension we run both kernels at a *stress* inner-step
budget (default 2*d, where the monolithic slice is known to bias logZ upward) and report:
  |logZ - logZ_true| (accuracy), n_steps, wall_s.
A kernel that stays accurate at 2*d while the other needs ~6*d is the more efficient choice.

Run (CPU, fast):
    JAX_PLATFORMS=cpu PYTHONPATH=<repo>/python \
        python workflows/ng15_sgwb_demo/scripts/ns_kernel_compare.py --dims 15 30 60
"""
import argparse
import csv
import os

import jax

jax.config.update("jax_enable_x64", True)

from ns_scaling_dimension import run_dim, DEFAULT_OUT  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dims", type=int, nargs="+", default=[15, 30, 60])
    ap.add_argument("--num-live", type=int, default=500)
    ap.add_argument("--num-delete", type=int, default=25)
    ap.add_argument("--inner-mult", type=float, default=2.0,
                    help="Inner steps = inner_mult*d (default 2 = the stress setting where "
                         "monolithic slice biases logZ; the discriminating test).")
    ap.add_argument("--atol", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.join(DEFAULT_OUT, "kernel_compare.csv"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("=" * 90)
    print("NS inner-kernel comparison: nss (monolithic slice) vs nsswig (slice-within-Gibbs)")
    print(f"num_live={args.num_live}  num_delete={args.num_delete}  "
          f"inner_steps={args.inner_mult}*d")
    print("=" * 90)
    print(f"{'d':>5} {'kernel':>8} {'logZ_true':>11} {'logZ_est':>11} {'|err|':>8} "
          f"{'inner':>6} {'steps':>7} {'wall_s':>8}  verdict")

    rows = []
    for d in args.dims:
        for kernel in ("nss", "nsswig"):
            logZ_true, res = run_dim(
                d, num_live=args.num_live, num_delete=args.num_delete,
                inner_mult=args.inner_mult, kernel=kernel,
            )
            est = res["logZ"]
            err = res["logZ_err"]
            abs_err = abs(est - logZ_true)
            ok = abs_err < max(3.0 * err, args.atol)
            rows.append({
                "d": d, "kernel": kernel, "logZ_true": logZ_true, "logZ_est": est,
                "logZ_err": err, "abs_err": abs_err,
                "num_inner_steps": res["num_inner_steps"], "n_steps": int(res["n_steps"]),
                "wall_s": res["wall_s"], "ok": int(ok),
            })
            print(f"{d:>5} {kernel:>8} {logZ_true:>11.3f} {est:>11.3f} {abs_err:>8.3f} "
                  f"{res['num_inner_steps']:>6d} {res['n_steps']:>7d} {res['wall_s']:>8.1f}"
                  f"  {'PASS' if ok else 'FAIL'}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("=" * 90)
    print(f"CSV -> {args.out}")


if __name__ == "__main__":
    main()
