#!/usr/bin/env python
"""K3 re-budget: the 67-psr likelihood-plus-gradient cost on the sparse marginal filter.

    python workflows/ng15_nb/scripts/bench_sparse.py [--delta-days none|7|14|30] \
        [--segments 64 128 256] [--batches 1 4 16 32] [--repeats 5] [--epoch-rule] [--out workflows/ng15_nb/results/k3/sparse_<delta>.json]

Same compression, point and parameters as `bench_union_grid.py`, but every scalar
observation is one step of `jk._run_kalman_filter_marginal_sparse` (approved
2026-10-18), so the step count is the number of observations at any Δ. The value is
compared with the dense union-grid result in results/k3/<delta>.json when present.

`--epoch-rule` compresses the pulsars that D2's rule selects
(results/c1/epoch_rule.json) into epoch cells instead, the layout Phases 3-6 use.

The sparse scan is latency-bound (~2 MFLOP per step), so `--batches` also times
value-and-gradient vmapped over B parameter points (B NUTS chains on one GPU); the
throughput figure is seconds per evaluation = wall time / B.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "workflows" / "ng15_nb" / "compression"))
sys.path.insert(0, str(REPO / "workflows" / "ng15_nb" / "scripts"))
import compress  # noqa: E402
import ingest  # noqa: E402
import union  # noqa: E402
from bench_union_grid import posterior_mean_point, timed  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ingest-dir", default=str(REPO / "workflows/ng15_nb/data/ingest"))
    ap.add_argument("--delta-days", default="none")
    ap.add_argument("--segments", nargs="*", type=int, default=[64, 128, 256])
    ap.add_argument("--batches", nargs="*", type=int, default=[1, 4, 16, 32])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--epoch-rule", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    delta = None if args.delta_days == "none" else float(args.delta_days)
    tag = "dmx_window" if delta is None else f"{delta:g}d"
    epoch = set()
    if args.epoch_rule:
        rule = REPO / "workflows/ng15_nb/results/c1/epoch_rule.json"
        epoch = set(json.loads(rule.read_text())["needs_epoch_cells"])
        tag += "_epoch_rule"
    out = Path(args.out or REPO / f"workflows/ng15_nb/results/k3/sparse_{tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    caches = [ingest.load(p) for p in sorted(Path(args.ingest_dir).glob("*.npz"))]
    comps = [
        compress.compress_pulsar(c, delta_days=delta, per_epoch=c["name"] in epoch)
        for c in caches
    ]
    names = [c["name"] for c in caches]
    inp = union.build_sparse_inputs(comps, union.hd_matrix(caches))
    M_sum = int(inp["obs"]["m"].shape[1])
    print(f"{len(comps)} psr, {inp['n_obs']} observations, M_sum {M_sum}", flush=True)
    p = jnp.asarray(posterior_mean_point(names))

    rows = []
    for seg in args.segments:
        value = jax.jit(lambda q, i, s=seg: union.array_loglike_sparse(q, i, s))
        vg = jax.jit(
            jax.value_and_grad(lambda q, i, s=seg: union.array_loglike_sparse(q, i, s))
        )
        t = time.perf_counter()
        jax.block_until_ready(vg(p, inp))
        compile_vg = time.perf_counter() - t
        (ll, g), ts_vg = timed(vg, p, inp, repeats=args.repeats)
        _, ts_v = timed(value, p, inp, repeats=args.repeats)
        stats = jax.devices()[0].memory_stats() or {}
        row = {
            "segment": seg,
            "value_s_median": float(np.median(ts_v)),
            "value_and_grad_s_median": float(np.median(ts_vg)),
            "value_s": ts_v,
            "value_and_grad_s": ts_vg,
            "compile_value_and_grad_s": compile_vg,
            "peak_bytes_in_use": int(stats.get("peak_bytes_in_use", -1)),
            "lnL": float(ll) + inp["const"],
            "grad_finite": bool(np.all(np.isfinite(np.asarray(g)))),
            "grad": np.asarray(g).tolist(),
        }
        rows.append(row)
        print(
            f"segment {seg}: value {row['value_s_median']:.3f} s, value+grad "
            f"{row['value_and_grad_s_median']:.3f} s, lnL {row['lnL']:.4f}",
            flush=True,
        )

    seg = 128
    batch_rows = []
    rng = np.random.default_rng(0)
    for B in args.batches:
        pts = jnp.asarray(
            np.asarray(p)[None, :] + 0.05 * rng.standard_normal((B, p.size))
        )
        vg_b = jax.jit(
            jax.vmap(
                jax.value_and_grad(lambda q, i: union.array_loglike_sparse(q, i, seg)),
                in_axes=(0, None),
            )
        )
        try:
            jax.block_until_ready(vg_b(pts, inp))
            (llb, gb), ts = timed(vg_b, pts, inp, repeats=args.repeats)
        except Exception as e:  # out of memory at large B
            print(f"batch {B}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            break
        stats = jax.devices()[0].memory_stats() or {}
        row = {
            "batch": B,
            "wall_s_median": float(np.median(ts)),
            "s_per_eval": float(np.median(ts)) / B,
            "peak_bytes_in_use": int(stats.get("peak_bytes_in_use", -1)),
            "all_finite": bool(
                np.all(np.isfinite(np.asarray(gb)))
                and np.all(np.isfinite(np.asarray(llb)))
            ),
        }
        batch_rows.append(row)
        print(
            f"batch {B}: wall {row['wall_s_median']:.3f} s, {row['s_per_eval']:.3f} s/eval",
            flush=True,
        )

    dense_path = REPO / f"workflows/ng15_nb/results/k3/{tag}.json"
    dense_lnL = (
        json.loads(dense_path.read_text())["lnL"] if dense_path.exists() else None
    )
    result = {
        "device": str(jax.devices()[0]),
        "delta": tag,
        "n_psr": len(comps),
        "n_obs": inp["n_obs"],
        "M_sum": M_sum,
        "epoch_cell_pulsars": sorted(epoch),
        "runs": rows,
        "batched_value_and_grad": batch_rows,
        "dense_lnL": dense_lnL,
        "sparse_minus_dense_lnL": (
            None if dense_lnL is None else rows[0]["lnL"] - dense_lnL
        ),
        "point": np.asarray(p).tolist(),
        "pulsars": names,
        "k3": {"target_s": 0.5, "stop_s": 1.0},
    }
    print("sparse - dense lnL:", result["sparse_minus_dense_lnL"], flush=True)
    out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
