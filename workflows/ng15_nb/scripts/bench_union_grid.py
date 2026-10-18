#!/usr/bin/env python
"""Task 2.6: the 67-psr union-grid likelihood-plus-gradient cost on one A100 (K3).

    python workflows/ng15_nb/scripts/bench_union_grid.py \
        [--delta-days none|7|14|30] [--ingest-dir workflows/ng15_nb/data/ingest] \
        [--repeats 5] [--out workflows/ng15_nb/results/k3/<delta>.json]

Every ingested pulsar is compressed at grid width Δ (or per DMX window) with two scalars
per cell (D2 as amended 2026-10-18) and placed on the masked union grid
(`union.build_union_inputs`, two steps per union time). Grid cells are stamped at their
centres, so pulsars share steps and the step count falls with Δ. The diffuse marginal filter's value and its gradient with
respect to all 2 + 2N sampled parameters (log10 ha, log10 γa, log10 σp, log10 γp) are then
timed after compilation. The point is the Phase 1 Discovery-OU posterior mean where it is
available, which only matters for the finiteness check; the cost does not depend on it.

K3 (spec §8): ≤ 0.5 s per evaluation is the target, > 1 s stops for a re-budget.

Reverse mode through the library's single `lax.scan` saves every step's residuals,
~15 MB per step at 67 psr (dominated by the M_sum² accumulator), so 220 GB here. The
gradient is therefore timed on a benchmark-only copy of the filter whose scan is split
into segments, each under `jax.checkpoint` (√T rematerialisation). Its arithmetic is
the library's, so its value must equal the stock value, which is checked. The library is
not edited; adopting the split there is a separate change.
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
import types  # noqa: E402

import compress  # noqa: E402
import ingest  # noqa: E402
import union  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)


def posterior_mean_point(names):
    """(log10 ha, log10 γa, log10 σp[N], log10 γp[N]) at the OU14 ε = 0 posterior mean."""
    import arviz as az
    from c1_compression import POSTERIOR

    manifest = json.loads((REPO / "data/ng15_nb/MANIFEST.json").read_text())
    order = [e["pulsar"] for e in manifest["feathers"]]
    idx = [order.index(n) for n in names]
    post = az.from_netcdf(POSTERIOR).posterior

    def mean(v):
        a = np.asarray(post[v])
        return a.reshape(-1, *a.shape[2:]).mean(0)

    return np.concatenate(
        [
            [mean("gw_log10_ha"), mean("gw_log10_gamma_a")],
            mean("log10_sigma_p")[idx],
            mean("log10_gamma_p")[idx],
        ]
    )


SCAN = "    (x, P, Xi, A, b, c, L), _ = lax.scan(step, (x, P, Xi, A, b, c, L), inputs)"
SEGMENTED = """    _n = inputs[0].shape[0]
    _seg = SEGMENT
    _inputs = jax.tree_util.tree_map(
        lambda a: a.reshape((_n // _seg, _seg) + a.shape[1:]), inputs
    )

    @jax.checkpoint
    def _segment(carry, seg_inputs):
        carry, _ = lax.scan(step, carry, seg_inputs)
        return carry, None

    (x, P, Xi, A, b, c, L), _ = lax.scan(_segment, (x, P, Xi, A, b, c, L), _inputs)"""


def segmented_filter_module(segment):
    """The library filter with its scan split into checkpointed segments (bench-only)."""
    src = (REPO / "python" / "argus" / "jax_kalman_filter.py").read_text()
    assert src.count(SCAN) == 1, "scan site moved"
    src = src.replace(SCAN, SEGMENTED.replace("SEGMENT", str(segment)))
    mod = types.ModuleType("argus._jk_segmented")
    mod.__package__ = "argus"
    exec(compile(src, "jk_segmented", "exec"), mod.__dict__)
    return mod


def segment_length(n):
    """The divisor of n closest to √n (the scan runs n = T − 1 steps)."""
    ds = [d for d in range(1, n + 1) if n % d == 0]
    return min(ds, key=lambda d: abs(np.log(d) - 0.5 * np.log(n)))


def timed(f, *args, repeats):
    out = jax.block_until_ready(f(*args))
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        out = jax.block_until_ready(f(*args))
        ts.append(time.perf_counter() - t)
    return out, ts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ingest-dir", default=str(REPO / "workflows/ng15_nb/data/ingest"))
    ap.add_argument("--delta-days", default="none")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out")
    args = ap.parse_args()
    delta = None if args.delta_days == "none" else float(args.delta_days)
    tag = "dmx_window" if delta is None else f"{delta:g}d"
    out = Path(args.out or REPO / f"workflows/ng15_nb/results/k3/{tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    t = time.perf_counter()
    caches = [ingest.load(p) for p in sorted(Path(args.ingest_dir).glob("*.npz"))]
    comps = [compress.compress_pulsar(c, delta_days=delta) for c in caches]
    compress_s = time.perf_counter() - t
    names = [c["name"] for c in caches]
    inp = union.build_union_inputs(comps, union.hd_matrix(caches))
    N, T = inp["Npsr"], int(inp["data"].shape[0])
    print(
        f"{N} psr, {inp['n_union_times']} union times, {T} steps, M_sum {inp['M_sum']}, "
        f"compression {compress_s:.0f} s",
        flush=True,
    )

    p = jnp.asarray(posterior_mean_point(names))

    # Value: the stock library filter.
    value = jax.jit(union.array_loglike)
    t = time.perf_counter()
    jax.block_until_ready(value(p, inp))
    compile_v = time.perf_counter() - t
    ll_stock, ts_v = timed(value, p, inp, repeats=args.repeats)
    print(f"value {np.median(ts_v):.3f} s", flush=True)

    # Value and gradient: the segmented copy.
    seg = segment_length(T - 1)
    union.jk = segmented_filter_module(seg)
    vg = jax.jit(jax.value_and_grad(union.array_loglike))
    t = time.perf_counter()
    jax.block_until_ready(vg(p, inp))
    compile_vg = time.perf_counter() - t
    (ll, g), ts_vg = timed(vg, p, inp, repeats=args.repeats)
    print(f"value+grad {np.median(ts_vg):.3f} s (segment {seg})", flush=True)

    dev = jax.devices()[0]
    stats = dev.memory_stats() or {}
    occupancy = float(np.asarray(inp["mask_matrices"]).mean())
    result = {
        "device": str(dev),
        "delta": tag,
        "n_psr": N,
        "n_union_times": inp["n_union_times"],
        "n_steps": T,
        "M_sum": inp["M_sum"],
        "n_params": int(p.size),
        "mask_occupancy": occupancy,
        "H_bytes": int(inp["H_matrices"].nbytes),
        "compression_s": compress_s,
        "segment": seg,
        "segmented_minus_stock_lnL": float(ll) - float(ll_stock),
        "compile_value_and_grad_s": compile_vg,
        "value_and_grad_s": ts_vg,
        "value_and_grad_s_median": float(np.median(ts_vg)),
        "compile_value_s": compile_v,
        "value_s": ts_v,
        "value_s_median": float(np.median(ts_v)),
        "peak_bytes_in_use": int(stats.get("peak_bytes_in_use", -1)),
        "lnL": float(ll) + inp["const"],
        "grad_finite": bool(np.all(np.isfinite(np.asarray(g)))),
        "grad_absmax": float(np.max(np.abs(np.asarray(g)))),
        "point": np.asarray(p).tolist(),
        "pulsars": names,
        "k3": {"target_s": 0.5, "stop_s": 1.0},
    }
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("point", "pulsars")}, indent=1
        ),
        flush=True,
    )
    out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
