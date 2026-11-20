#!/usr/bin/env python
"""GPU job for Phase 1 of the detection-statistic spec (tasks 1.1, 1.2, 1.4).

    python workflows/ng15_nb/stat/opening_job.py check [--batches 1 8 32]
    python workflows/ng15_nb/stat/opening_job.py full --batch B

check  Task 1.1 on the 67-pulsar filter: wall time and peak memory of one G (batch 1)
       against one parameter gradient, then the batched G at each `--batches` size
       (out-of-memory is caught and recorded). V2 at 3 draws: lnL at C = I against
       `SparseArrayFilter.get_likelihood` at ε = 0 (gate ≤ 1e-8 nats), and against the
       probe's stored lnL (recorded, not gated: the only check of the draw → θ mapping).
       Exits non-zero if a single G takes more than 30 s (spec §10).
full   Task 1.4: G for all 400 draws in fixed batches of B; V1 on 12 of them (3 per
       chain) for Γ_HD and scramble 0, against the forward-mode ε-derivative through
       the harness (gate ≤ 1e-6 relative) and a central difference (reported); then a
       rerun of 5 draws (gate ≤ 1e-8). V1 reads its G rows from the 400-draw array, so
       the per-draw S of task 1.5 reproduces V1's contracted values exactly.

Tracked outputs go to `results/stat/opening/`; the full G array (~14 MB) goes to the
gitignored `workflows/ng15_nb/outputs/stat/opening/` beside the ingest caches, with its
sha-256 recorded in the tracked JSON.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import orf_gradient as og  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

RESULTS = og.WF / "results/stat/opening"
OUTPUTS = og.DEFAULT_INGEST.parent.parent / "outputs/stat/opening"
SCRAMBLES = json.loads((RESULTS / "scrambles_summary.json").read_text())
V2_DRAWS = [(0, 0), (1, 50), (3, 99)]
V1_DRAWS = [(c, d) for c in range(4) for d in (0, 50, 99)]
RERUN_DRAWS = [(0, 1), (1, 2), (2, 3), (3, 4), (2, 77)]
V1_RTOL, V2_ATOL, RERUN_ATOL, MAX_G_S = 1e-6, 1e-8, 1e-8, 30.0
FD_H = 1e-4


def timed(fn, *args, repeat=2):
    t = time.perf_counter()
    jax.block_until_ready(fn(*args))
    first = time.perf_counter() - t
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        out = jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t)
    return out, first, times


def setup():
    kf = og.build_filter()
    with np.load(SCRAMBLES["path"]) as a:
        if og.sha256(SCRAMBLES["path"]) != SCRAMBLES["sha256"]:
            raise SystemExit("scramble archive hash differs from scrambles_summary.json")
        if not np.array_equal(a["true_orf"], np.asarray(kf.inp["orf"])):
            raise SystemExit("scramble archive true_orf differs from the filter's ORF")
        scr0 = np.asarray(a["orfs"][0])
    p, extras = og.load_draws()
    assert p.shape == (4, 100, 2 + 2 * kf.Npsr), p.shape
    meta = {
        "device": str(jax.devices()[0]),
        "jax": jax.__version__,
        "git_sha": __import__("common").git_sha(),
        "draws": {k: extras[k] for k in ("path", "sha256")},
        "filter": kf.summary(),
    }
    return kf, og.make_fns(kf), p, extras, scr0, meta


def check(args):
    kf, f, p, extras, _, meta = setup()
    p0 = jnp.asarray(p[0, 0])
    _, g_first, g_times = timed(f["grad_orf"], p0[None])
    mem_g = og.peak_bytes()
    pgrad = jax.jit(jax.grad(lambda q: f["lnl_at"](jnp.eye(kf.Npsr), q)))
    _, pg_first, pg_times = timed(pgrad, p0)
    timing = {
        "G_batch1": {"compile_plus_first_s": g_first, "steady_s": g_times},
        "param_grad_batch1": {"compile_plus_first_s": pg_first, "steady_s": pg_times},
        "peak_bytes_after_G_batch1": mem_g,
        "peak_bytes_after_param_grad": og.peak_bytes(),
    }
    print(json.dumps(timing, indent=2), flush=True)
    batched = {}
    for b in args.batches:
        if b == 1:
            continue
        P = jnp.asarray(p.reshape(-1, p.shape[-1])[:b])
        try:
            _, first, times = timed(f["grad_orf"], P, repeat=1)
            batched[b] = {
                "compile_plus_first_s": first,
                "steady_s": times,
                "steady_s_per_draw": times[0] / b,
                "peak_bytes": og.peak_bytes(),
            }
        except Exception as e:  # noqa: BLE001 (record OOM, keep going)
            batched[b] = {"error": repr(e)[:400]}
        print(b, json.dumps(batched[b]), flush=True)
    timing["G_batched"] = batched

    eye = jnp.eye(kf.Npsr)
    v2 = []
    for c, d in V2_DRAWS:
        q = jnp.asarray(p[c, d])
        mine = float(f["lnl_at"](eye, q))
        harness = float(f["harness_lnl"](q))
        stored = float(extras["stored_lnL"][c, d])
        v2.append(
            {
                "chain": c,
                "draw": d,
                "lnL_stat": mine,
                "lnL_harness": harness,
                "abs_diff": abs(mine - harness),
                "lnL_stored": stored,
                "diff_vs_stored": mine - stored,
            }
        )
    v2_pass = all(r["abs_diff"] <= V2_ATOL for r in v2)
    out = {
        "task": "1.1 (timing, V2)",
        **meta,
        "timing": timing,
        "V2": {"atol": V2_ATOL, "pass": v2_pass, "draws": v2},
        "single_G_over_30s": min(g_times) > MAX_G_S,
    }
    (RESULTS / "check.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["V2"], indent=2))
    if out["single_G_over_30s"] or not v2_pass:
        raise SystemExit("STOP: V2 failed or one G took more than 30 s (spec §10)")


def full(args):
    kf, f, p, extras, scr0, meta = setup()
    hd = np.asarray(kf.inp["orf"])
    flat = p.reshape(-1, p.shape[-1])
    t = time.perf_counter()
    G = og.batched_G(f["grad_orf"], flat, args.batch)
    wall_G = time.perf_counter() - t
    finite = bool(np.all(np.isfinite(G)))
    G4 = G.reshape(4, 100, kf.Npsr, kf.Npsr)
    print(f"G for {len(flat)} draws in {wall_G:.0f} s, finite {finite}", flush=True)

    # V1: contraction against forward mode and a central difference, per draw
    idx = [c * 100 + d for c, d in V1_DRAWS]
    v1 = {}
    for name, orf in (("hd", hd), ("scramble_0", scr0)):
        contracted = og.score(G[idx], [orf])[:, 0]
        t = time.perf_counter()
        jvp = np.array([float(f["eps_jvp"](jnp.asarray(orf), jnp.asarray(flat[i]))) for i in idx])
        wall_jvp = time.perf_counter() - t
        fd = np.array(
            [
                (
                    float(f["eps_value"](jnp.asarray(orf), jnp.asarray(flat[i]), FD_H))
                    - float(f["eps_value"](jnp.asarray(orf), jnp.asarray(flat[i]), -FD_H))
                )
                / (2 * FD_H)
                for i in idx
            ]
        )
        v1[name] = {"contracted": contracted, "jvp": jvp, "fd": fd, "wall_jvp_s": wall_jvp}
    # Denominator fixed before looking: max(|direct|, median |S_HD| over these draws).
    scale = float(np.median(np.abs(v1["hd"]["jvp"])))
    v1_out = {"rtol": V1_RTOL, "fd_h": FD_H, "scale": scale, "draws": V1_DRAWS}
    for name, r in v1.items():
        den = np.maximum(np.abs(r["jvp"]), scale)
        rel = np.abs(r["contracted"] - r["jvp"]) / den
        rel_fd = np.abs(r["contracted"] - r["fd"]) / np.maximum(np.abs(r["fd"]), scale)
        v1_out[name] = {
            "contracted": r["contracted"].tolist(),
            "jvp": r["jvp"].tolist(),
            "fd": r["fd"].tolist(),
            "max_rel_vs_jvp": float(rel.max()),
            "max_rel_vs_fd": float(rel_fd.max()),
            "wall_jvp_s": r["wall_jvp_s"],
            "pass": bool(rel.max() <= V1_RTOL),
        }
    v1_out["pass"] = all(v1_out[n]["pass"] for n in v1)

    rerun_idx = [c * 100 + d for c, d in RERUN_DRAWS]
    G_re = og.batched_G(f["grad_orf"], flat[rerun_idx], args.batch)
    rerun_diff = float(np.max(np.abs(G_re - G[rerun_idx])))

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    g_path = OUTPUTS / "G_v3probe_eps000.npz"
    np.savez_compressed(g_path, G=G4, p=p, names=np.array(kf.names))
    np.savez_compressed(
        RESULTS / "G_means.npz",
        chain_mean_G=G4.mean(axis=1),
        pooled_mean_G=G4.mean(axis=(0, 1)),
        names=np.array(kf.names),
    )
    out = {
        "task": "1.4 (G for 400 draws), 1.2 (V1)",
        **meta,
        "batch": args.batch,
        "wall_G_s": wall_G,
        "wall_G_s_per_draw": wall_G / len(flat),
        "peak_bytes": og.peak_bytes(),
        "n_draws": len(flat),
        "all_finite": finite,
        "G_symmetry_max_abs": float(np.max(np.abs(G - np.swapaxes(G, 1, 2)))),
        "G_archive": {"path": str(g_path), "sha256": og.sha256(g_path)},
        "V1": v1_out,
        "rerun": {"draws": RERUN_DRAWS, "max_abs_diff": rerun_diff,
                  "atol": RERUN_ATOL, "pass": rerun_diff <= RERUN_ATOL},
    }
    (RESULTS / "full.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("wall_G_s", "all_finite", "rerun")}, indent=2))
    print(json.dumps({n: {k: v1_out[n][k] for k in ("max_rel_vs_jvp", "max_rel_vs_fd", "pass")} for n in v1}, indent=2))
    if not (finite and v1_out["pass"] and out["rerun"]["pass"]):
        raise SystemExit("STOP: a gate failed (finite, V1 or the rerun)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("check")
    c.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    fl = sub.add_parser("full")
    fl.add_argument("--batch", type=int, required=True)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    check(args) if args.mode == "check" else full(args)


if __name__ == "__main__":
    main()
