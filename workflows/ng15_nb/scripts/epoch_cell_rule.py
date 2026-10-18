#!/usr/bin/env python
"""D2's epoch-cell rule over all pulsars (task 2.5, amended 2026-10-18).

    python workflows/ng15_nb/scripts/epoch_cell_rule.py [--shard 0 --n-shards 4] \
        [--n-points 20] [--seed 2501]
    python workflows/ng15_nb/scripts/epoch_cell_rule.py --summarise

Epoch cells (one position scalar per ECORR epoch, DMX kept in β) are near-exact, so they
are each pulsar's reference: a pulsar uses epoch cells when its two-scalar DMX-window
likelihood's ΔlnL spread against its epoch-cell likelihood exceeds 0.01 nat at `--n-points` draws from its Phase 1
Discovery-OU posterior (the C1 point set). Both sides are the Argus marginal filter with
accum0 (`oracle.filter_loglike`); no dense oracle is needed, so J1713+0747 is included.
The gradient relative error is reported but does not gate (amended 2026-10-18): it is a
difference between two models, each with its exact gradient.

Writes workflows/ng15_nb/results/c1/epoch_rule/<pulsar>.json per pulsar, and with
`--summarise` the totals (pulsars needing epoch cells, and M_sum with and without them)
to workflows/ng15_nb/results/c1/epoch_rule.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "workflows" / "ng15_nb" / "compression"))
sys.path.insert(0, str(REPO / "workflows" / "ng15_nb" / "scripts"))
import compress  # noqa: E402
import ingest  # noqa: E402
import oracle  # noqa: E402
from c1_compression import POSTERIOR, posterior_points  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

OUT = REPO / "workflows/ng15_nb/results/c1/epoch_rule"
SPREAD_MAX = 0.01  # C1's spread bound; the gradient error is reported, not gated


def needs_epoch_cells(row):
    return bool(row["dlnL_spread"] > SPREAD_MAX)


def evaluate(cache, comp, pts, f_vg):
    fi = oracle.filter_inputs(comp)
    vg = [f_vg(jnp.asarray(p), fi) for p in pts]
    v = np.array([float(v) for v, _ in vg]) + comp["const"]
    return v, np.array([np.asarray(g) for _, g in vg])


def run_pulsar(path, n_points, seed, f_vg):
    cache = ingest.load(path)
    name = cache["name"]
    pts = posterior_points(name, n_points, np.random.default_rng(seed))
    window = compress.compress_pulsar(cache)
    epoch = compress.compress_pulsar(cache, per_epoch=True)
    vw, gw = evaluate(cache, window, pts, f_vg)
    ve, ge = evaluate(cache, epoch, pts, f_vg)
    dl = vw - ve
    gerr = np.max(np.abs(gw - ge), axis=1) / np.max(np.abs(ge), axis=1)
    row = {
        "pulsar": name,
        "n_toa": int(cache["toas"].size),
        "window_cells": int(window["times"].size),
        "window_velocity_steps": int(window["has_velocity"].sum()),
        "epoch_cells": int(epoch["times"].size),
        "window_m_cols": int(window["m"].shape[1]),
        "epoch_m_cols": int(epoch["m"].shape[1]),
        "points": pts.tolist(),
        "dlnL": dl.tolist(),
        "dlnL_spread": float(np.ptp(dl)),
        "grad_rel_err": gerr.tolist(),
        "grad_rel_err_max": float(gerr.max()),
    }
    row["needs_epoch_cells"] = needs_epoch_cells(row)
    return row


def summarise():
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))]
    for r in rows:  # re-apply the rule, so a change of rule needs no re-run
        r["needs_epoch_cells"] = needs_epoch_cells(r)
    need = [r for r in rows if r["needs_epoch_cells"]]
    m_window = sum(r["window_m_cols"] for r in rows)
    m_rule = sum(
        r["epoch_m_cols"] if r["needs_epoch_cells"] else r["window_m_cols"]
        for r in rows
    )
    steps_window = sum(r["window_cells"] + r["window_velocity_steps"] for r in rows)
    steps_rule = sum(
        (
            r["epoch_cells"]
            if r["needs_epoch_cells"]
            else r["window_cells"] + r["window_velocity_steps"]
        )
        for r in rows
    )
    result = {
        "n_pulsars": len(rows),
        "bounds": {"dlnL_spread": SPREAD_MAX},
        "needs_epoch_cells": [r["pulsar"] for r in need],
        "M_sum_window": m_window,
        "M_sum_rule": m_rule,
        "n_obs_window": steps_window,
        "n_obs_rule": steps_rule,
        "per_pulsar": {
            r["pulsar"]: {
                k: r[k]
                for k in (
                    "dlnL_spread",
                    "grad_rel_err_max",
                    "needs_epoch_cells",
                    "window_m_cols",
                    "epoch_m_cols",
                )
            }
            for r in rows
        },
    }
    (OUT.parent / "epoch_rule.json").write_text(json.dumps(result, indent=1) + "\n")
    for r in sorted(rows, key=lambda r: -r["dlnL_spread"]):
        print(
            f"{r['pulsar']:12s} spread {r['dlnL_spread']:.2e} gerr {r['grad_rel_err_max']:.1e}"
            f" M {r['window_m_cols']}->{r['epoch_m_cols']}"
            + ("  EPOCH" if r["needs_epoch_cells"] else "")
        )
    print(
        f"{len(need)}/{len(rows)} need epoch cells; M_sum {m_window} -> {m_rule}; "
        f"observations {steps_window} -> {steps_rule}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ingest-dir", default=str(REPO / "workflows/ng15_nb/data/ingest"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--n-points", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2501)
    ap.add_argument("--summarise", action="store_true")
    args = ap.parse_args()
    if args.summarise:
        summarise()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(args.ingest_dir).glob("*.npz"))[args.shard :: args.n_shards]
    f_vg = jax.jit(jax.value_and_grad(oracle.filter_loglike))
    for path in paths:
        row = run_pulsar(path, args.n_points, args.seed, f_vg)
        row["posterior_source"] = str(POSTERIOR.relative_to(REPO))
        row["device"] = str(jax.devices()[0])
        (OUT / f"{row['pulsar']}.json").write_text(json.dumps(row, indent=1) + "\n")
        print(
            f"{row['pulsar']} spread {row['dlnL_spread']:.2e} gerr {row['grad_rel_err_max']:.1e}"
            f" M {row['window_m_cols']}->{row['epoch_m_cols']}"
            + ("  EPOCH" if row["needs_epoch_cells"] else ""),
            flush=True,
        )


if __name__ == "__main__":
    main()
