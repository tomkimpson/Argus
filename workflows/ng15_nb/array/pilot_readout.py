#!/usr/bin/env python
"""Readout of the 67-psr ε = 0 rung against Phase 5's Done-when (spec task 5.2).

    python workflows/ng15_nb/array/pilot_readout.py <output_dir> [--log <slurm .out>] \
        [--out workflows/ng15_nb/results/array/<output_id>.json]

Reads `<id>_results.nc`, or `<id>_partial.nc` while a run is going (the reliable progress
signal), and the checkpoint's final sampler state. Reports:
- the Done-when numbers: max r̂, divergences as a share of draws, min bulk ESS (and min
  tail ESS), over every sampled site and pulsar;
- per chain, the freeze check: step size, inverse-mass spread, final potential energy,
  the share of draws that repeat the previous draw, the median lnL and the medians of the
  two GW sites. A chain is flagged `frozen` if more than half its draws repeat, or its
  step size is below a tenth of the other chains' median. Medians that sit apart are
  reported as `split_lnL` (more than 3 pooled per-chain sd from the other chains' median);
- with `--log`, the median s/iteration from the SLURM log's progress bars.

Thresholds: r̂ ≤ 1.01, divergences ≤ 0.5%, bulk ESS ≥ 400, no frozen chain. The §10
early warning (any step size < 0.01) is reported separately.
"""

import argparse
import json
import pickle
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

RHAT_MAX = 1.01
DIV_MAX = 0.005
ESS_MIN = 400
STEP_WARN = 0.01
# The production model's sampled sites (run_array.py --check traces them). Other posterior
# variables are deterministic (log10 h_a, γa, ε, the fixed EFAC/EQUAD).
SAMPLED = ("log10_pivot_psd", "log10_gamma_a", "log10_γp", "log10_σp")


def git_sha(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


def load(out_dir):
    import arviz as az

    out_dir = Path(out_dir)
    run_id = out_dir.name
    final = out_dir / f"{run_id}_results.nc"
    partial = out_dir / f"{run_id}_partial.nc"
    nc = final if final.exists() else partial
    if not nc.exists():
        raise SystemExit(f"no {final.name} or {partial.name} in {out_dir}")
    ck = out_dir / f"{run_id}_checkpoint.pkl"
    state = pickle.load(open(ck, "rb"))["sampler_state"] if ck.exists() else None
    return run_id, nc, az.from_netcdf(nc), state


def sampled_sites(idata, state):
    """The sampled sites: the mass matrix's keys when there is a state, else SAMPLED."""
    if state is not None:
        imm = state.adapt_state.inverse_mass_matrix
        names = [n for key in imm for n in key]
    else:
        names = [n for n in SAMPLED if n in idata.posterior]
    return idata.posterior[names]


def convergence(idata, post):
    import arviz as az

    rhat = az.rhat(post, method="rank")
    bulk = az.ess(post, method="bulk")
    tail = az.ess(post, method="tail")

    def extreme(ds, fn):
        return float(fn([fn(np.asarray(ds[v])) for v in ds.data_vars]))

    def worst(ds, fn):
        vals = {v: fn(np.asarray(ds[v])) for v in ds.data_vars}
        k = max(vals, key=vals.get) if fn is np.nanmax else min(vals, key=vals.get)
        return k

    div = np.asarray(idata.sample_stats["diverging"])
    n_draws = int(div.size)
    return {
        "n_chains": int(post.sizes["chain"]),
        "n_draws_per_chain": int(post.sizes["draw"]),
        "rhat_max": extreme(rhat, np.nanmax),
        "rhat_max_site": worst(rhat, np.nanmax),
        "ess_bulk_min": extreme(bulk, np.nanmin),
        "ess_bulk_min_site": worst(bulk, np.nanmin),
        "ess_tail_min": extreme(tail, np.nanmin),
        "divergences": int(div.sum()),
        "divergence_frac": float(div.sum() / n_draws),
    }


def per_chain(idata, post, state):
    n_chain = post.sizes["chain"]
    flat = np.concatenate(
        [np.asarray(post[v]).reshape(n_chain, post.sizes["draw"], -1) for v in post.data_vars],
        axis=2,
    )
    repeat = np.mean(np.all(np.diff(flat, axis=1) == 0, axis=2), axis=1)
    ll = np.asarray(idata.log_likelihood[list(idata.log_likelihood.data_vars)[0]])
    ll = ll.reshape(n_chain, post.sizes["draw"], -1).sum(axis=2)
    med = np.median(ll, axis=1)
    sd = np.std(ll, axis=1)
    chains = []
    step = (
        np.asarray(state.adapt_state.step_size).reshape(-1)
        if state is not None
        else np.full(n_chain, np.nan)
    )
    for c in range(n_chain):
        others = [i for i in range(n_chain) if i != c]
        rec = {
            "chain": c,
            "step_size": float(step[c]),
            "repeat_frac": float(repeat[c]),
            "lnL_median": float(med[c]),
            "lnL_sd": float(sd[c]),
            "medians": {
                s: float(np.median(np.asarray(post[s])[c]))
                for s in ("log10_pivot_psd", "log10_gamma_a")
                if s in post
            },
        }
        if state is not None:
            # {sites: (chain, d) diagonal or (chain, d, d) dense block}
            diag = np.concatenate(
                [
                    np.asarray(m)[c] if np.ndim(m) == 2 else np.diag(np.asarray(m)[c])
                    for m in state.adapt_state.inverse_mass_matrix.values()
                ]
            )
            rec["inverse_mass_log10_range"] = [
                float(np.log10(diag.min())),
                float(np.log10(diag.max())),
            ]
            rec["potential_energy_final"] = float(
                np.asarray(state.potential_energy).reshape(-1)[c]
            )
        rec["frozen"] = bool(
            repeat[c] > 0.5
            or (np.isfinite(step[c]) and step[c] < 0.1 * np.median(step[others]))
        )
        pooled = np.sqrt(np.mean(sd[others] ** 2))
        rec["split_lnL"] = bool(abs(med[c] - np.median(med[others])) > 3 * pooled)
        chains.append(rec)
    return chains


def s_per_it(log):
    text = Path(log).read_text(errors="replace")
    rates = [float(r) for r in re.findall(r"([\d.]+)s/it[\],]", text)]
    return float(np.median(rates)) if rates else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir")
    ap.add_argument("--log")
    ap.add_argument("--out")
    args = ap.parse_args()

    run_id, nc, idata, state = load(args.out_dir)
    post = sampled_sites(idata, state)
    conv = convergence(idata, post)
    conv["sampled_sites"] = list(post.data_vars)
    chains = per_chain(idata, post, state)
    steps = [c["step_size"] for c in chains]
    rec = {
        "spec_task": "5.2",
        "run": run_id,
        "nc": str(nc),
        "complete": nc.name.endswith("_results.nc"),
        "readout_sha": git_sha(Path(__file__).resolve().parent),
        **conv,
        "chains": chains,
        "s_per_it_median": s_per_it(args.log) if args.log else None,
        "step_size_warning": bool(np.nanmin(steps) < STEP_WARN),
        "thresholds": {
            "rhat_max": RHAT_MAX,
            "divergence_frac": DIV_MAX,
            "ess_bulk_min": ESS_MIN,
        },
    }
    prov = Path(args.out_dir) / "provenance.json"
    if prov.exists():
        p = json.loads(prov.read_text())
        rec["config_sha256"] = p.get("config_sha256")
        rec["library_sha"] = p.get("git_sha")  # stamp_provenance.py --repo
    rec["passes_done_when"] = bool(
        rec["complete"]
        and conv["rhat_max"] <= RHAT_MAX
        and conv["divergence_frac"] <= DIV_MAX
        and conv["ess_bulk_min"] >= ESS_MIN
        and not any(c["frozen"] for c in chains)
    )
    text = json.dumps(rec, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    sys.exit(main())
