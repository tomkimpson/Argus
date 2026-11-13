#!/usr/bin/env python
"""Lever A's pass/fail readout for the v3 probe (spec §12, 2026-11-13; task 5.2c).

    python workflows/ng15_nb/array/probe3_criteria.py <output_dir> \
        [--out workflows/ng15_nb/results/array/probe3_v3_criteria.json]

Reads `<id>_results.nc` or `<id>_partial.nc` with h5py, plus the checkpoint's step sizes,
and scores the approved criteria. They are benchmarked on the lever-A preview of probe 2,
not on "off the ceiling": 25 % of the restricted mass lies within 0.5 dex of −7.0 by
construction.

PASS needs all of:
- A1 every chain's γa median within `--gamma-a-spread` (0.3 dex) of every other's;
  the pooled median's distance from −7.9 (the preview's) is reported, not gated;
- A2 J1713+0747 and J0645+5158 in their on-and-flat state (σp > −16 and γp > −6, the
  diagnosis script's "white-like" state) in ≥ 70 % of every chain's draws. A chain that
  holds J1713 off below the ceiling is the named failure mode: the on-ridge is too thin
  for NUTS to hold, and lever B is next;
- A3 split r̂ of γa and of the pivot ≤ 1.05;
- A4 the worst γp and σp split r̂ both below probe 2's, and fewer σp sites over 1.1 than
  probe 2's, all with B1855+09 left out on both sides (probe 2 without it: γp 1.10,
  J1903+0327; σp 1.25, J0030+0451; 7 σp sites over 1.1);
- A5 the smallest per-chain step size not below probe 2's (0.019).
B1855+09 (a split between two on states, uncoupled from γa) is reported but not gated.

`--restrict-ceiling X` scores the draws with γa ≤ X only, which emulates a lower ceiling on
an existing run (the γa prior is uniform). Chains then have unequal lengths, so split r̂
(A3, A4) and step sizes (A5) are not scored. On probe 2 at −7.0 it reproduces the preview
(153 draws kept, J1713 on 0/97/71/8 %) and fails A2.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "single_psr"))
import common  # noqa: E402  (manifest order = the array's pulsar order)
from probe_gamma_a_diagnosis import git_sha, split_rhat  # noqa: E402

GATED = ("J1713+0747", "J0645+5158")
REPORTED = ("B1855+09",)
# Probe 2's split r̂ with B1855+09 left out (this script's split_rhat on its partial .nc).
PROBE2 = {"gamma_p_rhat": 1.097, "sigma_p_rhat": 1.246, "n_sigma_p_over_1.1": 7,
          "min_step_size": 0.019, "gamma_a_preview_median": -7.9}


def load(out_dir):
    run_id = out_dir.name
    nc = out_dir / f"{run_id}_results.nc"
    if not nc.exists():
        nc = out_dir / f"{run_id}_partial.nc"
    if not nc.exists():
        raise SystemExit(f"no results or partial .nc in {out_dir}")
    with h5py.File(nc, "r") as f:
        post = f["posterior"]
        d = {k: post[k][...] for k in ("log10_gamma_a", "log10_pivot_psd", "log10_γp", "log10_σp")}
    step = None
    ck = out_dir / f"{run_id}_checkpoint.pkl"
    if ck.exists():
        try:
            state = pickle.load(open(ck, "rb"))["sampler_state"]
            step = np.asarray(state.adapt_state.step_size).ravel().tolist()
        except Exception as e:  # noqa: BLE001
            step = f"unreadable: {e!r}"
    return nc, d, step


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir")
    ap.add_argument("--out")
    ap.add_argument("--restrict-ceiling", type=float)
    ap.add_argument("--gamma-a-spread", type=float, default=0.3)
    ap.add_argument("--on-frac", type=float, default=0.70)
    ap.add_argument("--rhat-max", type=float, default=1.05)
    ap.add_argument("--on-threshold", type=float, default=-16.0)
    ap.add_argument("--flat-threshold", type=float, default=-6.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    nc, d, step = load(out_dir)
    ga, pv, gp, sp = d["log10_gamma_a"], d["log10_pivot_psd"], d["log10_γp"], d["log10_σp"]
    n_chains, n_draws, n_psr = gp.shape
    names = common.pulsars()
    assert n_psr == len(names), (n_psr, len(names))
    keep = ga <= args.restrict_ceiling if args.restrict_ceiling is not None else np.ones_like(ga, bool)
    on = (sp > args.on_threshold) & (gp > args.flat_threshold)

    per_chain = []
    for c in range(n_chains):
        k = keep[c]
        row = {"chain": c, "n": int(k.sum()),
               "gamma_a_median": float(np.median(ga[c][k])) if k.any() else None}
        for p in GATED + REPORTED:
            i = names.index(p)
            row[f"{p}_on_frac"] = float(on[c, k, i].mean()) if k.any() else None
            row[f"{p}_sigma_p_median"] = float(np.median(sp[c, k, i])) if k.any() else None
        per_chain.append(row)

    meds = [r["gamma_a_median"] for r in per_chain if r["gamma_a_median"] is not None]
    spread = float(max(meds) - min(meds)) if len(meds) == n_chains else None
    crit = {}
    crit["A1_gamma_a_agreement"] = {
        "spread_dex": spread, "limit": args.gamma_a_spread,
        "pooled_median": float(np.median(ga[keep])) if keep.any() else None,
        "preview_median": PROBE2["gamma_a_preview_median"],
        "pass": spread is not None and spread <= args.gamma_a_spread}
    on_fracs = {p: [r[f"{p}_on_frac"] for r in per_chain] for p in GATED}
    off_chains = {p: [r["chain"] for r in per_chain
                      if r[f"{p}_on_frac"] is None or r[f"{p}_on_frac"] < args.on_frac]
                  for p in GATED}
    crit["A2_owner_on"] = {
        "on_frac_by_chain": on_fracs, "limit": args.on_frac,
        "chains_below": off_chains,
        "pass": all(not v for v in off_chains.values())}

    if args.restrict_ceiling is None:
        r_gp = np.array([split_rhat(gp[:, :, i]) for i in range(n_psr)])
        r_sp = np.array([split_rhat(sp[:, :, i]) for i in range(n_psr)])
        gated = np.array([n not in REPORTED for n in names])
        g_gp, g_sp = np.where(gated, r_gp, -np.inf), np.where(gated, r_sp, -np.inf)
        r_ga, r_pv = split_rhat(ga), split_rhat(pv)
        crit["A3_common_rhat"] = {"gamma_a": r_ga, "pivot": r_pv, "limit": args.rhat_max,
                                  "pass": r_ga <= args.rhat_max and r_pv <= args.rhat_max}
        crit["A4_per_pulsar_rhat"] = {
            "excluded": list(REPORTED),
            "max_gamma_p": float(g_gp.max()), "worst_gamma_p": names[int(g_gp.argmax())],
            "max_sigma_p": float(g_sp.max()), "worst_sigma_p": names[int(g_sp.argmax())],
            "n_sigma_p_over_1.1": int((g_sp > 1.1).sum()),
            "n_gamma_p_over_1.1": int((g_gp > 1.1).sum()),
            "probe2": {k: PROBE2[k] for k in ("gamma_p_rhat", "sigma_p_rhat", "n_sigma_p_over_1.1")},
            "B1855+09_sigma_p_rhat_not_gated": float(r_sp[names.index("B1855+09")]),
            "pass": bool(g_gp.max() < PROBE2["gamma_p_rhat"]
                         and g_sp.max() < PROBE2["sigma_p_rhat"]
                         and (g_sp > 1.1).sum() < PROBE2["n_sigma_p_over_1.1"])}
        ok = isinstance(step, list) and len(step) > 0
        crit["A5_step_size"] = {"per_chain": step, "probe2_min": PROBE2["min_step_size"],
                                "pass": bool(ok and min(step) >= PROBE2["min_step_size"])}
        scored = list(crit)
    else:
        scored = ["A1_gamma_a_agreement", "A2_owner_on"]

    verdict = "PASS" if all(crit[k]["pass"] for k in scored) else "FAIL"
    report = {
        "run": out_dir.name, "nc": str(nc), "criteria_sha": git_sha(HERE),
        "n_chains": n_chains, "n_draws_per_chain": n_draws,
        "restrict_ceiling": args.restrict_ceiling, "n_kept": int(keep.sum()),
        "on_state": f"log10 σp > {args.on_threshold} and log10 γp > {args.flat_threshold}",
        "per_chain": per_chain, "criteria": crit, "scored": scored, "verdict": verdict,
        "note": None if args.restrict_ceiling is None else
        "restricted draws: A3-A5 not scored (unequal chain lengths, step sizes are v2's)",
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
        print(f"wrote {args.out}")
    print(json.dumps({"verdict": verdict, "per_chain": per_chain,
                      **{k: crit[k]["pass"] for k in scored}}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
