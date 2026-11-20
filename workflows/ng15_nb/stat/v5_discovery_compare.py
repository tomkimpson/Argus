#!/usr/bin/env python
"""V5: Argus's pooled-mean S against the Discovery OU-14 v3 ε = 0 integrand (task 1.6).

    python workflows/ng15_nb/stat/v5_discovery_compare.py

Comparison only (spec D3): this is the one file under `stat/` that reads Discovery output,
and nothing it reads feeds any Argus computation. The reference is the **v3**-box rung,
the box the Argus draws were sampled in:
- mean and MC uncertainty: `results/lnb_hd_curn_ou14_v3.json` rungs[0] (tracked);
- draw-to-draw sd and chain means: `outputs/ng15nb_ou14_v3_eps000/summary.json`
  (gitignored; path and sha-256 recorded).
§3's draft numbers (24.6, sd 4.1) are the v1 box and are not used.

The two quantities are the same derivative of two different likelihoods (14 common
Fourier modes against Argus's full-band OU common process), so V5 is reported and only
loosely gated; the draft gate is |Δ| ≤ 1 Discovery draw sd.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import orf_gradient as og  # noqa: E402

RESULTS = og.WF / "results/stat/opening"
LNB = og.WF / "results/lnb_hd_curn_ou14_v3.json"
SUMMARY = og.DEFAULT_INGEST.parent.parent / "outputs/ng15nb_ou14_v3_eps000/summary.json"


def main():
    rung = json.loads(LNB.read_text())["rungs"][0]
    assert rung["eps"] == 0.0
    disc = json.loads(SUMMARY.read_text())
    assert abs(disc["integrand_mean"] - rung["mean"]) < 1e-12
    argus = json.loads((RESULTS / "readout.json").read_text())
    S = np.load(RESULTS / "per_draw_S_hd.npz")["S"]
    chain_means = S.mean(axis=1)
    a_mean = float(S.mean())
    a_chain_sd = float(chain_means.std(ddof=1))
    a_draw_sd = float(S.std(ddof=1))
    d_sd = float(disc["integrand_sd"])
    diff = a_mean - rung["mean"]
    out = {
        "spec": "spec/detection-statistic.md task 1.6 (V5)",
        "argus": {
            "pooled_mean_S": a_mean,
            "pooled_mean_S_readout": argus["pooled"]["S"],
            "chain_means": chain_means.tolist(),
            "chain_mean_sd": a_chain_sd,
            "se_from_chain_means": a_chain_sd / np.sqrt(len(chain_means)),
            "draw_sd": a_draw_sd,
            "n_draws": int(S.size),
            "note": "4 x 100 unconverged v3 probe draws; not a posterior",
        },
        "discovery_ou14_v3_eps000": {
            "mean": rung["mean"],
            "mc_uncert": rung["uncert"],
            "draw_sd": d_sd,
            "chain_means": disc["integrand_chain_means"],
            "n_draws": rung["n_draws"],
            "sources": {
                "lnb_json": str(LNB.relative_to(og.REPO_ROOT)),
                "summary_json": str(SUMMARY),
                "summary_sha256": og.sha256(SUMMARY),
            },
        },
        "difference": diff,
        "difference_in_discovery_draw_sd": diff / d_sd,
        "draft_gate": "|difference| <= 1 Discovery draw sd",
        "draft_gate_pass": bool(abs(diff) <= d_sd),
    }
    (RESULTS / "v5.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
