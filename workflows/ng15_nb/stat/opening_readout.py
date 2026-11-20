#!/usr/bin/env python
"""Phase 1 readout: S per draw, pooled and per-chain significance, K1's inputs (task 1.5).

    python workflows/ng15_nb/stat/opening_readout.py

CPU only, from the saved G array (`opening_job.py full`) and the scramble archive (task
1.3), both hash-checked. Because S is linear in G, the mean of S over any set of draws is
S at the mean G, and the null for that mean is the mean G scored on every scramble:
- pooled: all 400 draws; per chain: each chain's 100;
- per-draw S for Γ_HD must reproduce V1's contracted values (≤ 1e-12 relative;
  rounding only, since the BLAS reduction order differs across nodes);
- p = (n + 1)/(N + 1) with n the scrambles scoring ≥ the observed value, binomial
  sd √(p(1 − p)/N); the floor at N = 5000 is 1/5001;
- z = (S − null mean)/null sd, reported for K1's draft replacement (not the gate).
Also the per-draw S band, and Spearman ρ of per-draw S against log10 σp of J1713+0747
and J0645+5158 (spec §10's B1 early warning).

`significance(G_mean, hd, orfs)` is the scoring step; Phase 2 reuses it on the optima.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import orf_gradient as og  # noqa: E402

RESULTS = og.WF / "results/stat/opening"
WATCH = ("J1713+0747", "J0645+5158")


def significance(G_mean, hd, orfs):
    """S of one (mean) G for the true ORF against the scrambles' S of the same G."""
    s = og.score(np.asarray(G_mean)[None], [hd] + list(orfs))[0]
    obs, null = float(s[0]), s[1:]
    N = len(null)
    n_ge = int(np.sum(null >= obs))
    p = (n_ge + 1) / (N + 1)
    return {
        "S": obs,
        "n_scrambles": N,
        "n_ge": n_ge,
        "p": p,
        "p_sd": float(np.sqrt(p * (1 - p) / N)),
        "p_floor": 1 / (N + 1),
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "z": float((obs - null.mean()) / null.std(ddof=1)),
        "null_q05_q50_q95": [float(q) for q in np.quantile(null, [0.05, 0.5, 0.95])],
    }


def without_uniform(orf):
    """`orf` with its off-diagonal mean removed (diagonal kept at 1).

    NG15's pulsars cluster on the sky, so Γ_HD's off-diagonal mean is 0.047 while an
    isotropic scramble's is ≈ 0: the scramble null tests HD against no correlation and
    cannot separate HD from a uniform (monopole-like) correlation. Scoring the mean-removed
    templates is HD with a free uniform term, a conservative variant (it also discards HD's
    own projection onto the uniform direction). Reported next to the headline, not gated.
    """
    n = orf.shape[0]
    off = ~np.eye(n, dtype=bool)
    out = np.array(orf, dtype=float)
    out[off] -= out[off].mean()
    return out


def load_inputs():
    full = json.loads((RESULTS / "full.json").read_text())
    scr = json.loads((RESULTS / "scrambles_summary.json").read_text())
    for rec in (full["G_archive"], scr):
        if og.sha256(rec["path"]) != rec["sha256"]:
            raise SystemExit(f"hash mismatch: {rec['path']}")
    with np.load(full["G_archive"]["path"]) as a:
        G4, names = a["G"], list(a["names"])
    with np.load(scr["path"]) as a:
        orfs, hd = a["orfs"], a["true_orf"]
    return full, scr, G4, names, orfs, hd


def main():
    from scipy.stats import spearmanr

    full, scr, G4, names, orfs, hd = load_inputs()
    n_chain, n_draw, n_psr, _ = G4.shape
    G = G4.reshape(-1, n_psr, n_psr)
    S_draw = og.score(G, [hd])[:, 0]

    # task 1.5's verify clause: per-draw S for Γ_HD reproduces task 1.2's values exactly
    idx = [c * n_draw + d for c, d in full["V1"]["draws"]]
    v1 = np.array(full["V1"]["hd"]["contracted"])
    # Not bitwise: the BLAS reduction order differs with the matrix shape and the CPU
    # (V1 was scored on the GPU node, 12 rows; this is 400 rows on another node).
    repro_rel = float(np.max(np.abs(S_draw[idx] - v1)) / np.max(np.abs(v1)))
    reproduces = repro_rel <= 1e-12

    pooled = significance(G.mean(axis=0), hd, orfs)
    pooled_no_uniform = significance(
        G.mean(axis=0), without_uniform(hd), [without_uniform(o) for o in orfs]
    )
    off = ~np.eye(n_psr, dtype=bool)
    uniform = {
        "hd_offdiag_mean": float(hd[off].mean()),
        "scramble_offdiag_mean_mean": float(np.mean([o[off].mean() for o in orfs])),
        "pooled_G_offdiag_mean": float(G.mean(axis=0)[off].mean()),
        "hd_with_free_uniform_term": pooled_no_uniform,
        "note": "scramble null tests HD vs no correlation; see without_uniform()",
    }
    chains = [significance(G4[c].mean(axis=0), hd, orfs) for c in range(n_chain)]
    ps = np.array([c["p"] for c in chains])
    zs = np.array([c["z"] for c in chains])

    _, extras = og.load_draws()
    sp = extras["log10_sigma_p"].reshape(-1, n_psr)
    rho = {}
    for name in WATCH:
        r = spearmanr(S_draw, sp[:, names.index(name)])
        # ρ only: a p-value would assume 400 independent draws (they are autocorrelated)
        rho[name] = {"spearman_rho": float(r.statistic)}

    k1 = {
        "pooled_p": pooled["p"],
        "pooled_p_sd": pooled["p_sd"],
        "trigger_pooled_p_gt_0.05": pooled["p"] > 0.05,
        "chain_p": ps.tolist(),
        "chain_p_span_factor": float(ps.max() / ps.min()),
        "trigger_chain_p_span_gt_10": bool(ps.max() / ps.min() > 10),
        "reported_only_chain_z": zs.tolist(),
        "reported_only_chain_z_span": float(zs.max() - zs.min()),
        "reported_only_z_span_gt_1": bool(zs.max() - zs.min() > 1),
    }
    k1["fires"] = k1["trigger_pooled_p_gt_0.05"] or k1["trigger_chain_p_span_gt_10"]
    out = {
        "spec": "spec/detection-statistic.md task 1.5",
        "inputs": {
            "G_archive": full["G_archive"],
            "scrambles": {k: scr[k] for k in ("path", "sha256", "seed", "n_scrambles")},
            "draws": full["draws"],
        },
        "per_draw_S_hd_reproduces_V1": reproduces,
        "per_draw_S_hd_vs_V1_max_rel": repro_rel,
        "reproduction_tolerance_rel": 1e-12,
        "per_draw_S_hd": {
            "mean": float(S_draw.mean()),
            "sd": float(S_draw.std(ddof=1)),
            "q05_q50_q95": [float(q) for q in np.quantile(S_draw, [0.05, 0.5, 0.95])],
            "min": float(S_draw.min()),
            "max": float(S_draw.max()),
            "chain_means": [
                float(S_draw[c * n_draw : (c + 1) * n_draw].mean())
                for c in range(n_chain)
            ],
        },
        "pooled": pooled,
        "uniform_component": uniform,
        "chains": chains,
        "S_vs_log10_sigma_p": rho,
        "K1": k1,
    }
    (RESULTS / "readout.json").write_text(json.dumps(out, indent=2) + "\n")
    np.savez_compressed(
        RESULTS / "per_draw_S_hd.npz", S=S_draw.reshape(n_chain, n_draw)
    )
    plot(S_draw, pooled, chains, G, orfs)
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "per_draw_S_hd_reproduces_V1",
                    "per_draw_S_hd_vs_V1_max_rel",
                    "per_draw_S_hd",
                    "pooled",
                    "K1",
                    "S_vs_log10_sigma_p",
                )
            },
            indent=2,
        )
    )
    if not reproduces:
        raise SystemExit("STOP: per-draw S does not reproduce V1's contracted values")


def plot(S_draw, pooled, chains, G, orfs):
    """Draft F2-style panel: pooled null, per-draw band, chain means (not the paper figure)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    null = og.score(G.mean(axis=0)[None], list(orfs))[0]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(null, bins=80, color="0.7", label=f"scramble null (N={len(null)})")
    lo, hi = np.quantile(S_draw, [0.05, 0.95])
    ax.axvspan(lo, hi, color="C0", alpha=0.15, label="per-draw S, 5–95 %")
    for c, r in enumerate(chains):
        ax.axvline(
            r["S"], color="C0", lw=0.8, ls="--", label="chain means" if c == 0 else None
        )
    ax.axvline(
        pooled["S"],
        color="C3",
        lw=1.5,
        label=f"pooled S, p {'≤' if pooled['n_ge'] == 0 else '='} {pooled['p']:.1e}"
        + (" (floor)" if pooled["n_ge"] == 0 else ""),
    )
    ax.set_xlabel(r"$S(\theta;\Gamma)=\partial\ln L/\partial\epsilon$ at $\epsilon=0$")
    ax.set_ylabel("scrambles")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "opening_F2_draft.png", dpi=150)


if __name__ == "__main__":
    main()
