#!/usr/bin/env python
"""Check C4 and figure F4: per-pulsar red noise, Argus vs Discovery-OU (spec task 3.3).

Reads results/single_psr/{argus,discovery}/<psr>.json (`common.summarise`) and compares
each pulsar's one-sided spin-noise residual PSD posterior at f = 1/T and 1/(5 yr).

"Within 1σ" (C4 does not define it; Phase 3 takes this reading) means the two medians
differ by at most the two posteriors' widths in quadrature,

    z = |m_A − m_D| / √(σ_A² + σ_D²) ≤ 1,   σ = (q84 − q16) / 2,

on log10 PSD. A pulsar passes C4 when z ≤ 1 at both frequencies, and C4 passes when
≥ 90% of the pulsars do. Pulsars piling up at the lower σp edge get the same test: both
posteriors are then prior-dominated upper limits, and they should still agree.

Also tallies tasks 3.1 and 3.2's diagnostics (r̂, bulk ESS, divergences, rails) and the
lower-edge pulsars for each pipeline.

    conda activate Argus
    python workflows/ng15_nb/single_psr/c4_compare.py
Writes results/single_psr/c4.json and results/single_psr/f4_psd_scatter.png.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

C4_FRACTION = 0.9
Z_MAX = 1.0


def z_score(a, d):
    """|Δ median| over the quadrature sum of half the 16-84% widths."""
    sa = 0.5 * (a["q84"] - a["q16"])
    sd = 0.5 * (d["q84"] - d["q16"])
    return abs(a["median"] - d["median"]) / math.hypot(sa, sd)


def compare(rec_a, rec_d):
    """Per-pulsar C4 row from the two pipelines' summaries."""
    z = {
        f: z_score(rec_a["log10_psd"][f], rec_d["log10_psd"][f])
        for f in rec_a["log10_psd"]
    }
    return {
        "pulsar": rec_a["pulsar"],
        "shallow_spectrum": rec_a["shallow_spectrum"],
        "z": z,
        "within_1sigma": bool(all(v <= Z_MAX for v in z.values())),
        "argus": rec_a["log10_psd"],
        "discovery": rec_d["log10_psd"],
        "lower_edge_pileup": {
            "argus": rec_a["lower_edge_pileup"],
            "discovery": rec_d["lower_edge_pileup"],
        },
        "at_f2_limit": {
            "argus": rec_a.get("at_f2_limit", False),
            "discovery": rec_d.get("at_f2_limit", False),
        },
    }


def rejudge(rec):
    """Re-apply the current edge rule and 3.1 verdict to a stored summary.

    Summaries written before the 2026-10-20 amendment gated the γp ceiling; their edge
    masses are unchanged, so the verdict is recomputed from them.
    """
    rec.update(common.verdict(rec["edge_mass"]))
    rec["passes_3_1"] = common.passes_3_1(rec["diagnostics"], rec)
    return rec


def tally(recs):
    """Task 3.1/3.2 verify-clause counts for one pipeline."""
    fails = {
        "r_hat": [
            r["pulsar"] for r in recs if r["diagnostics"]["max_r_hat"] > common.RHAT_MAX
        ],
        "ess": [
            r["pulsar"]
            for r in recs
            if r["diagnostics"]["min_ess_bulk"] < common.ESS_MIN
        ],
        "divergences": [
            r["pulsar"] for r in recs if r["diagnostics"]["divergences"] > 0
        ],
        "rails": {r["pulsar"]: r["rails"] for r in recs if r["rails"]},
    }
    return {
        "n": len(recs),
        "n_pass": sum(r["passes_3_1"] for r in recs),
        "fails": fails,
        "lower_edge_pileup": [r["pulsar"] for r in recs if r["lower_edge_pileup"]],
        "at_f2_limit": [r["pulsar"] for r in recs if r["at_f2_limit"]],
        "gamma_p_floor_mass_gt_5pct": [
            r["pulsar"] for r in recs if r["edge_mass"]["log10_gamma_p"]["floor"] > 0.05
        ],
        "max_r_hat": max(r["diagnostics"]["max_r_hat"] for r in recs),
        "min_ess_bulk": min(r["diagnostics"]["min_ess_bulk"] for r in recs),
        "total_divergences": sum(r["diagnostics"]["divergences"] for r in recs),
    }


def figure(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    freqs = list(rows[0]["z"])
    fig, axes = plt.subplots(1, len(freqs), figsize=(5.2 * len(freqs), 5))
    for ax, f in zip(axes, freqs):
        for r in rows:
            a, d = r["argus"][f], r["discovery"][f]
            colour = (
                "C3"
                if not r["within_1sigma"]
                else ("C1" if r["shallow_spectrum"] else "C0")
            )
            marker = "v" if r["lower_edge_pileup"]["argus"] else "o"
            ax.errorbar(
                d["median"],
                a["median"],
                xerr=[[d["median"] - d["q16"]], [d["q84"] - d["median"]]],
                yerr=[[a["median"] - a["q16"]], [a["q84"] - a["median"]]],
                fmt=marker,
                color=colour,
                ms=4,
                lw=0.8,
                alpha=0.85,
            )
            if not r["within_1sigma"]:
                ax.annotate(r["pulsar"], (d["median"], a["median"]), fontsize=7)
        lo, hi = ax.get_xlim()
        lo, hi = min(lo, ax.get_ylim()[0]), max(hi, ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], "k:", lw=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"Discovery-OU log10 S(f = {f}) [s²/Hz]")
        ax.set_ylabel(f"Argus log10 S(f = {f}) [s²/Hz]")
    handles = [
        plt.Line2D([], [], color="C0", marker="o", ls="", label="within 1σ"),
        plt.Line2D(
            [], [], color="C1", marker="o", ls="", label="within 1σ, shallow spectrum"
        ),
        plt.Line2D([], [], color="C3", marker="o", ls="", label="outlier"),
        plt.Line2D(
            [],
            [],
            color="grey",
            marker="v",
            ls="",
            label="lower σp edge (no detection)",
        ),
    ]
    axes[0].legend(handles=handles, fontsize=8, loc="upper left")
    fig.suptitle("F4: per-pulsar spin-noise PSD, median and 16–84%, 67 NG15 pulsars")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=str(common.RESULTS))
    args = ap.parse_args()
    root = Path(args.results)
    names = common.pulsars()
    load = lambda p, n: json.loads((root / p / f"{n}.json").read_text())  # noqa: E731
    missing = {
        p: [n for n in names if not (root / p / f"{n}.json").exists()]
        for p in ("argus", "discovery")
    }
    if any(missing.values()):
        raise SystemExit(f"missing summaries: {missing}")
    recs = {p: [rejudge(load(p, n)) for n in names] for p in ("argus", "discovery")}
    rows = [compare(a, d) for a, d in zip(recs["argus"], recs["discovery"])]
    n_in = sum(r["within_1sigma"] for r in rows)
    per_freq = {
        f: sum(r["z"][f] <= Z_MAX for r in rows) / len(rows) for f in rows[0]["z"]
    }
    out = {
        "definition": "z = |m_A - m_D| / hypot(sigma_A, sigma_D), sigma = (q84 - q16)/2, "
        "on log10 one-sided spin PSD; within 1 sigma iff z <= 1 at both frequencies",
        "freqs_hz": recs["argus"][0]["freqs_hz"],
        "T_yr": recs["argus"][0]["T_yr"],
        "n_pulsars": len(rows),
        "n_within_1sigma": n_in,
        "fraction_within_1sigma": n_in / len(rows),
        "fraction_within_1sigma_per_freq": per_freq,
        "c4_threshold": C4_FRACTION,
        "passes_c4": n_in / len(rows) >= C4_FRACTION,
        "outliers": [
            {
                "pulsar": r["pulsar"],
                "z": r["z"],
                "shallow_spectrum": r["shallow_spectrum"],
            }
            for r in rows
            if not r["within_1sigma"]
        ],
        "task_3_1_argus": tally(recs["argus"]),
        "task_3_2_discovery": tally(recs["discovery"]),
        "argus_git_sha": sorted({r["git_sha"] for r in recs["argus"]}),
        "discovery_git_sha": sorted({r["git_sha"] for r in recs["discovery"]}),
        "per_pulsar": rows,
        "git_sha": common.git_sha(),
    }
    (root / "c4.json").write_text(json.dumps(out, indent=2) + "\n")
    figure(rows, root / "f4_psd_scatter.png")
    print(
        json.dumps({k: v for k, v in out.items() if k != "per_pulsar"}, indent=2),
        flush=True,
    )
    print(
        f"max z: {max(max(r['z'].values()) for r in rows):.3f}; "
        f"median z: {np.median([max(r['z'].values()) for r in rows]):.3f}"
    )


if __name__ == "__main__":
    main()
