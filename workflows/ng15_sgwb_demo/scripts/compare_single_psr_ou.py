#!/usr/bin/env python
"""OU-adequacy step 3 readout: Argus single-pulsar OU fits vs the published NG15 band.

For every pulsar selected by ``ng15_red_noise_budget.py`` with a finished run
``outputs/ng15_single_<PSR>/ng15_single_<PSR>_results.nc`` (slurm_scripts/ng15_single_psr.sh):

1. Health gates from ``extract_stage_a.health_check`` (r_hat, ESS, finite). Railing
   against a prior edge is reported but not gating (see ``main``).
2. The posterior 5-95% band of the ONE-sided Argus OU residual PSD
   ``2 sigma_p^2 / (f0^2 w^2 (gamma_p^2 + w^2))`` (one-sided, as ``inject_powerlaw_gwb.ou_psd``)
   at the pulsar's in-band frequencies from step 1.
3. PASS iff the Argus band overlaps the published NG15 power-law 5-95% band at EVERY
   in-band frequency. A health failure makes the row UNREADABLE, not FAIL.

Caveat: Argus's data treatment differs from NANOGrav's (30-day binning, DMX dropped from the
design matrix, one scalar EFAC/EQUAD), so a FAIL here with a step-2 PASS points at the data
treatment rather than the OU kernel.

Outputs ``notes/ng15_single_psr_ou.md`` and ``outputs/ng15_ou_adequacy/step3_spectra.png``.

    JAX_PLATFORMS=cpu python workflows/ng15_sgwb_demo/scripts/compare_single_psr_ou.py
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_stage_a import health_check, load_run  # noqa: E402
from ng15_red_noise_budget import DEFAULT_OUT_DIR, Q_HI, Q_LO  # noqa: E402

_WF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_NOTES = os.path.join(_WF, "notes", "ng15_single_psr_ou.md")
N_DRAWS = 2000


def ou_band(f, log10_gamma, log10_sigma, f0):
    """5/50/95% log10 one-sided OU residual PSD over posterior draws at freqs ``f``."""
    idx = np.linspace(0, log10_gamma.size - 1, min(N_DRAWS, log10_gamma.size)).astype(int)
    g = 10.0 ** log10_gamma[idx, None]
    s2 = 10.0 ** (2.0 * log10_sigma[idx, None])
    w = 2.0 * np.pi * np.asarray(f)[None, :]
    logS = np.log10(2.0 * s2 / (f0**2 * w**2 * (g**2 + w**2)))
    return np.percentile(logS, [Q_LO, 50.0, Q_HI], axis=0)


def main():
    """Read every finished run, apply gates, compare bands, write notes and plot."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--budget", default=os.path.join(DEFAULT_OUT_DIR, "budget.json"))
    ap.add_argument("--outputs", default=os.path.join(_WF, "outputs"))
    ap.add_argument("--notes", default=DEFAULT_NOTES)
    ap.add_argument("--rhat-max", type=float, default=1.05)
    ap.add_argument("--ess-min", type=float, default=200.0)
    ap.add_argument("--gamma-range", type=float, nargs=2, default=(-12.0, -4.0))
    ap.add_argument("--sigma-range", type=float, nargs=2, default=(-20.0, -9.0))
    args = ap.parse_args()

    rows = [r for r in json.load(open(args.budget)) if r["selected"]]
    results = []
    for r in rows:
        psr = r["psr"]
        nc = os.path.join(args.outputs, f"ng15_single_{psr}", f"ng15_single_{psr}_results.nc")
        if not os.path.exists(nc):
            print(f"{psr:<12} no results file -- skipped")
            continue
        run = load_run(nc)
        _, reasons = health_check(run, args.rhat_max, args.ess_min,
                                  tuple(args.gamma_range), tuple(args.sigma_range))
        # Railing is reported, not gating: a gamma_p pinned at the ceiling means the OU is
        # at its f^-2 limit, which is the shallow-spectrum diagnostic itself.
        rails = [x for x in reasons if "railed" in x]
        reasons = [x for x in reasons if "railed" not in x]
        ok = not reasons
        b = r["band"]
        f = np.array(b["f"])
        lo, med, hi = ou_band(f, run["log10_gamma"], run["log10_sigma"], r["f0"])
        overlap = (lo <= np.array(b["hi"])) & (hi >= np.array(b["lo"]))
        offset = med - np.array(b["med"])
        verdict = "UNREADABLE" if not ok else ("PASS" if overlap.all() else "FAIL")
        full = r["band_full"]
        flo, fmed, fhi = ou_band(np.array(full["f"]), run["log10_gamma"],
                                 run["log10_sigma"], r["f0"])
        results.append({
            "psr": psr, "verdict": verdict, "reasons": reasons, "rails": rails,
            "n_in_band": len(f), "n_overlap": int(overlap.sum()),
            "max_abs_offset_dex": float(np.max(np.abs(offset))),
            "offset_f1_dex": float(offset[0]),
            "log10_gamma_med": float(np.median(run["log10_gamma"])),
            "log10_sigma_med": float(np.median(run["log10_sigma"])),
            "step2_log10_gamma": r["fit"]["log10_gamma_p"],
            "step2_log10_sigma": r["fit"]["log10_sigma_p"],
            "step2_pass": r["fit"]["inside_band"],
            "rhat": max(run["rhat_gamma"], run["rhat_sigma"]),
            "ess": min(run["ess_gamma"], run["ess_sigma"]),
            "div": run["divergence_frac"],
            "plot": {"f": full["f"], "pl": [full["lo"], full["med"], full["hi"]],
                     "ou": [flo.tolist(), fmed.tolist(), fhi.tolist()],
                     "white": r["log10_P_white"], "f_edge": f[-1]},
        })
        print(f"{psr:<12} {verdict:<10} overlap {overlap.sum()}/{len(f)}  "
              f"max|offset| {np.max(np.abs(offset)):.2f} dex  {'; '.join(reasons + rails)}")

    write_notes(results, args.notes)
    plot(results, os.path.join(DEFAULT_OUT_DIR, "step3_spectra.png"))
    with open(os.path.join(DEFAULT_OUT_DIR, "step3.json"), "w") as fh:
        json.dump([{k: v for k, v in x.items() if k != "plot"} for x in results], fh, indent=1)


def write_notes(results, path):
    """Markdown table of the step-3 verdicts."""
    lines = [
        "# Step 3: single-pulsar Argus OU fits vs published NG15 power-law band",
        "",
        "Generated by `scripts/compare_single_psr_ou.py`. One-sided OU PSD (sidedness "
        "corrected). PASS = Argus 5-95% band overlaps the NG15 5-95% band at every in-band "
        "frequency. Offsets are log10(Argus median / NG15 median).",
        "",
        "| pulsar | verdict | overlap | max abs offset [dex] | offset @f_1 [dex] "
        "| log10 gamma_p (Argus / step 2) | log10 sigma_p (Argus / step 2) "
        "| step 2 | r_hat | ESS | div |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for x in sorted(results, key=lambda x: -x["n_in_band"]):
        v = x["verdict"] if x["verdict"] == "PASS" else f"**{x['verdict']}**"
        lines.append(
            f"| {x['psr']} | {v} | {x['n_overlap']}/{x['n_in_band']} | "
            f"{x['max_abs_offset_dex']:.2f} | {x['offset_f1_dex']:+.2f} | "
            f"{x['log10_gamma_med']:.2f} / {x['step2_log10_gamma']:.2f} | "
            f"{x['log10_sigma_med']:.2f} / {x['step2_log10_sigma']:.2f} | "
            f"{'PASS' if x['step2_pass'] else 'FAIL'} | {x['rhat']:.3f} | {x['ess']:.0f} | "
            f"{x['div']:.3f} |"
        )
    bad = [x for x in results if x["reasons"]]
    if bad:
        lines += ["", "Health failures:"]
        lines += [f"- {x['psr']}: {'; '.join(x['reasons'])}" for x in bad]
    railed = [x for x in results if x["rails"]]
    if railed:
        lines += ["", "Prior-edge railing (reported, not gating):"]
        lines += [f"- {x['psr']}: {'; '.join(x['rails'])}" for x in railed]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def plot(results, path):
    """One panel per pulsar: NG15 PL band, Argus OU band, white floor."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not results:
        return
    res = sorted(results, key=lambda x: -x["n_in_band"])
    nc = 4
    nr = int(np.ceil(len(res) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 3.2 * nr), squeeze=False)
    for ax, x in zip(axes.flat, res):
        p = x["plot"]
        f = np.array(p["f"])
        ax.fill_between(f, 10 ** np.array(p["pl"][0]), 10 ** np.array(p["pl"][2]),
                        alpha=0.3, color="C0", label="NG15 PL 5-95%")
        ax.fill_between(f, 10 ** np.array(p["ou"][0]), 10 ** np.array(p["ou"][2]),
                        alpha=0.3, color="C3", label="Argus OU 5-95%")
        ax.axhline(10 ** p["white"], color="grey", ls=":", label="white floor")
        ax.axvline(p["f_edge"], color="grey", lw=0.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{x['psr']}  [{x['verdict']}]", fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(res):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=6)
    fig.supxlabel("f [Hz]")
    fig.supylabel("one-sided residual PSD [s$^3$]")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
