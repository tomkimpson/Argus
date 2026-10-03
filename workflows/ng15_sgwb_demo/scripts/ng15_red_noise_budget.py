#!/usr/bin/env python
"""Can Argus's OU per-pulsar red noise represent the red noise NANOGrav sees in NG15?

Steps 1 and 2 of the OU-adequacy check (notes/RESULTS_ou_adequacy_ng15.md). CPU only.

**Step 1 -- red-noise budget.** For every canonical NG15 wideband pulsar, read the
published single-pulsar power-law red-noise posterior (``{PSR}_red_noise_log10_A``,
``{PSR}_red_noise_gamma`` in the release's PTMCMC chains) and compare its one-sided
residual PSD ``P(f) = A^2/(12 pi^2) (f/f_yr)^-gamma f_yr^-3`` with the white floor of the
data Argus actually sees: TOAs inverse-variance binned on a 30-day grid (as
``build_aligned_feathers.py`` does), with the collapsed scalar EFAC/EQUAD of
``data/ng15_psr_noise_full.json``. The one-sided white PSD is ``P_w = 2 sigma_w^2 dt`` with
``sigma_w^2 = 1/mean(1/sigma_k^2)`` over the binned epochs (the inverse-variance-weighted
noise level) and ``dt = T/N_epoch``. Frequencies are ``f_k = k/T``.

  * selected  <=>  5th percentile of P(f_1) > P_w   (red noise confidently above the floor)
  * in-band   =    the f_k (k <= 30, NANOGrav's red-noise basis) where the
                   posterior-median P(f_k) > P_w

**Step 2 -- can one OU spectrum match it?** For each selected pulsar, fit the ONE-sided
OU residual PSD

    S_OU(f) = 2 sigma_r^2 / (w^2 (gamma_p^2 + w^2)),   w = 2 pi f,  sigma_r = sigma_p/f0

to the posterior band of P over its in-band f_k, minimising the worst-case deviation from
the median in units of the band half-width. The factor 2 makes it one-sided like the
power law (``check_psd_sidedness.py``; same convention as ``inject_powerlaw_gwb.ou_psd``). PASS iff
the best-fit OU lies inside the 5-95% band at every in-band f_k. With <= 2 in-band bins a
2-parameter OU always fits, so those rows are flagged as uninformative.

Caveat: the reference band is a posterior *under a power-law model*. A wide band (weak
constraint) passes easily; step 3 (single-pulsar Argus fits) is the direct test.

Outputs: ``outputs/ng15_ou_adequacy/budget.json`` (consumed by step 3),
``outputs/ng15_ou_adequacy/step2_spectra.png`` and ``notes/ng15_red_noise_budget.md``.

    JAX_PLATFORMS=cpu python workflows/ng15_sgwb_demo/scripts/ng15_red_noise_budget.py
    JAX_PLATFORMS=cpu python workflows/ng15_sgwb_demo/scripts/ng15_red_noise_budget.py --self-test
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reduce_ng15_white_noise import NG15_ROOT  # noqa: E402
from stage_symlinks import _find_canonical, enumerate_canonical_pulsars  # noqa: E402

SEC_PER_DAY = 86400.0
F_YR = 1.0 / (365.25 * SEC_PER_DAY)
F_5YR = F_YR / 5.0
CADENCE_DAYS = 30.0
BURN_IN = 0.25
N_DRAWS = 2000
Q_LO, Q_HI = 5.0, 95.0
N_RN_FREQ = 30  # NG15 per-pulsar red-noise Fourier modes

_WF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_NOISE_JSON = os.path.join(_WF, "data", "ng15_psr_noise_full.json")
DEFAULT_OUT_DIR = os.path.join(_WF, "outputs", "ng15_ou_adequacy")
DEFAULT_NOTES = os.path.join(_WF, "notes", "ng15_red_noise_budget.md")

# OU fit search box (log10 gamma_p in s^-1 matches the Stage A prior [-12, -6]).
LOG10_GAMMA_GRID = np.linspace(-12.0, -6.0, 241)


# --------------------------------------------------------------------------------------
# PSDs (one-sided, s^3)
# --------------------------------------------------------------------------------------
def powerlaw_psd(f, log10_A, gamma):
    """One-sided enterprise power-law residual PSD; broadcasts draws x freqs.

    Matches ``inject_powerlaw_gwb.powerlaw_psd`` (copied so this script stays JAX-free).
    """
    A2 = 10.0 ** (2.0 * np.asarray(log10_A))
    return A2 / (12.0 * np.pi**2) * (f / F_YR) ** (-np.asarray(gamma)) * F_YR**-3


def ou_psd_onesided(f, log10_gamma_p, log10_sigma_r2):
    """One-sided OU residual PSD, ``2 sigma_r^2 / (w^2 (gamma^2 + w^2))``."""
    w = 2.0 * np.pi * np.asarray(f)
    g = 10.0**log10_gamma_p
    return 2.0 * 10.0**log10_sigma_r2 / (w**2 * (g**2 + w**2))


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------
def load_red_noise_draws(psr, noise_dir, n_draws=N_DRAWS, burn_in=BURN_IN):
    """Return (log10_A, gamma) arrays of thinned post-burn-in posterior draws."""
    with open(os.path.join(noise_dir, f"{psr}.wb.pars.txt")) as f:
        names = [line.strip() for line in f if line.strip()]
    ia = names.index(f"{psr}_red_noise_log10_A")
    ig = names.index(f"{psr}_red_noise_gamma")
    chain = np.loadtxt(
        os.path.join(noise_dir, f"{psr}.wb.chain_1.txt"), usecols=(ia, ig)
    )
    chain = chain[int(burn_in * len(chain)) :]
    idx = np.linspace(0, len(chain) - 1, min(n_draws, len(chain))).astype(int)
    return chain[idx, 0], chain[idx, 1]


def read_tim(psr, tim_dir):
    """Return (mjd, err_s) for the canonical wideband tim of ``psr``."""
    path = _find_canonical(tim_dir, psr, "tim")
    mjd, err = [], []
    with open(path) as f:
        for line in f:
            tok = line.split()
            if len(tok) < 5 or tok[0] in ("C", "FORMAT", "MODE"):
                continue
            mjd.append(float(tok[2]))
            err.append(float(tok[3]) * 1e-6)  # us -> s
    return np.array(mjd), np.array(err)


def binned_white_floor(mjd, err, efac, log10_equad, cadence=CADENCE_DAYS):
    """One-sided white PSD of the 30-day inverse-variance-binned data.

    Returns (P_w, T_seconds, n_epoch).
    """
    idx = np.floor((mjd - mjd.min()) / cadence).astype(int)
    occupied = np.unique(idx)
    inv = np.bincount(idx, weights=1.0 / err**2)[occupied]
    sig_bin2 = 1.0 / inv
    var = efac**2 * sig_bin2 + 10.0 ** (2.0 * log10_equad)
    sigma_w2 = 1.0 / np.mean(1.0 / var)
    T = (mjd.max() - mjd.min()) * SEC_PER_DAY
    n = occupied.size
    return 2.0 * sigma_w2 * T / n, T, n


# --------------------------------------------------------------------------------------
# Step 2 fit
# --------------------------------------------------------------------------------------
def fit_ou_to_band(f, lo, med, hi):
    """Minimax fit of a one-sided OU PSD to a log10 band; returns a result dict.

    ``lo``, ``med``, ``hi`` are log10 PSDs at frequencies ``f``. The objective is
    ``max_k |log10 S_OU(f_k) - med_k| / h_k`` with ``h_k`` the half-width on the side
    the model lies (so an asymmetric band is handled exactly). The amplitude is profiled
    on a grid for every gamma, then the best grid point is polished with Nelder-Mead.
    """
    lo, med, hi = map(np.asarray, (lo, med, hi))

    def shape(lg):
        return np.log10(ou_psd_onesided(f, lg, 0.0))

    def objective(p):
        model = shape(p[0]) + p[1]
        d = model - med
        h = np.where(d > 0, hi - med, med - lo)
        return np.max(np.abs(d) / np.maximum(h, 1e-6))

    best = (np.inf, None)
    for lg in LOG10_GAMMA_GRID:
        s = shape(lg)
        base = np.median(med - s)
        for amp in base + np.linspace(-1.5, 1.5, 151):
            v = objective((lg, amp))
            if v < best[0]:
                best = (v, (lg, amp))
    # Bounded polish: gamma_p stays in the Stage A prior box (outside it the corner sits
    # far above the band and the OU is just its f^-2 asymptote anyway).
    res = minimize(
        objective,
        best[1],
        method="Nelder-Mead",
        bounds=[(LOG10_GAMMA_GRID[0], LOG10_GAMMA_GRID[-1]), (None, None)],
        options={"xatol": 1e-4, "fatol": 1e-5},
    )
    p = np.array(res.x if res.fun <= best[0] else best[1], dtype=float)
    model = shape(p[0]) + p[1]
    inside = bool(np.all((model >= lo - 1e-9) & (model <= hi + 1e-9)))
    return {
        "log10_gamma_p": float(p[0]),
        "log10_sigma_r2": float(p[1]),
        "objective": float(objective(p)),
        "max_dev_dex": float(np.max(np.abs(model - med))),
        "corner_hz": float(10.0 ** p[0] / (2.0 * np.pi)),
        "inside_band": inside,
        "model_log10": model.tolist(),
    }


def self_test():
    """Exact-OU band must pass at ~0 dex; a steep (gamma=6) power law must fail."""
    f = np.arange(1, 31) / (15.0 / F_YR)
    true = np.log10(ou_psd_onesided(f, -8.3, -30.0))
    r = fit_ou_to_band(f, true - 0.2, true, true + 0.2)
    ok1 = r["inside_band"] and r["max_dev_dex"] < 0.02
    print(
        f"exact OU   : inside={r['inside_band']} max_dev={r['max_dev_dex']:.4f} dex "
        f"gamma={r['log10_gamma_p']:.3f} (true -8.3)  -> {'OK' if ok1 else 'FAIL'}"
    )
    pl = np.log10(powerlaw_psd(f, -13.0, 6.0))
    r = fit_ou_to_band(f, pl - 0.2, pl, pl + 0.2)
    ok2 = not r["inside_band"]
    print(
        f"PL gamma=6 : inside={r['inside_band']} max_dev={r['max_dev_dex']:.3f} dex"
        f"  -> {'OK' if ok2 else 'FAIL'}"
    )
    pl = np.log10(powerlaw_psd(f, -13.0, 3.0))
    r = fit_ou_to_band(f, pl - 0.2, pl, pl + 0.2)
    print(
        f"PL gamma=3 : inside={r['inside_band']} max_dev={r['max_dev_dex']:.3f} dex"
        "  (informational: inside OU's 2-4 slope range)"
    )
    return ok1 and ok2


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def analyse_pulsar(psr, noise_dir, tim_dir, wn, f0):
    """Budget (step 1) and, if selected, OU fit (step 2) for one pulsar."""
    log10_A, gamma = load_red_noise_draws(psr, noise_dir)
    mjd, err = read_tim(psr, tim_dir)
    P_w, T, n_ep = binned_white_floor(mjd, err, wn["efac"], wn["equad"])
    f = np.arange(1, n_ep // 2 + 1) / T
    logP = np.log10(powerlaw_psd(f[None, :], log10_A[:, None], gamma[:, None]))
    lo, med, hi = np.percentile(logP, [Q_LO, 50.0, Q_HI], axis=0)
    log_pw = np.log10(P_w)
    # NANOGrav's red-noise model has only N_RN_FREQ Fourier modes (k/T, k <= 30), so its
    # posterior says nothing above 30/T: the in-band set is capped there.
    in_band = (med > log_pw) & (np.arange(1, f.size + 1) <= N_RN_FREQ)
    p5 = np.log10(powerlaw_psd(F_5YR, log10_A, gamma))
    row = {
        "psr": psr,
        "f0": f0,
        "T_yr": T * F_YR,
        "n_epoch": int(n_ep),
        "log10_P_white": float(log_pw),
        "log10_A_med": float(np.median(log10_A)),
        "gamma_med": float(np.median(gamma)),
        "p_gamma_2_4": float(np.mean((gamma >= 2) & (gamma <= 4))),
        "red_over_white_f1_dex": float(med[0] - log_pw),
        "red_over_white_5yr_dex": float(np.median(p5) - log_pw),
        "selected": bool(lo[0] > log_pw),
        "n_in_band": int(in_band.sum()),
    }
    if row["selected"]:
        # In-band = the low-frequency run above the floor (P is monotone in f).
        k = np.flatnonzero(in_band)
        fit = fit_ou_to_band(f[k], lo[k], med[k], hi[k])
        fit["log10_sigma_p"] = 0.5 * fit["log10_sigma_r2"] + np.log10(f0)
        fit["informative"] = bool(k.size >= 3)
        row["fit"] = fit
        row["band"] = {
            "f": f[k].tolist(),
            "lo": lo[k].tolist(),
            "med": med[k].tolist(),
            "hi": hi[k].tolist(),
        }
        row["band_full"] = {
            "f": f.tolist(),
            "lo": lo.tolist(),
            "med": med.tolist(),
            "hi": hi.tolist(),
        }
    return row


def read_f0(psr, par_dir):
    """Spin frequency F0 (Hz) from the canonical wideband par file."""
    path = _find_canonical(par_dir, psr, "par")
    with open(path) as f:
        for line in f:
            tok = line.split()
            if tok and tok[0] == "F0":
                return float(tok[1])
    raise ValueError(f"no F0 in {path}")


def write_notes(rows, path):
    """Markdown table of the budget plus the step-2 verdicts."""
    sel = [r for r in rows if r["selected"]]
    lines = [
        "# NG15 wideband per-pulsar red-noise budget (steps 1-2 of the OU-adequacy check)",
        "",
        "Generated by `scripts/ng15_red_noise_budget.py`. Published power-law red-noise "
        "posteriors (NG15 wideband noise chains, 25% burn-in, 2000 draws) vs the one-sided "
        "white floor of the 30-day-binned data with the collapsed scalar EFAC/EQUAD. "
        "`f_1 = 1/T`. Dex columns are log10(median red PSD / white PSD). "
        "**Selected** = 5th-percentile red PSD at f_1 above the white floor.",
        "",
        "| pulsar | T [yr] | N_ep | log10 A | gamma | P(2<=gamma<=4) | red/white @f_1 [dex] "
        "| red/white @1/5yr [dex] | in-band bins | selected |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: -r["red_over_white_f1_dex"]):
        lines.append(
            f"| {r['psr']} | {r['T_yr']:.1f} | {r['n_epoch']} | {r['log10_A_med']:.2f} | "
            f"{r['gamma_med']:.2f} | {r['p_gamma_2_4']:.2f} | "
            f"{r['red_over_white_f1_dex']:+.2f} | {r['red_over_white_5yr_dex']:+.2f} | "
            f"{r['n_in_band']} | {'**yes**' if r['selected'] else 'no'} |"
        )
    lines += [
        "",
        f"**{len(sel)} of {len(rows)} pulsars selected.**",
        "",
        "## Step 2: best one-sided OU spectrum vs the published 5-95% band (in-band bins only)",
        "",
        "PASS = best-fit OU inside the band at every in-band frequency. Objective = worst "
        "deviation in band half-widths (<= 1 is inside). Rows with < 3 in-band bins are "
        "uninformative (a 2-parameter OU fits any 2 points).",
        "",
        "| pulsar | in-band bins | f range [nHz] | objective | max dev from median [dex] "
        "| log10 gamma_p | corner [nHz] | log10 sigma_p | verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(sel, key=lambda r: -r["n_in_band"]):
        ft = r["fit"]
        fr = r["band"]["f"]
        verdict = "PASS" if ft["inside_band"] else "**FAIL**"
        if not ft["informative"]:
            verdict += " (uninformative)"
        lines.append(
            f"| {r['psr']} | {r['n_in_band']} | {fr[0]*1e9:.2f}-{fr[-1]*1e9:.2f} | "
            f"{ft['objective']:.2f} | {ft['max_dev_dex']:.3f} | {ft['log10_gamma_p']:.2f} | "
            f"{ft['corner_hz']*1e9:.2f} | {ft['log10_sigma_p']:.2f} | {verdict} |"
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def plot(rows, path):
    """One panel per selected pulsar: PL band, best OU, white floor."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel = sorted([r for r in rows if r["selected"]], key=lambda r: -r["n_in_band"])
    if not sel:
        return
    nc = 4
    nr = int(np.ceil(len(sel) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 3.2 * nr), squeeze=False)
    for ax, r in zip(axes.flat, sel):
        b = r["band_full"]
        f = np.array(b["f"])
        ax.fill_between(
            f,
            10 ** np.array(b["lo"]),
            10 ** np.array(b["hi"]),
            alpha=0.3,
            label="NG15 PL 5-95%",
        )
        ax.plot(f, 10 ** np.array(b["med"]), lw=1)
        ft = r["fit"]
        ax.plot(
            f,
            ou_psd_onesided(f, ft["log10_gamma_p"], ft["log10_sigma_r2"]),
            "k--",
            lw=1.2,
            label="best OU",
        )
        ax.axhline(10 ** r["log10_P_white"], color="grey", ls=":", label="white floor")
        ax.axvline(r["band"]["f"][-1], color="grey", lw=0.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        tag = "PASS" if ft["inside_band"] else "FAIL"
        ax.set_title(f"{r['psr']}  [{tag}, {r['n_in_band']} bins]", fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(sel) :]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=6)
    fig.supxlabel("f [Hz]")
    fig.supylabel("one-sided residual PSD [s$^3$]")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    """Parse args, run steps 1-2 over all canonical pulsars, write outputs."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ng15-wideband", default=NG15_ROOT)
    ap.add_argument("--noise-json", default=DEFAULT_NOISE_JSON)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--notes", default=DEFAULT_NOTES)
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="Validate the OU fitter on synthetic bands and exit",
    )
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if self_test() else 1)

    wn_all = json.load(open(args.noise_json))
    root = args.ng15_wideband
    psrs = enumerate_canonical_pulsars(os.path.join(root, "par"))
    rows = []
    for psr in psrs:
        f0 = read_f0(psr, os.path.join(root, "par"))
        r = analyse_pulsar(
            psr, os.path.join(root, "noise"), os.path.join(root, "tim"), wn_all[psr], f0
        )
        tag = ""
        if r["selected"]:
            tag = "PASS" if r["fit"]["inside_band"] else "FAIL"
        print(
            f"{psr:<12} red/white@f1 {r['red_over_white_f1_dex']:+6.2f} dex  "
            f"in-band {r['n_in_band']:3d}  {'SELECTED ' + tag if r['selected'] else ''}"
        )
        rows.append(r)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "budget.json"), "w") as f:
        json.dump(rows, f, indent=1)
    write_notes(rows, args.notes)
    plot(rows, os.path.join(args.out_dir, "step2_spectra.png"))
    sel = [r for r in rows if r["selected"]]
    print(
        f"\n{len(sel)}/{len(rows)} selected; "
        f"{sum(r['fit']['inside_band'] for r in sel)} pass step 2."
    )


if __name__ == "__main__":
    main()
