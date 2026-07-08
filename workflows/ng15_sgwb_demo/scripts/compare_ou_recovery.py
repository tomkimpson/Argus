#!/usr/bin/env python
"""T2.4 decision-gate analysis: does the single-corner OU recovery absorb a true
power-law GWB without biasing the recovered *band* amplitude?

Argus has no power-law spectral model -- its GWB is always a single-corner
Ornstein-Uhlenbeck (OU) red-noise process under a fixed Hellings-Downs template.
Stage 2 injects a synthetic signal onto the real NG15 sampling geometry (so truth is
known) and recovers it with that OU model. This script performs the shape-agnostic,
*band-referenced* comparison the injection truth sidecar was built for:

  * A PTA constrains only ~1 decade of frequency (~1/T .. a few/T), so the robust
    observable is the residual power spectral density (PSD) at a pivot frequency near
    the sensitive band, NOT the spectral index. We compare the recovered OU PSD at the
    pivot to the injected power-law PSD at the same pivot.
  * Primary pivot: f = 1/(5 yr) (~2.5/T over this ~12 yr baseline, near the most
    sensitive band). Also reported: f = 1/yr (higher, less constrained).

PSD conventions (must match scripts/inject_powerlaw_gwb.py):
  * injected power-law residual PSD  P(f) = A^2/(12 pi^2) (f/f_yr)^-gamma f_yr^-3  [s^3]
    (recorded in injection_truth.json as psd_at_freqs_s3 + pivot_psd_s3)
  * recovered OU residual PSD        S_r(f) = sigma_a2 / ((2 pi f)^2 (gamma_a^2 + (2 pi f)^2))
    with sigma_a2 = (ha^2/12) gamma_a   (ha = 10^log10_ha, gamma_a = 10^log10_gamma_a)

Pure post-processing (arviz / numpy / matplotlib); no GPU, no argus import, no library
edits. Run on CPU from the repo root:

    JAX_PLATFORMS=cpu python workflows/ng15_sgwb_demo/scripts/compare_ou_recovery.py

Outputs:
  outputs/ng15_inject_powerlaw/comparison.json           (machine-readable verdict)
  outputs/ng15_inject_powerlaw/plots/spectral_overlay.{png,pdf}
"""
import argparse
import json
import os

import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- constants (match inject_powerlaw_gwb.py) ---
SEC_PER_YEAR = 365.25 * 86400.0
F_YR = 1.0 / SEC_PER_YEAR

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # workflows/ng15_sgwb_demo


def powerlaw_psd(freqs, log10_A, gamma):
    """Injected power-law residual PSD [s^3]. Matches inject_powerlaw_gwb.py:194."""
    A = 10.0**log10_A
    return (A**2 / (12.0 * np.pi**2)) * (freqs / F_YR) ** (-gamma) * F_YR**-3


def ou_residual_psd(freqs, log10_ha, log10_gamma_a):
    """Recovered OU residual PSD [s^3]. Matches the OU-truth note in injection_truth.json.

    Broadcasts: freqs (..., Nf) against draws (Ndraw, ...) -> (Ndraw, Nf) if inputs are
    arranged as column/row. Here we call it per-frequency-grid with vector draws.
    """
    ha = 10.0**np.asarray(log10_ha)
    gamma_a = 10.0**np.asarray(log10_gamma_a)
    sigma_a2 = (ha**2 / 12.0) * gamma_a
    w = 2.0 * np.pi * np.asarray(freqs)
    return sigma_a2 / (w**2 * (gamma_a**2 + w**2))


def _summ(a):
    """(median, std, hdi 3%/97%) of a 1-D sample array."""
    hdi = az.hdi(np.asarray(a), hdi_prob=0.94)
    return float(np.median(a)), float(np.std(a)), float(hdi[0]), float(hdi[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="powerlaw", choices=["powerlaw", "ou"],
                    help="which injection to analyse (default: powerlaw = T2.4 gate)")
    ap.add_argument("--run-prefix", default="ng15_inject",
                    help="output-dir/results prefix: outputs/<prefix>_<mode>/ (default "
                         "ng15_inject = lite runs; use ng15_confirm for the hi-res re-runs)")
    ap.add_argument("--gamma-a-prior", nargs=2, type=float, default=[-10.5, -7.0],
                    metavar=("MIN", "MAX"),
                    help="log10_gamma_a prior edges for the rail check (config default)")
    ap.add_argument("--rail-tol", type=float, default=0.25,
                    help="dex from a prior edge counted as 'railing' (default 0.25)")
    args = ap.parse_args()

    tag = f"{args.run_prefix}_{args.mode}"
    out_dir = os.path.join(ROOT, "outputs", tag)
    truth_path = os.path.join(ROOT, "data", f"inject_{args.mode}", "injection_truth.json")
    nc_path = os.path.join(out_dir, f"{tag}_results.nc")
    for p in (truth_path, nc_path):
        if not os.path.exists(p):
            raise SystemExit(f"missing input: {p}\n(has the SLURM run completed?)")

    with open(truth_path) as f:
        truth = json.load(f)

    # --- recovered OU posterior draws ---
    post = az.from_netcdf(nc_path).posterior
    log10_ha = post["log10_ha"].values.reshape(-1)
    log10_ga = post["log10_gamma_a"].values.reshape(-1)
    ha_s = _summ(log10_ha)
    ga_s = _summ(log10_ga)

    T_yr = truth["T_span_yr"]
    # Band-reference pivots (Hz). Prefer values recorded in the power-law truth sidecar;
    # otherwise compute them (OU truth records no pivots).
    pivots = {"1/(5yr)": 1.0 / (5.0 * SEC_PER_YEAR), "1/yr": F_YR}

    def injected_psd_at(f):
        # power-law truth records exact pivots; else evaluate the recorded shape.
        if args.mode == "powerlaw":
            return powerlaw_psd(np.array([f]), truth["log10_A_gw"], truth["gamma"])[0]
        # OU injection: evaluate the OU PSD at the injected (ha, gamma_a).
        return float(ou_residual_psd(f, truth["log10_ha"], truth["log10_gamma_a"]))

    comparison = {
        "mode": args.mode,
        "T_span_yr": T_yr,
        "recovered": {
            "log10_ha": {"median": ha_s[0], "std": ha_s[1], "hdi94": [ha_s[2], ha_s[3]]},
            "log10_gamma_a": {"median": ga_s[0], "std": ga_s[1], "hdi94": [ga_s[2], ga_s[3]]},
            "n_draws": int(log10_ha.size),
        },
        "band_referenced": {},
    }

    print("=" * 78)
    print(f"T2.4 band-referenced comparison  (mode={args.mode}, T={T_yr:.2f} yr)")
    print("=" * 78)
    print(f"recovered log10_ha       = {ha_s[0]:+.3f} +/- {ha_s[1]:.3f}  "
          f"(94% HDI [{ha_s[2]:+.3f}, {ha_s[3]:+.3f}])")
    print(f"recovered log10_gamma_a  = {ga_s[0]:+.3f} +/- {ga_s[1]:.3f}  "
          f"(94% HDI [{ga_s[2]:+.3f}, {ga_s[3]:+.3f}])")
    if args.mode == "powerlaw":
        print(f"injected power-law       : log10_A_gw={truth['log10_A_gw']}, "
              f"gamma={truth['gamma']:.4f}")

    for label, f in pivots.items():
        s_pl = injected_psd_at(f)                       # scalar [s^3]
        s_ou = ou_residual_psd(f, log10_ha, log10_ga)   # (Ndraw,) [s^3]
        l_ou = np.log10(s_ou)
        l_pl = np.log10(s_pl)
        med, std, lo, hi = _summ(l_ou)
        bias_psd_dex = med - l_pl                        # + => OU over-estimates PSD
        bias_amp_dex = bias_psd_dex / 2.0                # PSD ~ A^2
        sigma = bias_psd_dex / std if std > 0 else float("nan")
        comparison["band_referenced"][label] = {
            "f_hz": f,
            "injected_psd_s3": s_pl,
            "recovered_psd_s3_median": float(10.0**med),
            "log10_psd_injected": l_pl,
            "log10_psd_recovered": {"median": med, "std": std, "hdi94": [lo, hi]},
            "bias_psd_dex": bias_psd_dex,
            "bias_amplitude_dex": bias_amp_dex,
            "sigma_level": sigma,
        }
        print("-" * 78)
        print(f"pivot f = {label}  ({f:.3e} Hz)")
        print(f"  injected  log10 PSD = {l_pl:+.3f}  (PSD {s_pl:.3e} s^3)")
        print(f"  recovered log10 PSD = {med:+.3f} +/- {std:.3f}  "
              f"(94% HDI [{lo:+.3f}, {hi:+.3f}])")
        print(f"  bias = {bias_psd_dex:+.3f} dex (PSD) = {bias_amp_dex:+.3f} dex (amp) "
              f"= {sigma:+.2f} sigma")

    # --- gamma_a rail check ---
    gmin, gmax = args.gamma_a_prior
    d_lo = ga_s[0] - gmin
    d_hi = gmax - ga_s[0]
    frac_near_lo = float(np.mean(log10_ga - gmin < args.rail_tol))
    frac_near_hi = float(np.mean(gmax - log10_ga < args.rail_tol))
    railed = (d_lo < args.rail_tol) or (d_hi < args.rail_tol) or \
             (frac_near_lo > 0.3) or (frac_near_hi > 0.3)
    comparison["gamma_a_rail_check"] = {
        "prior": [gmin, gmax],
        "median_dist_to_low_edge_dex": d_lo,
        "median_dist_to_high_edge_dex": d_hi,
        "frac_draws_within_tol_of_low": frac_near_lo,
        "frac_draws_within_tol_of_high": frac_near_hi,
        "railed": bool(railed),
    }
    print("-" * 78)
    print(f"gamma_a rail check (prior [{gmin}, {gmax}], tol {args.rail_tol} dex):")
    print(f"  median dist to edges: low {d_lo:.2f} dex, high {d_hi:.2f} dex; "
          f"frac near low {frac_near_lo:.2f}, near high {frac_near_hi:.2f}")
    print(f"  RAILED: {railed}  "
          f"(a low-edge rail = OU corner straining to mimic f^-{truth.get('gamma', 0):.2f})"
          if args.mode == "powerlaw" else f"  RAILED: {railed}")

    # --- spectral overlay plot ---
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    f_modes = np.array(truth["freqs_hz"]) if "freqs_hz" in truth else None
    f_grid = np.logspace(np.log10(1.0 / (T_yr * SEC_PER_YEAR)),
                         np.log10(2.0 * F_YR), 200)

    # recovered OU PSD band across the grid (subsample draws for speed)
    idx = np.linspace(0, log10_ha.size - 1, min(400, log10_ha.size)).astype(int)
    ou_grid = np.vstack([ou_residual_psd(f_grid, log10_ha[i], log10_ga[i]) for i in idx])
    ou_med = np.median(ou_grid, axis=0)
    ou_lo, ou_hi = np.percentile(ou_grid, [16, 84], axis=0)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    if args.mode == "powerlaw":
        pl_curve = powerlaw_psd(f_grid, truth["log10_A_gw"], truth["gamma"])
        ax.plot(f_grid, pl_curve, color="C3", lw=2,
                label=f"injected power-law (log10A={truth['log10_A_gw']}, "
                      f"$\\gamma$={truth['gamma']:.2f})")
        if f_modes is not None and "psd_at_freqs_s3" in truth:
            ax.scatter(f_modes, truth["psd_at_freqs_s3"], color="C3", s=18, zorder=5,
                       label="injected PSD at Fourier modes")
    ax.plot(f_grid, ou_med, color="C0", lw=2, label="recovered OU (median)")
    ax.fill_between(f_grid, ou_lo, ou_hi, color="C0", alpha=0.25,
                    label="recovered OU (68%)")
    for label, f in pivots.items():
        ax.axvline(f, color="0.5", ls=":", lw=1)
        ax.text(f, ax.get_ylim()[0] if False else 1, label, rotation=90,
                va="bottom", ha="right", fontsize=8, color="0.4",
                transform=ax.get_xaxis_transform())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("residual PSD [s$^3$]")
    ax.set_title(f"T2.4 spectral overlay: power-law injection vs OU recovery\n"
                 f"(NG15 6-pulsar subset, {truth.get('nepoch', '?')} epochs, "
                 f"T={T_yr:.1f} yr)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(plots_dir, f"spectral_overlay.{ext}"), dpi=150)
    plt.close(fig)
    print("-" * 78)
    print(f"spectral overlay -> {os.path.join(plots_dir, 'spectral_overlay.png')}")

    # --- verdict + dump ---
    primary = comparison["band_referenced"]["1/(5yr)"]
    comparison["verdict"] = {
        "pivot": "1/(5yr)",
        "bias_amplitude_dex": primary["bias_amplitude_dex"],
        "sigma_level": primary["sigma_level"],
        "within_1sigma": bool(abs(primary["sigma_level"]) <= 1.0),
    }
    cmp_path = os.path.join(out_dir, "comparison.json")
    with open(cmp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print("=" * 78)
    v = comparison["verdict"]
    print(f"VERDICT @ f=1/(5yr): band-amplitude bias {v['bias_amplitude_dex']:+.3f} dex "
          f"({v['sigma_level']:+.2f} sigma) -> "
          f"{'WITHIN ~1 sigma (greenlight candidate)' if v['within_1sigma'] else 'BIASED (gate finding)'}")
    print(f"comparison.json -> {cmp_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
