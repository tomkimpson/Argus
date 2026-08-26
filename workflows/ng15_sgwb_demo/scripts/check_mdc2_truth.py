#!/usr/bin/env python
"""M1 truth-recovery gate: check an MDC2 array run against the injected GWB.

Argus's GWB is a single-corner OU process while MDC2 injected a power-law, so
raw log10_ha is not comparable to the injected log10_A. Following the T2.4/T3.3
band-referenced convention (compare_ou_recovery.py), the observable is the
residual PSD at a pivot frequency near the sensitive band:

  * injected power-law residual PSD  P(f) = A^2/(12 pi^2) (f/f_yr)^-gamma f_yr^-3 [s^3]
  * recovered OU residual PSD        S_r(f) = sigma_a2 / ((2 pi f)^2 (gamma_a^2 + (2 pi f)^2))
    with sigma_a2 = (ha^2/12) gamma_a

Pass criterion per run (the B-vs-C decision artifact for issue #111):
  1. injected pivot PSD inside the recovered 95% credible interval at f=1/(5yr),
  2. r_hat < 1.01 on the GW sites and divergence fraction < 1%,
  3. |median bias| < 1 sigma preferred (reported, not gating).

Injected truth: log10_A = -14.886 (group1_gw_parameters.json, dataset2). The
MDC2 repo does NOT record the injected spectral index anywhere (checked
2026-07-21: the json holds only the amplitude and the repo has no generation
scripts); gamma = 13/3 is the standard SMBHB value used by libstempo's
toasim.add_gwb default (alpha = -2/3), which is what every published MDC2
analysis assumes. Override with --gamma if better provenance appears.

Pure CPU post-processing:

    JAX_PLATFORMS=cpu python workflows/ng15_sgwb_demo/scripts/check_mdc2_truth.py \
        --run mdc2_stageB_hd --run mdc2_stageC_hd
"""

import argparse
import json
import os

import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEC_PER_YEAR = 365.25 * 86400.0
F_YR = 1.0 / SEC_PER_YEAR

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # workflows/ng15_sgwb_demo

MDC2_LOG10_A = -14.886056647693163  # group1_gw_parameters.json, dataset2
MDC2_GAMMA = 13.0 / 3.0  # assumed SMBHB standard — see module docstring


def powerlaw_psd(freqs, log10_A, gamma):
    """Injected power-law residual PSD [s^3]. Matches compare_ou_recovery.py."""
    A = 10.0**log10_A
    return (
        (A**2 / (12.0 * np.pi**2)) * (np.asarray(freqs) / F_YR) ** (-gamma) * F_YR**-3
    )


def ou_residual_psd(freqs, log10_ha, log10_gamma_a):
    """Recovered OU residual PSD [s^3]. Matches compare_ou_recovery.py."""
    ha = 10.0 ** np.asarray(log10_ha)
    gamma_a = 10.0 ** np.asarray(log10_gamma_a)
    sigma_a2 = (ha**2 / 12.0) * gamma_a
    w = 2.0 * np.pi * np.asarray(freqs)
    return sigma_a2 / (w**2 * (gamma_a**2 + w**2))


def check_run(tag, log10_A, gamma, results_dir):
    """Evaluate one run against the injected truth. Returns the verdict dict."""
    nc_path = os.path.join(results_dir, tag, f"{tag}_results.nc")
    if not os.path.exists(nc_path):
        raise SystemExit(f"missing input: {nc_path}\n(has the run completed?)")

    idata = az.from_netcdf(nc_path)
    post = idata.posterior
    log10_ha = post["log10_ha"].values.reshape(-1)
    log10_ga = post["log10_gamma_a"].values.reshape(-1)

    # Sampling health: r_hat on the actual SAMPLED latent sites, not derived
    # deterministics. In ridge mode (issue #109) log10_ha is a deterministic and
    # the free sites are log10_pivot_psd_prime / log10_gamma_a_prime; in direct
    # mode the free sites are log10_ha_prime / log10_gamma_a_prime (or the raw
    # names for a fixed param). Pick whichever pair is present.
    ridge = "log10_pivot_psd_prime" in post
    gw_sites = (
        ["log10_pivot_psd_prime", "log10_gamma_a_prime"]
        if ridge
        else ["log10_ha_prime", "log10_gamma_a_prime"]
    )
    gw_sites = [s for s in gw_sites if s in post]
    if not gw_sites:  # fully-fixed GW (shouldn't happen for B/C) -> use derived
        gw_sites = ["log10_ha", "log10_gamma_a"]
    rhat = az.rhat(idata, var_names=gw_sites)
    rhat_max = float(max(rhat[s].values.max() for s in gw_sites))
    div_frac = 0.0
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        div_frac = float(np.mean(idata.sample_stats["diverging"].values))

    pivots = {"1/(5yr)": 1.0 / (5.0 * SEC_PER_YEAR), "1/yr": F_YR}
    verdict = {
        "run": tag,
        "injected": {"log10_A": log10_A, "gamma": gamma},
        "recovered": {
            "log10_ha_median": float(np.median(log10_ha)),
            "log10_gamma_a_median": float(np.median(log10_ga)),
            "n_draws": int(log10_ha.size),
        },
        "health": {
            "rhat_max_gw": rhat_max,
            "divergence_frac": div_frac,
            "healthy": bool(rhat_max < 1.01 and div_frac < 0.01),
        },
        "band_referenced": {},
    }

    print("=" * 78)
    print(f"M1 truth gate: {tag}")
    print("=" * 78)
    print(
        f"health: max r_hat(GW) = {rhat_max:.4f}, divergences = {100*div_frac:.2f}% "
        f"-> {'HEALTHY' if verdict['health']['healthy'] else 'UNHEALTHY'}"
    )

    for label, f in pivots.items():
        l_inj = float(np.log10(powerlaw_psd(f, log10_A, gamma)))
        l_rec = np.log10(ou_residual_psd(f, log10_ha, log10_ga))
        med, std = float(np.median(l_rec)), float(np.std(l_rec))
        lo, hi = (float(q) for q in np.percentile(l_rec, [2.5, 97.5]))
        covered = bool(lo <= l_inj <= hi)
        bias_sigma = (med - l_inj) / std if std > 0 else float("nan")
        verdict["band_referenced"][label] = {
            "f_hz": f,
            "injected_log10_psd": l_inj,
            "recovered_log10_psd": {"median": med, "std": std, "ci95": [lo, hi]},
            "covered_95": covered,
            "bias_sigma": float(bias_sigma),
        }
        print(
            f"pivot f = {label}: injected log10 PSD {l_inj:+.3f} | recovered "
            f"{med:+.3f} +/- {std:.3f} (95% CI [{lo:+.3f}, {hi:+.3f}]) | "
            f"covered: {covered} | bias {bias_sigma:+.2f} sigma"
        )

    primary = verdict["band_referenced"]["1/(5yr)"]
    verdict["pass"] = bool(primary["covered_95"] and verdict["health"]["healthy"])
    print(
        f"VERDICT ({tag}): {'PASS' if verdict['pass'] else 'FAIL'} "
        f"(covered_95={primary['covered_95']}, healthy={verdict['health']['healthy']}, "
        f"bias {primary['bias_sigma']:+.2f} sigma)"
    )

    return verdict, (log10_ha, log10_ga)


def overlay_plot(draws_by_run, log10_A, gamma, out_png):
    """Recovered OU PSD bands for every run + the injected power-law."""
    f_grid = np.logspace(np.log10(F_YR / 20.0), np.log10(2.0 * F_YR), 200)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(
        f_grid,
        powerlaw_psd(f_grid, log10_A, gamma),
        color="k",
        lw=2,
        ls="--",
        label=f"injected power-law (log10A={log10_A:.3f}, $\\gamma$={gamma:.2f})",
    )
    for color, (tag, (log10_ha, log10_ga)) in zip(
        ("C0", "C1", "C2", "C3"), draws_by_run.items()
    ):
        idx = np.linspace(0, log10_ha.size - 1, min(400, log10_ha.size)).astype(int)
        grid = np.vstack(
            [ou_residual_psd(f_grid, log10_ha[i], log10_ga[i]) for i in idx]
        )
        ax.plot(
            f_grid, np.median(grid, axis=0), color=color, lw=2, label=f"{tag} (median)"
        )
        ax.fill_between(
            f_grid, *np.percentile(grid, [16, 84], axis=0), color=color, alpha=0.2
        )
    for label, f in (("1/(5yr)", 1.0 / (5.0 * SEC_PER_YEAR)), ("1/yr", F_YR)):
        ax.axvline(f, color="0.5", ls=":", lw=1)
        ax.text(
            f,
            1,
            label,
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=8,
            color="0.4",
            transform=ax.get_xaxis_transform(),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("residual PSD [s$^3$]")
    ax.set_title("M1 truth gate: MDC2 injected power-law vs recovered OU bands")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    """Parse command-line arguments and evaluate each run."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--run",
        action="append",
        dest="runs",
        required=True,
        help="run tag under outputs/ (repeatable), e.g. --run mdc2_stageB_hd "
        "--run mdc2_stageC_hd",
    )
    ap.add_argument("--log10-a", type=float, default=MDC2_LOG10_A)
    ap.add_argument("--gamma", type=float, default=MDC2_GAMMA)
    ap.add_argument("--results-dir", default=os.path.join(ROOT, "outputs"))
    ap.add_argument(
        "--out", default=os.path.join(ROOT, "outputs", "mdc2_truth_gate.json")
    )
    args = ap.parse_args()

    verdicts, draws_by_run = {}, {}
    for tag in args.runs:
        verdict, draws = check_run(tag, args.log10_a, args.gamma, args.results_dir)
        verdicts[tag] = verdict
        draws_by_run[tag] = draws

    out_png = os.path.splitext(args.out)[0] + ".png"
    overlay_plot(draws_by_run, args.log10_a, args.gamma, out_png)

    with open(args.out, "w") as f:
        json.dump(verdicts, f, indent=2)
    print("=" * 78)
    print(f"verdicts -> {args.out}")
    print(f"overlay  -> {out_png}")


if __name__ == "__main__":
    main()
