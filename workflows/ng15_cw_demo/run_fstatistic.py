"""All-sky coherent Earth-term F-statistic (F_e) + B-statistic CW detection scan.

Runs a **sampling-free** continuous-wave detection scan on the prepared NG15
wideband demo data (`demo_data_wb/`), over a frequency x sky grid, using
`argus.cw_fstatistic`. Reports the peak detection statistic, its sky/frequency
location, and significance (analytic chi^2_4 + trials, plus an empirical null from a
no-injection dataset if provided). Writes a frequency profile, an all-sky map at the
peak frequency, and a summary JSON.

This is the detection-oriented counterpart to the (multimodal, hard-to-sample) NUTS
run: F_e analytically maximises over the extrinsic amplitudes, so it needs no MCMC
and is robust to the non-convergence seen with NUTS.

Usage
-----
    python run_fstatistic.py --data-dir ./demo_data_wb \
        [--null-dir ./demo_data_wb_noinj] [--nfreq 40 --nra 24 --ndec 12]
"""

import argparse
import configparser
import json
import os
import sys

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pandas as pd
from scipy.stats import chi2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python"))

from argus import io_manager
from argus.data_loader import LoadWidebandPulsarData
from argus.cw_kalman_filter import CWKalmanFilter
from argus.cw_fstatistic import scan_grid


def _init_logger():
    cfg = configparser.ConfigParser()
    cfg.add_section("Logging")
    cfg.set("Logging", "level", "WARNING")
    cfg.set("Logging", "enable_file_logging", "false")
    io_manager.setup_single_logger(cfg, enable_file_logging=False)


def load_fixed_noise(data_dir, names):
    """Load the fixed per-pulsar noise arrays (in metadata/pulsar order `names`)."""
    nd = json.load(open(os.path.join(data_dir, "noise_params.json")))
    spin = pd.read_pickle(os.path.join(data_dir, "spin_injections.pkl")).set_index("psr")
    efac = jnp.array([nd[n]["efac"] for n in names])
    equad = jnp.array([10.0 ** nd[n]["equad"] for n in names])
    sigma_p = jnp.array([spin.loc[n, "optimal_sigma"] for n in names])
    gamma_p = jnp.array([spin.loc[n, "optimal_gamma"] for n in names])
    return gamma_p, sigma_p, efac, equad


def run_scan(data_dir, f_grid, ra_grid, sindec_grid, sigma_a):
    data = LoadWidebandPulsarData.get_processed_residuals(data_dir, mode="cw")
    names = list(data["metadata"]["name"])
    kf = CWKalmanFilter(data, include_pulsar_term=False, phase_parameterization=True)
    gamma_p, sigma_p, efac, equad = load_fixed_noise(data_dir, names)
    res = scan_grid(kf, gamma_p, sigma_p, efac, equad, f_grid, ra_grid, sindec_grid,
                    sigma_a=sigma_a)
    return res, names


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "demo_data_wb"))
    ap.add_argument("--null-dir", default=None, help="No-injection dataset for the empirical null.")
    ap.add_argument("--nfreq", type=int, default=40)
    ap.add_argument("--nra", type=int, default=24)
    ap.add_argument("--ndec", type=int, default=12)
    ap.add_argument("--fmin", type=float, default=5e-9)
    ap.add_argument("--fmax", type=float, default=5e-8)
    ap.add_argument("--sigma-a", type=float, default=1e-6, help="B-statistic amplitude prior width (s).")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "outputs", "fstatistic"))
    args = ap.parse_args()

    _init_logger()
    os.makedirs(args.out_dir, exist_ok=True)

    f_grid = np.geomspace(args.fmin, args.fmax, args.nfreq)
    ra_grid = np.linspace(0, 2 * np.pi, args.nra, endpoint=False)
    sindec_grid = np.linspace(-0.99, 0.99, args.ndec)

    print(f"F-statistic scan: {args.nfreq} freqs x {args.nra}x{args.ndec} sky = "
          f"{args.nfreq * args.nra * args.ndec} grid points")
    res, names = run_scan(args.data_dir, f_grid, ra_grid, sindec_grid, args.sigma_a)
    twoF = res["twoF"]           # (Nf, Nra, Ndec)
    lnB = res["lnB"]
    dec_grid = res["dec_grid"]

    # Peak.
    pk = np.unravel_index(np.nanargmax(twoF), twoF.shape)
    peak_2F = float(twoF[pk])
    peak_f = float(f_grid[pk[0]])
    peak_ra = float(ra_grid[pk[1]])
    peak_dec = float(dec_grid[pk[2]])
    peak_lnB = float(lnB[pk])

    # Significance: 2F_e ~ chi^2_4 under H0 (single point) + Bonferroni trials.
    n_trials = twoF.size
    p_single = float(chi2.sf(peak_2F, df=4))
    p_trials = float(min(1.0, p_single * n_trials))

    summary = {
        "pulsars": names,
        "grid": {"nfreq": args.nfreq, "nra": args.nra, "ndec": args.ndec,
                 "fmin": args.fmin, "fmax": args.fmax},
        "peak": {"twoF": peak_2F, "SNR": float(np.sqrt(max(peak_2F, 0.0))),
                 "lnB": peak_lnB, "f_gw": peak_f, "ra": peak_ra, "dec": peak_dec},
        "significance": {"p_single_point_chi2_4": p_single,
                         "p_trials_corrected": p_trials, "n_trials": n_trials},
        "sigma_a": args.sigma_a,
    }

    # Injection truth for comparison, if present.
    inj_path = os.path.join(args.data_dir, "injection.json")
    if os.path.exists(inj_path):
        inj = json.load(open(inj_path))
        summary["injection"] = inj
        summary["peak"]["f_gw_over_injected"] = peak_f / inj["f_gw"]

    # Empirical null from a no-injection dataset.
    if args.null_dir and os.path.isdir(args.null_dir):
        print(f"Empirical null from {args.null_dir} ...")
        res0, _ = run_scan(args.null_dir, f_grid, ra_grid, sindec_grid, args.sigma_a)
        null_2F = res0["twoF"]
        null_max = float(np.nanmax(null_2F))
        p_emp = float((np.sum(null_2F >= peak_2F) + 1) / (null_2F.size + 1))
        summary["significance"]["empirical_null_max_2F"] = null_max
        summary["significance"]["p_empirical_vs_null_grid"] = p_emp
        summary["significance"]["peak_exceeds_null_max"] = bool(peak_2F > null_max)

    with open(os.path.join(args.out_dir, "fstat_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Plot 1: 2F_e profile vs frequency (max over sky at each freq).
    prof = np.nanmax(twoF.reshape(args.nfreq, -1), axis=1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(f_grid, prof, "-o", ms=3)
    ax.axhline(chi2.ppf(1 - 1e-3, df=4), ls="--", c="grey", label=r"$\chi^2_4$ 99.9%")
    if "injection" in summary:
        ax.axvline(summary["injection"]["f_gw"], c="r", ls=":", label="injected $f_{gw}$")
    ax.set_xlabel("GW frequency [Hz]"); ax.set_ylabel(r"$\max_{sky}\ 2\mathcal{F}_e$")
    ax.set_title("Earth-term F-statistic profile"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "fstat_profile.png"), dpi=130)
    plt.close(fig)

    # Plot 2: all-sky 2F_e map at the peak frequency.
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(twoF[pk[0]].T, origin="lower", aspect="auto",
                   extent=[0, 2 * np.pi, dec_grid.min(), dec_grid.max()], cmap="viridis")
    ax.plot(peak_ra, peak_dec, "w*", ms=14, label="peak")
    if "injection" in summary:
        ax.plot(inj["alpha_gw"], inj["delta_gw"], "r+", ms=14, mew=2, label="injected")
    ax.set_xlabel("RA [rad]"); ax.set_ylabel("Dec [rad]")
    ax.set_title(f"$2\\mathcal{{F}}_e$ sky map @ f={peak_f:.2e} Hz"); ax.legend()
    fig.colorbar(im, ax=ax, label=r"$2\mathcal{F}_e$")
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "fstat_skymap.png"), dpi=130)
    plt.close(fig)

    # Report.
    print("\n=== F-statistic detection summary ===")
    print(f"  peak 2F_e = {peak_2F:.1f}  (SNR ~ {np.sqrt(max(peak_2F,0)):.1f}),  lnB = {peak_lnB:.1f}")
    print(f"  peak (f_gw, RA, Dec) = ({peak_f:.3e} Hz, {peak_ra:.2f}, {peak_dec:.2f})")
    if "injection" in summary:
        print(f"  injected (f_gw, RA, Dec) = ({inj['f_gw']:.3e}, {inj['alpha_gw']:.2f}, {inj['delta_gw']:.2f})")
    print(f"  p (single-point chi2_4) = {p_single:.2e};  p (x{n_trials} trials) = {p_trials:.2e}")
    if "p_empirical_vs_null_grid" in summary["significance"]:
        print(f"  empirical null max 2F_e = {summary['significance']['empirical_null_max_2F']:.1f}; "
              f"peak exceeds null max: {summary['significance']['peak_exceeds_null_max']}")
    print(f"  outputs -> {args.out_dir}")


if __name__ == "__main__":
    main()
