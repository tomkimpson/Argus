#!/usr/bin/env python
"""Compare Argus and ENTERPRISE CW search results.

Produces overlay corner plots and per-parameter summary statistics.
Reuses the publication_corner_plot infrastructure.

Usage:
    python compare_argus_enterprise.py <argus_nc> <enterprise_nc>
    python compare_argus_enterprise.py <argus_nc> <enterprise_nc> -o outputs/comparison
"""
import argparse
import os

import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from corner import corner

# Shared constants (same as publication_corner_plot.py)
INJECTIONS = {
    "log10_h0": -13.350,
    "log10_f_gw": -8.215,
    "alpha_gw": 4.067,
    "delta_gw": 0.140,
    "cos_iota": 0.907,
    "psi": 0.646,
    "Phi0": 0.175,
}

CW_VARS = ["log10_h0", "log10_f_gw", "alpha_gw", "delta_gw", "cos_iota", "psi", "Phi0"]

LABELS = [
    r"$\log_{10}\,h_0$",
    r"$\log_{10}\,f_{\rm gw}$",
    r"$\alpha_{\rm gw}$ [rad]",
    r"$\delta_{\rm gw}$ [rad]",
    r"$\cos\,\iota$",
    r"$\psi$ [rad]",
    r"$\Phi_0$ [rad]",
]

PRIOR_RANGES = [
    (-16.0, -12.0),
    (-9.0, -7.0),
    (0.0, 2 * np.pi),
    (-np.pi / 2, np.pi / 2),
    (-1.0, 1.0),
    (0.0, np.pi),
    (0.0, 2 * np.pi),
]

TRUTHS = [INJECTIONS[v] for v in CW_VARS]


def load_samples(nc_path):
    """Load CW samples from ArviZ NetCDF, return (n_total, 7) array."""
    ds = xr.open_dataset(nc_path, group="posterior")
    n_chains = ds.dims["chain"]
    n_draws = ds.dims["draw"]
    samples = np.column_stack([ds[var].values.flatten() for var in CW_VARS])
    return samples, n_chains, n_draws


def summary_table(argus_samples, enterprise_samples):
    """Print a per-parameter summary comparing both methods."""
    header = f"{'Parameter':15s} | {'Argus median':>14s} {'68% CI':>14s} | {'Enterprise median':>18s} {'68% CI':>14s} | {'Injection':>10s}"
    print(header)
    print("-" * len(header))
    for i, var in enumerate(CW_VARS):
        inj = INJECTIONS[var]
        # Argus
        a16, a50, a84 = np.percentile(argus_samples[:, i], [16, 50, 84])
        # Enterprise
        e16, e50, e84 = np.percentile(enterprise_samples[:, i], [16, 50, 84])
        print(
            f"{var:15s} | {a50:14.4f} [{a16:.4f}, {a84:.4f}] | "
            f"{e50:18.4f} [{e16:.4f}, {e84:.4f}] | {inj:10.4f}"
        )


def make_overlay_corner(argus_samples, enterprise_samples, output_path, title=None):
    """Generate overlay corner plot: Argus (blue) vs ENTERPRISE (orange)."""
    if title is None:
        title = "CW Posterior: Argus (blue) vs ENTERPRISE (orange)"

    fig = corner(
        argus_samples,
        labels=LABELS,
        truths=TRUTHS,
        truth_color="red",
        levels=(0.68, 0.95),
        plot_density=True,
        plot_contours=True,
        fill_contours=True,
        color="C0",
        range=PRIOR_RANGES,
        hist_kwargs={"density": True, "alpha": 0.4},
        label_kwargs={"fontsize": 11},
    )
    corner(
        enterprise_samples,
        fig=fig,
        levels=(0.68, 0.95),
        plot_density=True,
        plot_contours=True,
        fill_contours=True,
        color="C1",
        range=PRIOR_RANGES,
        hist_kwargs={"density": True, "alpha": 0.4},
    )
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.savefig(output_path + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(output_path + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}.png")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Argus and ENTERPRISE CW search results."
    )
    parser.add_argument("argus_nc", help="Argus ArviZ NetCDF file")
    parser.add_argument("enterprise_nc", help="ENTERPRISE ArviZ NetCDF file")
    parser.add_argument("-o", "--output", default=None,
                        help="Output path (no extension)")
    parser.add_argument("-t", "--title", default=None, help="Plot title")
    args = parser.parse_args()

    if args.output is None:
        args.output = "outputs/argus_vs_enterprise_comparison/corner_comparison"

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Argus:      {args.argus_nc}")
    print(f"Enterprise: {args.enterprise_nc}")

    argus_samples, a_chains, a_draws = load_samples(args.argus_nc)
    enterprise_samples, e_chains, e_draws = load_samples(args.enterprise_nc)

    print(f"\nArgus:      {a_chains} chains x {a_draws} draws = {len(argus_samples)} samples")
    print(f"Enterprise: {e_chains} chains x {e_draws} draws = {len(enterprise_samples)} samples")

    print("\n=== Per-Parameter Comparison ===")
    summary_table(argus_samples, enterprise_samples)

    make_overlay_corner(argus_samples, enterprise_samples, args.output, title=args.title)


if __name__ == "__main__":
    main()
