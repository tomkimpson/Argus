#!/usr/bin/env python
"""Generate publication-quality CW source parameter corner plots.

Usage:
    python publication_corner_plot.py <netcdf_file> [options]
    python publication_corner_plot.py <file1> <file2> --compare [options]

Examples:
    # Single run
    python publication_corner_plot.py outputs/cw_8chain_A/no_gw/cw_8chain_A_results.nc

    # Custom output path and title
    python publication_corner_plot.py outputs/cw_8chain_A/no_gw/cw_8chain_A_results.nc \
        -o outputs/my_corner.png \
        -t "My custom title"

    # Compare two runs (overlay)
    python publication_corner_plot.py \
        outputs/cw_phase_reparam_intensive/no_gw/cw_phase_reparam_intensive_results.nc \
        outputs/cw_8chain_A/no_gw/cw_8chain_A_results.nc \
        --compare --labels "Intensive" "8-chain A"
"""
import argparse
import os
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from corner import corner

# ============================================================
# IPTA MDC2 Dataset 3b injection values
# ============================================================

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
    (-16.0, -12.0),      # log10_h0
    (-9.0, -7.0),        # log10_f_gw
    (0.0, 2 * np.pi),    # alpha_gw
    (-np.pi/2, np.pi/2), # delta_gw
    (-1.0, 1.0),         # cos_iota
    (0.0, np.pi),        # psi
    (0.0, 2 * np.pi),    # Phi0
]

TRUTHS = [INJECTIONS[v] for v in CW_VARS]

CHAIN_COLORS = ["C0", "C2", "C1", "C3", "C4", "C5", "C6", "C7"]


# ============================================================
# Data loading
# ============================================================

def load_per_chain(nc_path):
    """Load CW source parameter samples preserving per-chain structure.

    Parameters
    ----------
    nc_path : str
        Path to ArviZ NetCDF file.

    Returns
    -------
    per_chain : list of ndarray
        Each element is (n_draws, 7) for one chain.
    combined : ndarray
        Shape (n_chains * n_draws, 7), all chains concatenated.
    n_chains : int
    n_draws : int
    """
    ds = xr.open_dataset(nc_path, group="posterior")
    n_chains = ds.dims["chain"]
    n_draws = ds.dims["draw"]
    per_chain = []
    for c in range(n_chains):
        chain_samples = np.column_stack([ds[var].values[c] for var in CW_VARS])
        per_chain.append(chain_samples)
    combined = np.vstack(per_chain)
    return per_chain, combined, n_chains, n_draws


# ============================================================
# Single-run publication corner plot
# ============================================================

def make_publication_corner(nc_path, title=None, output_path=None):
    """Generate a publication-quality corner plot for a single run.

    Parameters
    ----------
    nc_path : str
        Path to ArviZ NetCDF file.
    title : str, optional
        Plot title. Auto-generated from filename if None.
    output_path : str, optional
        Output path (without extension). Saves .png and .pdf.
        Defaults to same directory as nc_path.

    Returns
    -------
    str
        Path to saved PNG file.
    """
    per_chain, combined, n_chains, n_draws = load_per_chain(nc_path)
    n_dim = len(CW_VARS)

    if title is None:
        basename = os.path.splitext(os.path.basename(nc_path))[0].replace("_results", "")
        title = (
            f"CW Posterior: {basename}\n"
            f"({n_chains} chains $\\times$ {n_draws} samples)"
        )

    if output_path is None:
        nc_dir = os.path.dirname(nc_path)
        basename = os.path.splitext(os.path.basename(nc_path))[0].replace("_results", "")
        output_path = os.path.join(nc_dir, f"corner_{basename}_publication")

    # Main corner: filled contours on combined data
    fig = corner(
        combined,
        labels=LABELS,
        truths=TRUTHS,
        truth_color="red",
        show_titles=False,
        label_kwargs={"fontsize": 11},
        quantiles=[0.16, 0.5, 0.84],
        levels=(0.68, 0.95),
        plot_density=True,
        plot_contours=True,
        fill_contours=True,
        color="C0",
        range=PRIOR_RANGES,
        hist_kwargs={"density": True, "alpha": 0.0},
    )
    axes = np.array(fig.axes).reshape((n_dim, n_dim))

    # Per-chain histograms on diagonal
    for i in range(n_dim):
        ax = axes[i, i]
        ax.clear()
        lo, hi = PRIOR_RANGES[i]
        bins = np.linspace(lo, hi, 50)
        for c in range(n_chains):
            ax.hist(
                per_chain[c][:, i],
                bins=bins,
                density=True,
                alpha=0.4,
                color=CHAIN_COLORS[c % len(CHAIN_COLORS)],
                label=f"Chain {c}" if i == 0 else None,
                histtype="stepfilled",
                linewidth=0.5,
                edgecolor=CHAIN_COLORS[c % len(CHAIN_COLORS)],
            )
            median_c = np.median(per_chain[c][:, i])
            ax.axvline(median_c, color=CHAIN_COLORS[c % len(CHAIN_COLORS)],
                       linewidth=1.0, linestyle="--", alpha=0.7)

        ax.axvline(TRUTHS[i], color="red", linewidth=1.5, linestyle="--",
                   label="Injection" if i == 0 else None)

        q16, q50, q84 = np.percentile(combined[:, i], [16, 50, 84])
        ax.set_title(
            f"{LABELS[i]} = {q50:.3f}$^{{+{q84-q50:.3f}}}_{{-{q50-q16:.3f}}}$",
            fontsize=9,
        )
        ax.set_xlim(lo, hi)
        ax.set_yticks([])

    axes[0, 0].legend(fontsize=6.5, loc="upper right", framealpha=0.9, handlelength=1.0)

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.savefig(output_path + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(output_path + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}.png")
    return output_path + ".png"


# ============================================================
# Two-run comparison overlay
# ============================================================

def make_comparison_corner(nc_path_1, nc_path_2, label_1="Run 1", label_2="Run 2",
                           title=None, output_path=None):
    """Generate an overlay corner plot comparing two runs.

    Parameters
    ----------
    nc_path_1, nc_path_2 : str
        Paths to ArviZ NetCDF files.
    label_1, label_2 : str
        Legend labels for each run.
    title : str, optional
    output_path : str, optional

    Returns
    -------
    str
        Path to saved PNG file.
    """
    _, combined_1, nc1, nd1 = load_per_chain(nc_path_1)
    _, combined_2, nc2, nd2 = load_per_chain(nc_path_2)

    if title is None:
        title = f"Blue: {label_1} | Orange: {label_2}"

    if output_path is None:
        output_path = "outputs/corner_comparison_publication"

    fig = corner(
        combined_1,
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
        combined_2,
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
    return output_path + ".png"


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality CW source corner plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("files", nargs="+", help="NetCDF file(s) to plot")
    parser.add_argument("-o", "--output", default=None, help="Output path (no extension)")
    parser.add_argument("-t", "--title", default=None, help="Plot title")
    parser.add_argument("--compare", action="store_true",
                        help="Overlay two runs (requires exactly 2 files)")
    parser.add_argument("--labels", nargs=2, default=None,
                        help="Labels for comparison (e.g. --labels 'Run A' 'Run B')")

    args = parser.parse_args()

    if args.compare:
        if len(args.files) != 2:
            parser.error("--compare requires exactly 2 files")
        labels = args.labels or ["Run 1", "Run 2"]
        make_comparison_corner(
            args.files[0], args.files[1],
            label_1=labels[0], label_2=labels[1],
            title=args.title, output_path=args.output,
        )
    else:
        for f in args.files:
            make_publication_corner(f, title=args.title, output_path=args.output)


if __name__ == "__main__":
    main()
