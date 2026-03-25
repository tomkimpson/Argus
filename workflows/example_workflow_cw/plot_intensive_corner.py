"""Publication-quality corner plot for the fixed-f_gw + pulsar term intensive run."""

import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import matplotlib
from corner import corner

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "text.usetex": False,
})

# Load results
results_path = "outputs/cw_fixed_fgw_pt_intensive/no_gw/cw_fixed_fgw_pt_intensive_results.nc"
inf_data = az.from_netcdf(results_path)

# CW source parameters (f_gw is fixed, not shown)
param_names = ["log10_h0", "alpha_gw", "delta_gw", "cos_iota", "psi", "Phi0"]
labels = [
    r"$\log_{10} h_0$",
    r"$\alpha_{\rm gw}$ [rad]",
    r"$\delta_{\rm gw}$ [rad]",
    r"$\cos \iota$",
    r"$\psi$ [rad]",
    r"$\Phi_0$ [rad]",
]

# Injection values
injections = {
    "log10_h0": -13.350,
    "alpha_gw": 4.067,
    "delta_gw": 0.140,
    "cos_iota": 0.907,
    "psi": 0.646,
    "Phi0": 0.175,
}

# Extract samples from all chains, flatten
samples_list = []
for p in param_names:
    vals = inf_data.posterior[p].values  # (chains, draws)
    samples_list.append(vals.flatten())

samples = np.column_stack(samples_list)
truths = [injections[p] for p in param_names]

# Per-chain colors for overplotting
chain_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
n_chains = inf_data.posterior[param_names[0]].shape[0]

# Prior ranges (from config)
prior_ranges = [
    (-16.0, -12.0),      # log10_h0
    (0.0, 6.283185),     # alpha_gw
    (-np.pi/2, np.pi/2), # delta_gw (from sin_delta_gw in [-1,1])
    (-1.0, 1.0),         # cos_iota
    (0.0, 3.14159),      # psi
    (0.0, 6.283185),     # Phi0
]

# Create corner plot
fig = corner(
    samples,
    labels=labels,
    truths=truths,
    truth_color="#e41a1c",
    range=prior_ranges,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 11},
    title_fmt=".3f",
    color="#2166ac",
    hist_kwargs={"density": True, "alpha": 0.7, "linewidth": 1.5},
    plot_datapoints=False,
    fill_contours=True,
    levels=(0.393, 0.865, 0.989),  # 1, 1.5, 2.5 sigma
    smooth=1.2,
    label_kwargs={"fontsize": 13},
)

# Overlay per-chain 1D histograms on the diagonal
axes = np.array(fig.axes).reshape(len(param_names), len(param_names))
for i, p in enumerate(param_names):
    ax = axes[i, i]
    for c in range(n_chains):
        chain_vals = inf_data.posterior[p].values[c, :]
        ax.hist(
            chain_vals,
            bins=40,
            density=True,
            alpha=0.3,
            color=chain_colors[c],
            histtype="stepfilled",
            linewidth=0.8,
        )

# Add legend for chains
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=chain_colors[c], alpha=0.4, label=f"Chain {c}")
                   for c in range(n_chains)]
legend_elements.append(plt.Line2D([0], [0], color="#e41a1c", lw=2, label="Injection"))
axes[0, -1].legend(handles=legend_elements, loc="center", frameon=False, fontsize=10)

# Title
fig.suptitle(
    r"CW Posterior: Fixed $f_{\rm gw}$ + Phase-Reparam Pulsar Term (4 chains $\times$ 2000 samples)",
    fontsize=14,
    y=1.02,
)

# Save
outdir = "outputs/cw_fixed_fgw_pt_intensive/no_gw/plots"
fig.savefig(f"{outdir}/corner_cw_fixed_fgw_pt_intensive_publication.png", dpi=300)
fig.savefig(f"{outdir}/corner_cw_fixed_fgw_pt_intensive_publication.pdf")
print(f"Saved to {outdir}/corner_cw_fixed_fgw_pt_intensive_publication.png")
print(f"Saved to {outdir}/corner_cw_fixed_fgw_pt_intensive_publication.pdf")
