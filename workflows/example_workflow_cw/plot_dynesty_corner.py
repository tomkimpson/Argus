"""Publication-quality corner plot for dynesty nested sampling CW results."""

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

# --- Load results ---
results_path = "outputs/cw_dynesty_light/no_gw/cw_dynesty_light_results.nc"
inf_data = az.from_netcdf(results_path)

# --- Injection values (MDC2 dataset 3b) ---
injections = {
    "log10_h0": -13.350,
    "alpha_gw": 4.067,
    "delta_gw": 0.140,
    "log10_f_gw": np.log10(6.1e-9),
    "cos_iota": np.cos(0.4363),
    "psi": 0.646,
    "Phi0": 0.175,
}

# --- Parameters to plot ---
param_names = ["log10_h0", "alpha_gw", "delta_gw", "log10_f_gw",
               "cos_iota", "psi", "Phi0"]
labels = [
    r"$\log_{10} h_0$",
    r"$\alpha_{\rm gw}$ [rad]",
    r"$\delta_{\rm gw}$ [rad]",
    r"$\log_{10} f_{\rm gw}$ [Hz]",
    r"$\cos \iota$",
    r"$\psi$ [rad]",
    r"$\Phi_0$ [rad]",
]

# --- Prior ranges ---
prior_ranges = [
    (-16.0, -12.0),
    (0.0, 2 * np.pi),
    (-np.pi / 2, np.pi / 2),
    (-9.0, -7.0),
    (-1.0, 1.0),
    (0.0, np.pi),
    (0.0, 2 * np.pi),
]

# --- Extract samples ---
samples = np.column_stack([
    inf_data.posterior[p].values.flatten() for p in param_names
])
truths = [injections[p] for p in param_names]

# --- Corner plot ---
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
    plot_datapoints=False,
    fill_contours=True,
    levels=(0.393, 0.865, 0.989),  # 1, 1.5, 2.5 sigma
    smooth=1.2,
    label_kwargs={"fontsize": 13},
)

# --- Legend ---
from matplotlib.patches import Patch
axes = np.array(fig.axes).reshape(len(param_names), len(param_names))
legend_elements = [
    Patch(facecolor="#2166ac", alpha=0.4, label="Dynesty posterior"),
    plt.Line2D([0], [0], color="#e41a1c", lw=2, label="Injection"),
]
axes[0, -1].legend(handles=legend_elements, loc="center", frameon=False, fontsize=10)

# --- Title ---
fig.suptitle(
    r"Dynesty nested sampling  |  $n_{\rm live} = 100$,  $\Delta \log \mathcal{Z}_{\rm tol} = 1.0$"
    "\n"
    r"$\log \mathcal{Z} = 63352.0 \pm 1.8$  |  5000 posterior samples  |  IPTA MDC2 3b (31 pulsars)",
    fontsize=14,
    y=1.02,
)

# --- Save ---
outdir = "outputs/cw_dynesty_light/no_gw/plots"
fig.savefig(f"{outdir}/corner_cw_dynesty_publication.png", dpi=300)
fig.savefig(f"{outdir}/corner_cw_dynesty_publication.pdf")
plt.close(fig)
print(f"Saved: {outdir}/corner_cw_dynesty_publication.png")
print(f"Saved: {outdir}/corner_cw_dynesty_publication.pdf")
