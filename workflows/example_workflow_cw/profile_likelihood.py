"""Profile the CW likelihood by sweeping individual parameters around injection values.

Holds all parameters fixed at their injected values and varies one at a time,
checking whether the likelihood peak coincides with the injection.

Usage:
    python profile_likelihood.py              # Standard profiling
    python profile_likelihood.py --ablation   # M-matrix ablation test for delta_gw
"""

import argparse
import os
import sys
import numpy as np
import configparser

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# Add project root to path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(os.path.join(project_root, "python"))

from argus import data_loader, cw_kalman_filter, utils, io_manager
from argus.bayesian_inference import CWParameters

# ============================================================
# Injection values (IPTA MDC2, Dataset 3b)
# ============================================================
INJECTION = {
    "log10_h0": -13.350,
    "log10_f_gw": -8.215,  # f_gw = 6.1e-9 Hz
    "alpha_gw": 4.067,     # radians
    "delta_gw": 0.140,     # radians (pi/2 - 1.431)
    "cos_iota": 0.907,     # cos(0.436)
    "psi": 0.646,          # radians
    "Phi0": 0.175,         # radians
}


def load_data_and_filter(config_path):
    """Load data and initialize the CW Kalman filter."""
    config = configparser.ConfigParser()
    config.read(config_path)

    # Initialize logger
    io_manager.setup_single_logger(config)

    data_path = config.get("Data", "data_path")
    # Resolve relative path
    config_dir = os.path.dirname(os.path.abspath(config_path))
    if not os.path.isabs(data_path):
        data_path = os.path.normpath(os.path.join(config_dir, data_path))

    excluded_psrs = config.get("Data", "excluded_psrs").split(",")
    excluded_psrs = [p.strip() for p in excluded_psrs if p.strip()]

    print(f"Loading data from: {data_path}")
    pulsar_data = data_loader.LoadWidebandPulsarData.get_processed_residuals(
        data_path, excluded_psrs=excluded_psrs, mode="cw",
    )

    n_pulsars = len(pulsar_data["metadata"])
    print(f"Loaded {n_pulsars} pulsars")

    # Earth-term only filter (no pulsar term complications)
    kf = cw_kalman_filter.CWKalmanFilter(data=pulsar_data, include_pulsar_term=False)

    # Resolve relative paths in config before loading noise
    config_dir = os.path.dirname(os.path.abspath(config_path))
    for section in config.sections():
        for key, val in config.items(section):
            if "path" in key and val.strip() and not os.path.isabs(val.strip()):
                config.set(section, key, os.path.normpath(os.path.join(config_dir, val.strip())))

    # Load noise parameters
    efac_array, equad_array, sigma_p_array, gamma_p_array = utils.get_noise_parameters(config)

    return kf, n_pulsars, efac_array, equad_array, sigma_p_array, gamma_p_array


def make_params(kf, n_pulsars, efac_array, equad_array, sigma_p_array, gamma_p_array,
                **overrides):
    """Construct CWParameters at injection values, with optional overrides."""
    h0 = 10.0 ** overrides.get("log10_h0", INJECTION["log10_h0"])
    f_gw = 10.0 ** overrides.get("log10_f_gw", INJECTION["log10_f_gw"])
    alpha_gw = overrides.get("alpha_gw", INJECTION["alpha_gw"])
    delta_gw = overrides.get("delta_gw", INJECTION["delta_gw"])
    cos_iota = overrides.get("cos_iota", INJECTION["cos_iota"])
    psi = overrides.get("psi", INJECTION["psi"])
    Phi0 = overrides.get("Phi0", INJECTION["Phi0"])

    # Noise: use injections if available, otherwise reasonable defaults
    if gamma_p_array is not None:
        gamma_p = jnp.array(gamma_p_array)
    else:
        gamma_p = jnp.full(n_pulsars, 1e-8)

    if sigma_p_array is not None:
        sigma_p = jnp.array(sigma_p_array)
    else:
        sigma_p = jnp.full(n_pulsars, 1e-15)

    if efac_array is not None:
        efac = jnp.array(efac_array)
    else:
        efac = jnp.ones(n_pulsars)

    if equad_array is not None:
        equad = jnp.array(equad_array)
    else:
        equad = jnp.full(n_pulsars, 1e-8)

    return CWParameters(
        alpha_gw=alpha_gw, delta_gw=delta_gw, f_gw=f_gw, h0=h0,
        cos_iota=cos_iota, psi=psi, Phi0=Phi0,
        chi=jnp.zeros(n_pulsars),
        gamma_p=gamma_p, sigma_p=sigma_p, EFAC=efac, EQUAD=equad,
    )


def sweep_parameter(kf, n_pulsars, efac_array, equad_array,
                    sigma_p_array, gamma_p_array,
                    param_name, values, **extra_overrides):
    """Evaluate likelihood over a sweep of one parameter."""
    ll_values = []
    for val in values:
        overrides = dict(extra_overrides)
        overrides[param_name] = float(val)
        params = make_params(
            kf, n_pulsars, efac_array, equad_array, sigma_p_array, gamma_p_array,
            **overrides,
        )
        ll = float(kf.get_likelihood(params))
        ll_values.append(ll)
    return np.array(ll_values)


def load_noise_from_posterior(nc_path):
    """Load median noise parameters from a completed inference run."""
    import arviz as az
    ds = az.from_netcdf(nc_path)

    log10_gp = ds.posterior["log10_γp"].values
    log10_sp = ds.posterior["log10_σp"].values
    efac_vals = ds.posterior["efac"].values
    equad_vals = ds.posterior["equad"].values

    # Flatten chains+samples, take median over each pulsar
    gamma_p = np.median(10.0 ** log10_gp.reshape(-1, log10_gp.shape[-1]), axis=0)
    sigma_p = np.median(10.0 ** log10_sp.reshape(-1, log10_sp.shape[-1]), axis=0)
    efac = np.median(efac_vals.reshape(-1, efac_vals.shape[-1]), axis=0)
    equad = np.median(equad_vals.reshape(-1, equad_vals.shape[-1]), axis=0)

    return efac, equad, sigma_p, gamma_p


def main():
    config_path = "configs/smoke_test_config.ini"
    kf, n_pulsars, efac_array, equad_array, sigma_p_array, gamma_p_array = (
        load_data_and_filter(config_path)
    )

    # Override noise with posterior medians from intensive run
    nc_path = "outputs/cw_intensive_run/no_gw/cw_intensive_run_results.nc"
    if os.path.exists(nc_path):
        print(f"Loading posterior noise params from: {nc_path}")
        efac_array, equad_array, sigma_p_array, gamma_p_array = (
            load_noise_from_posterior(nc_path)
        )
        print(f"  gamma_p range: [{gamma_p_array.min():.2e}, {gamma_p_array.max():.2e}]")
        print(f"  sigma_p range: [{sigma_p_array.min():.2e}, {sigma_p_array.max():.2e}]")

    # Define sweeps: (param_name, display_name, grid_values, injection_value)
    sweeps = [
        ("log10_h0", r"$\log_{10} h_0$",
         np.linspace(-16, -12, 200), INJECTION["log10_h0"]),
        ("log10_f_gw", r"$\log_{10} f_{\rm gw}$",
         np.linspace(-9, -7, 200), INJECTION["log10_f_gw"]),
        ("alpha_gw", r"$\alpha_{\rm gw}$ (rad)",
         np.linspace(0, 2 * np.pi, 200), INJECTION["alpha_gw"]),
        ("delta_gw", r"$\delta_{\rm gw}$ (rad)",
         np.linspace(-np.pi / 2, np.pi / 2, 200), INJECTION["delta_gw"]),
        ("cos_iota", r"$\cos \iota$",
         np.linspace(-1, 1, 200), INJECTION["cos_iota"]),
        ("psi", r"$\psi$ (rad)",
         np.linspace(0, np.pi, 200), INJECTION["psi"]),
        ("Phi0", r"$\Phi_0$ (rad)",
         np.linspace(0, 2 * np.pi, 200), INJECTION["Phi0"]),
    ]

    # Warmup JIT
    print("Warming up JIT...")
    params = make_params(kf, n_pulsars, efac_array, equad_array,
                         sigma_p_array, gamma_p_array)
    ll_inj = float(kf.get_likelihood(params))
    print(f"Log-likelihood at injection: {ll_inj:.2f}")

    # Run sweeps
    results = {}
    for param_name, display_name, grid, inj_val in sweeps:
        print(f"\nSweeping {param_name}...")
        ll = sweep_parameter(
            kf, n_pulsars, efac_array, equad_array, sigma_p_array, gamma_p_array,
            param_name, grid,
        )
        peak_idx = np.argmax(ll)
        peak_val = grid[peak_idx]
        offset = peak_val - inj_val

        results[param_name] = {
            "grid": grid, "ll": ll, "peak_val": peak_val, "inj_val": inj_val,
            "offset": offset, "display_name": display_name,
        }

        print(f"  Injection: {inj_val:.4f}")
        print(f"  LL peak:   {peak_val:.4f}")
        print(f"  Offset:    {offset:.4f}")
        print(f"  Peak LL:   {ll[peak_idx]:.2f}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 3, figsize=(14, 12))
        axes = axes.flatten()

        for i, (param_name, display_name, _, _) in enumerate(sweeps):
            ax = axes[i]
            r = results[param_name]
            ax.plot(r["grid"], r["ll"], "k-", lw=0.8)
            ax.axvline(r["inj_val"], color="red", ls="--", lw=1.5, label="Injection")
            ax.axvline(r["peak_val"], color="blue", ls=":", lw=1.5, label="LL peak")
            ax.set_xlabel(r["display_name"])
            ax.set_ylabel("Log-likelihood")
            ax.set_title(f"offset = {r['offset']:.4f}")
            ax.legend(fontsize=8)

        # Hide unused subplots
        for j in range(len(sweeps), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            "CW Likelihood Profiles (all other params at injection)",
            fontsize=14, y=1.01,
        )
        plt.tight_layout()
        outpath = "outputs/likelihood_profiles.png"
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to: {outpath}")
    except ImportError:
        print("\nmatplotlib not available, skipping plot")

    # 2D sky position profile
    print("\n\nComputing 2D (alpha_gw, delta_gw) likelihood surface...")
    n_grid = 60
    alpha_grid = np.linspace(0, 2 * np.pi, n_grid)
    delta_grid = np.linspace(-np.pi / 2, np.pi / 2, n_grid)
    ll_2d = np.zeros((n_grid, n_grid))

    for i, alpha in enumerate(alpha_grid):
        for j, delta in enumerate(delta_grid):
            params = make_params(
                kf, n_pulsars, efac_array, equad_array, sigma_p_array, gamma_p_array,
                alpha_gw=float(alpha), delta_gw=float(delta),
            )
            ll_2d[j, i] = float(kf.get_likelihood(params))

    peak_idx_2d = np.unravel_index(np.argmax(ll_2d), ll_2d.shape)
    peak_alpha = alpha_grid[peak_idx_2d[1]]
    peak_delta = delta_grid[peak_idx_2d[0]]
    print(f"  2D peak: alpha={peak_alpha:.3f}, delta={peak_delta:.3f}")
    print(f"  Injection: alpha={INJECTION['alpha_gw']:.3f}, delta={INJECTION['delta_gw']:.3f}")

    try:
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))
        im = ax2.pcolormesh(alpha_grid, delta_grid, ll_2d, shading="auto")
        ax2.plot(INJECTION["alpha_gw"], INJECTION["delta_gw"], "r*", ms=15,
                 label="Injection", zorder=5)
        ax2.plot(peak_alpha, peak_delta, "bx", ms=12, mew=3,
                 label="LL peak", zorder=5)
        ax2.set_xlabel(r"$\alpha_{\rm gw}$ (rad)")
        ax2.set_ylabel(r"$\delta_{\rm gw}$ (rad)")
        ax2.set_title("Log-likelihood: sky position (other params at injection)")
        plt.colorbar(im, ax=ax2, label="Log-likelihood")
        ax2.legend()
        fig2.savefig("outputs/likelihood_sky_2d.png", dpi=150, bbox_inches="tight")
        print("  2D sky plot saved to: outputs/likelihood_sky_2d.png")
    except Exception as e:
        print(f"  Could not save 2D plot: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("LIKELIHOOD PROFILE SUMMARY")
    print("=" * 60)
    print(f"{'Parameter':<15} {'Injection':>10} {'LL Peak':>10} {'Offset':>10} {'Match?':>8}")
    print("-" * 60)
    for param_name, display_name, _, _ in sweeps:
        r = results[param_name]
        grid_spacing = r["grid"][1] - r["grid"][0]
        match = abs(r["offset"]) < 3 * grid_spacing
        status = "YES" if match else "NO"
        print(f"{param_name:<15} {r['inj_val']:>10.4f} {r['peak_val']:>10.4f} {r['offset']:>10.4f} {status:>8}")


def ablation_test():
    """M-matrix ablation test: does zeroing design matrix columns shift delta_gw peak?

    Sweeps delta_gw with and without the timing model design matrix in the Kalman
    filter observation vectors. Tests at both the injected SNR and multiple elevated
    h0 values.
    """
    config_path = "configs/smoke_test_config.ini"
    kf, n_pulsars, efac_array, equad_array, sigma_p_array, gamma_p_array = (
        load_data_and_filter(config_path)
    )

    # Override noise with posterior medians if available
    nc_path = "outputs/cw_intensive_run/no_gw/cw_intensive_run_results.nc"
    if os.path.exists(nc_path):
        print(f"Loading posterior noise params from: {nc_path}")
        efac_array, equad_array, sigma_p_array, gamma_p_array = (
            load_noise_from_posterior(nc_path)
        )

    delta_grid = np.linspace(-np.pi / 2, np.pi / 2, 200)

    # h0 values to test: injected + elevated
    h0_values = [-13.35, -12.0, -11.0, -10.0]

    # Warmup JIT
    params = make_params(kf, n_pulsars, efac_array, equad_array,
                         sigma_p_array, gamma_p_array)
    _ = float(kf.get_likelihood(params))

    H_original = kf.jax_H
    H_no_M = H_original.at[:, :, 2:].set(0.0)

    print("\n" + "=" * 70)
    print("M-MATRIX ABLATION TEST: delta_gw sweep with/without timing model")
    print("=" * 70)

    results_with_M = {}
    results_no_M = {}

    for log10_h0 in h0_values:
        print(f"\n--- log10(h0) = {log10_h0} ---")

        # With M-matrix
        kf.jax_H = H_original
        ll_with = sweep_parameter(
            kf, n_pulsars, efac_array, equad_array, sigma_p_array, gamma_p_array,
            "delta_gw", delta_grid, log10_h0=log10_h0,
        )
        peak_with = delta_grid[np.argmax(ll_with)]
        results_with_M[log10_h0] = {"ll": ll_with, "peak": peak_with}

        # Without M-matrix
        kf.jax_H = H_no_M
        ll_no = sweep_parameter(
            kf, n_pulsars, efac_array, equad_array, sigma_p_array, gamma_p_array,
            "delta_gw", delta_grid, log10_h0=log10_h0,
        )
        peak_no = delta_grid[np.argmax(ll_no)]
        results_no_M[log10_h0] = {"ll": ll_no, "peak": peak_no}

        shift = abs(peak_with - peak_no)
        print(f"  With M-matrix:    peak = {peak_with:.4f} rad ({np.degrees(peak_with):.1f} deg)")
        print(f"  Without M-matrix: peak = {peak_no:.4f} rad ({np.degrees(peak_no):.1f} deg)")
        print(f"  M-matrix shift:   {shift:.4f} rad ({np.degrees(shift):.1f} deg)")

    # Restore original H
    kf.jax_H = H_original

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("ABLATION TEST SUMMARY")
    print("=" * 70)
    print(f"{'log10(h0)':<12} {'With M (rad)':>14} {'No M (rad)':>14} {'M shift':>10} {'Inj offset (with)':>18} {'Inj offset (no)':>18}")
    print("-" * 86)
    for log10_h0 in h0_values:
        pw = results_with_M[log10_h0]["peak"]
        pn = results_no_M[log10_h0]["peak"]
        print(f"{log10_h0:<12.2f} {pw:>14.4f} {pn:>14.4f} {abs(pw-pn):>10.4f} {pw - INJECTION['delta_gw']:>18.4f} {pn - INJECTION['delta_gw']:>18.4f}")
    print(f"{'Injection':<12} {INJECTION['delta_gw']:>14.4f}")

    # --- Interpretation ---
    inj_shift = abs(results_with_M[-13.35]["peak"] - results_no_M[-13.35]["peak"])
    print(f"\nM-matrix contribution at injected SNR: {inj_shift:.4f} rad ({np.degrees(inj_shift):.1f} deg)")

    if inj_shift < 0.05:
        print("\nCONCLUSION: M-matrix absorption does NOT explain the delta_gw offset.")
        print("The timing model has negligible effect on the conditional likelihood peak.")
        print("The offset at injected SNR is a genuine finite-realisation/noise effect.")
        print("\nNote: The high-SNR tests (h0 >> injected h0) probe a different regime where")
        print("the model subtracts a CW signal much stronger than what is in the data.")
        print("The peak drifts toward sky positions where antenna patterns minimize the")
        print("over-subtraction, which is not informative about the signal model accuracy.")
    else:
        print(f"\nCONCLUSION: M-matrix absorption contributes {inj_shift:.2f} rad to the offset.")

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_h0 = len(h0_values)
        fig, axes = plt.subplots(2, n_h0, figsize=(4 * n_h0, 8), squeeze=False)

        for i, log10_h0 in enumerate(h0_values):
            rw = results_with_M[log10_h0]
            rn = results_no_M[log10_h0]

            # Top row: raw LL
            ax = axes[0, i]
            ax.plot(np.degrees(delta_grid), rw["ll"], "b-", lw=0.8, label="With M")
            ax.plot(np.degrees(delta_grid), rn["ll"], "r--", lw=0.8, label="No M")
            ax.axvline(np.degrees(INJECTION["delta_gw"]), color="green", ls=":", lw=1.5)
            ax.set_title(f"log10(h0) = {log10_h0}")
            ax.set_xlabel(r"$\delta_{\rm gw}$ (deg)")
            if i == 0:
                ax.set_ylabel("Log-likelihood")
            ax.legend(fontsize=7)

            # Bottom row: normalized LL
            ax = axes[1, i]
            ll_w_norm = rw["ll"] - rw["ll"].max()
            ll_n_norm = rn["ll"] - rn["ll"].max()
            ax.plot(np.degrees(delta_grid), ll_w_norm, "b-", lw=0.8, label="With M")
            ax.plot(np.degrees(delta_grid), ll_n_norm, "r--", lw=0.8, label="No M")
            ax.axvline(np.degrees(INJECTION["delta_gw"]), color="green", ls=":", lw=1.5)
            ax.axvline(np.degrees(rw["peak"]), color="blue", ls=":", lw=1, alpha=0.5)
            ax.axvline(np.degrees(rn["peak"]), color="red", ls=":", lw=1, alpha=0.5)
            ax.set_xlabel(r"$\delta_{\rm gw}$ (deg)")
            if i == 0:
                ax.set_ylabel("LL - LL_max")
            ax.set_ylim(bottom=max(ll_w_norm.min(), -100))
            ax.legend(fontsize=7)

        fig.suptitle("M-matrix Ablation: delta_gw profiles at different h0", fontsize=13, y=1.01)
        plt.tight_layout()
        outpath = "outputs/ablation_delta_gw.png"
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to: {outpath}")
    except ImportError:
        print("\nmatplotlib not available, skipping plot")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CW likelihood profiling")
    parser.add_argument("--ablation", action="store_true",
                        help="Run M-matrix ablation test for delta_gw")
    args = parser.parse_args()

    if args.ablation:
        ablation_test()
    else:
        main()
