#!/usr/bin/env python
"""T3.5 diagnostic: characterise the log10_ha <-> log10_gamma_a ridge that broke
convergence in the full-array quick-look run (job 14195666, max r_hat ~18.6, min ESS 4).

The full-array posterior is a strong, near-linear correlation ("ridge") between the two GW
latents. NUTS samples log10_ha_prime / log10_gamma_a_prime as *independent* Normal(0,1)
sites, so a diagonal mass matrix mixes poorly across the ridge and different chains settle
at different points along it -> one outlier chain inflates r_hat across essentially every
parameter. This script confirms that story from an existing MCMC .nc and quantifies the 2x2
correlation a per-block dense mass matrix would need to learn.

Pure post-processing (arviz / numpy / matplotlib); no GPU, no argus import, no library edits.
Run on CPU from anywhere:

    JAX_PLATFORMS=cpu python workflows/ng15_sgwb_demo/scripts/diagnose_gw_ridge.py \
        [--nc <results.nc>] [--outdir <dir>]

Defaults target the quick-look run. Outputs:
  <outdir>/ridge_diag.json                 (machine-readable summary)
  <outdir>/ridge_scatter.png               (log10_ha vs log10_gamma_a, colored by chain)
  <outdir>/gw_traces.png                    (per-chain traces of the two GW params)
"""

import argparse
import json
import os

import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # workflows/ng15_sgwb_demo

DEFAULT_NC = os.path.join(
    ROOT, "outputs", "ng15_full_quicklook", "ng15_full_quicklook_results.nc"
)
DEFAULT_OUTDIR = os.path.join(ROOT, "outputs", "ng15_full_quicklook", "ridge_diag")


def _chain_draw(idata, name):
    """Return the (n_chain, n_draw) array for a posterior variable, or None if absent."""
    if name not in idata.posterior:
        return None
    return np.asarray(idata.posterior[name].values)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--nc", default=DEFAULT_NC, help="InferenceData .nc file to diagnose"
    )
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR, help="output directory")
    args = ap.parse_args()

    print(f"Loading InferenceData: {args.nc}")
    idata = az.from_netcdf(args.nc)
    os.makedirs(args.outdir, exist_ok=True)

    ha = _chain_draw(idata, "log10_ha")
    ga = _chain_draw(idata, "log10_gamma_a")
    if ha is None or ga is None:
        raise SystemExit("log10_ha / log10_gamma_a not found in posterior")
    # The primes are the actual NUTS latents the dense block would act on.
    ha_p = _chain_draw(idata, "log10_ha_prime")
    ga_p = _chain_draw(idata, "log10_gamma_a_prime")

    n_chain, n_draw = ha.shape
    print(f"chains={n_chain}, draws/chain={n_draw}")

    # --- per-chain means: identify the outlier chain along the ridge ---
    ha_chain_mean = ha.mean(axis=1)
    ga_chain_mean = ga.mean(axis=1)
    # Outlier = chain whose log10_ha mean is farthest from the median chain-mean.
    ha_med = float(np.median(ha_chain_mean))
    outlier_chain = int(np.argmax(np.abs(ha_chain_mean - ha_med)))

    # --- ridge correlation (this is what a dense 2x2 mass block must capture) ---
    def corr2(a, b):
        af, bf = a.reshape(-1), b.reshape(-1)
        cov = np.cov(np.vstack([af, bf]))
        r = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        return cov, float(r)

    cov_det, r_det = corr2(ha, ga)  # deterministic (log10_ha, log10_gamma_a)
    prime_info = None
    if ha_p is not None and ga_p is not None:
        cov_p, r_p = corr2(ha_p, ga_p)
        prime_info = {
            "cov_2x2": cov_p.tolist(),
            "correlation": r_p,
            "note": "covariance of the NUTS latents; the dense block learns this 2x2",
        }

    # --- r_hat / ESS for the two GW params ---
    def rhat_ess(name):
        try:
            return {
                "r_hat": float(az.rhat(idata, var_names=[name])[name].values),
                "ess_bulk": float(
                    az.ess(idata, var_names=[name], method="bulk")[name].values
                ),
            }
        except Exception as e:  # pragma: no cover - diagnostic robustness
            return {"error": str(e)}

    # --- does the outlier chain also drive the high-r_hat log10_sigma_p tail? ---
    sp = _chain_draw(idata, "log10_σp")
    sp_check = None
    if sp is not None and sp.ndim == 3:  # (chain, draw, n_pulsar)
        n_psr = sp.shape[2]
        # per-chain, per-pulsar means: (chain, n_pulsar)
        sp_chain_mean = sp.mean(axis=1)
        # For each pulsar, is the log10_ha outlier chain also the sigma_p outlier chain?
        tail = list(
            range(max(0, n_psr - 5), n_psr)
        )  # last 5 (the sigma_p[63..67] seen)
        agree = []
        for p in tail:
            col = sp_chain_mean[:, p]
            sp_outlier = int(np.argmax(np.abs(col - np.median(col))))
            agree.append(sp_outlier == outlier_chain)
        sp_check = {
            "n_pulsars": int(n_psr),
            "tail_pulsars_checked": tail,
            "sigma_p_outlier_is_ha_outlier": [bool(a) for a in agree],
            "fraction_agree": float(np.mean(agree)) if agree else None,
            "interpretation": (
                "high fraction => the sigma_p r_hat inflation is the SAME outlier chain "
                "following the ridge (one problem), not independent pathology"
            ),
        }

    summary = {
        "nc": args.nc,
        "n_chain": int(n_chain),
        "n_draw": int(n_draw),
        "log10_ha_chain_means": ha_chain_mean.tolist(),
        "log10_gamma_a_chain_means": ga_chain_mean.tolist(),
        "outlier_chain_index": outlier_chain,
        "outlier_chain_log10_ha_mean": float(ha_chain_mean[outlier_chain]),
        "other_chains_log10_ha_mean": float(
            np.mean(np.delete(ha_chain_mean, outlier_chain))
        ),
        "ridge_correlation_deterministic": {
            "cov_2x2": cov_det.tolist(),
            "correlation": r_det,
        },
        "ridge_correlation_primes": prime_info,
        "log10_ha_diag": rhat_ess("log10_ha"),
        "log10_gamma_a_diag": rhat_ess("log10_gamma_a"),
        "sigma_p_outlier_check": sp_check,
    }

    out_json = os.path.join(args.outdir, "ridge_diag.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_json}")

    # --- plots ---
    # 1) scatter colored by chain
    fig, axscat = plt.subplots(figsize=(6, 5))
    for c in range(n_chain):
        axscat.scatter(
            ha[c],
            ga[c],
            s=4,
            alpha=0.35,
            label=f"chain {c}" + (" (outlier)" if c == outlier_chain else ""),
        )
    axscat.set_xlabel("log10_ha")
    axscat.set_ylabel("log10_gamma_a")
    axscat.set_title(f"GW ridge (r={r_det:+.3f}), colored by chain")
    axscat.legend(markerscale=3, fontsize=8)
    fig.tight_layout()
    p_scatter = os.path.join(args.outdir, "ridge_scatter.png")
    fig.savefig(p_scatter, dpi=130)
    plt.close(fig)
    print(f"Wrote {p_scatter}")

    # 2) per-chain traces
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for c in range(n_chain):
        lbl = f"chain {c}" + (" (outlier)" if c == outlier_chain else "")
        a0.plot(ha[c], lw=0.7, alpha=0.8, label=lbl)
        a1.plot(ga[c], lw=0.7, alpha=0.8)
    a0.set_ylabel("log10_ha")
    a1.set_ylabel("log10_gamma_a")
    a1.set_xlabel("draw")
    a0.legend(fontsize=8, ncol=n_chain)
    a0.set_title("Per-chain traces of the GW parameters")
    fig.tight_layout()
    p_trace = os.path.join(args.outdir, "gw_traces.png")
    fig.savefig(p_trace, dpi=130)
    plt.close(fig)
    print(f"Wrote {p_trace}")

    # --- console summary ---
    print("\n=== ridge diagnostic summary ===")
    print(f"log10_ha chain means: {np.round(ha_chain_mean, 3).tolist()}")
    print(
        f"outlier chain = {outlier_chain} "
        f"(log10_ha mean {ha_chain_mean[outlier_chain]:.3f} vs "
        f"others {np.mean(np.delete(ha_chain_mean, outlier_chain)):.3f})"
    )
    print(f"(log10_ha, log10_gamma_a) correlation = {r_det:+.3f}")
    if prime_info:
        print(f"(prime) correlation                 = {prime_info['correlation']:+.3f}")
    print(f"log10_ha r_hat/ESS: {summary['log10_ha_diag']}")
    if sp_check:
        print(
            f"sigma_p tail follows ha-outlier chain: "
            f"{sp_check['fraction_agree']:.2f} fraction of "
            f"{len(sp_check['tail_pulsars_checked'])} checked"
        )


if __name__ == "__main__":
    main()
