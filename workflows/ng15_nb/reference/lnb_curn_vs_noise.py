"""lnB(CURN / intrinsic noise) for a Discovery reference rung, by region Savage-Dickey.

Spec task 1.5. The common-process-vs-noise Bayes factor comes from the ε = 0 (CURN) rung
alone: below an amplitude threshold c deep enough that the likelihood is flat,
p(s | d) / π(s) = Z_noise / Z_CURN. That estimator already exists for Argus
(``workflows/ng15_sgwb_demo/scripts/lnb_gw_vs_noise.py``), and since issue #125 it takes
the amplitude under its exact U(lo, hi) prior, so it is reused unchanged (threshold
ladder, plateau gate, γ flatness KS test, Poisson lower bound).

The thresholds are those of the first version of this script, which mapped the amplitude
onto an N(0, 1) latent, prime = Φ⁻¹((s − lo)/(hi − lo)), and cut at prime = c for
c in −0.5 … −2.5: here s_c = lo + Φ(c)(hi − lo), whose uniform prior mass is Φ(c). The
estimates and lower bounds are the same as that version's (the region fractions, the
indicator and tail ESS are all invariant under the monotone map).

NANOGrav's published curnγ-vs-noise B exceeds 10¹⁴, so no draw is expected anywhere near
the lower edge: the estimator will report reliable: false and a one-sided lower bound of
order ln(prior mass × ESS), ≈ 4–6 nats at these run lengths. That bound is the
reportable number; K2 turns on lnB(HD/CURN) only.

    conda activate Argus
    python workflows/ng15_nb/reference/lnb_curn_vs_noise.py \\
        workflows/ng15_nb/outputs/ng15nb_pl_eps000 --out .../lnb_curn_noise_pl.json
"""

import argparse
import configparser
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "workflows/ng15_sgwb_demo/scripts"))
from lnb_gw_vs_noise import DEFAULT_DEPTHS, analyse  # noqa: E402

# (amplitude site, its config box key), (index site, its config box key), per kernel.
SITES = {
    "pl": (("gw_log10_A", "gw_log10_A"), ("gw_gamma", "gw_gamma")),
    "ou": (
        ("gw_log10_pivot_psd", "log10_pivot_psd"),
        ("gw_log10_gamma_a", "log10_gamma_a"),
    ),
}


def thresholds(lo, hi):
    """Amplitude thresholds with uniform prior mass Phi(c), c in DEFAULT_DEPTHS."""
    return tuple(float(lo + norm.cdf(c) * (hi - lo)) for c in DEFAULT_DEPTHS)


def check_in_box(x, lo, hi):
    """Refuse draws outside the prior box (a config/run mismatch)."""
    x = np.asarray(x, dtype=float)
    if np.any((x < lo) | (x > hi)):
        raise ValueError(f"draws outside the prior box [{lo}, {hi}]")
    return x


def run(run_dir):
    import arviz as az

    run_dir = Path(run_dir)
    cfg = configparser.ConfigParser()
    cfg.read(run_dir / "resolved_config.ini")
    eps = cfg.getfloat("PriorModel", "orf_epsilon_value")
    if eps != 0.0:
        raise SystemExit(
            f"{run_dir}: eps = {eps}; CURN vs noise needs the eps = 0 rung"
        )
    kernel = cfg.get("Model", "kernel")
    (amp_site, amp_key), (idx_site, idx_key) = SITES[kernel]
    box = lambda key: (
        cfg.getfloat("PriorModel", f"{key}_min"),
        cfg.getfloat("PriorModel", f"{key}_max"),
    )  # noqa: E731

    post = az.from_netcdf(run_dir / "samples.nc").posterior
    amp = check_in_box(post[amp_site].values, *box(amp_key))
    gamma = (
        check_in_box(post[idx_site].values, *box(idx_key))
        if idx_site in post.data_vars
        else None
    )
    res = analyse(
        amp, box(amp_key), gamma, box(idx_key), thresholds=thresholds(*box(amp_key))
    )
    res.update(
        run=str(run_dir),
        kernel=kernel,
        amplitude_site=amp_site,
        amplitude_box=box(amp_key),
        prior_mapping="thresholds s_c = lo + Phi(c) (hi - lo), c in -0.5 ... -2.5",
    )
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("--out")
    args = ap.parse_args()
    res = run(args.run_dir)
    lb = res["lower_bound"]
    print(
        f"{res['run']}: reliable={res['reliable']}  lnB={res['ln_bayes_factor']}  "
        f"lower bound (95%) = {lb['ln_bayes_factor_lower']:.2f} at c={lb['c']:.3f} "
        f"({lb['n_below']} draws below, ESS {lb['ess']:.0f})"
    )
    for msg in res["failed_diagnostics"]:
        print(f"  - {msg}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
