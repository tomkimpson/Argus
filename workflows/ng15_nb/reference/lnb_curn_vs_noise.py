"""lnB(CURN / intrinsic noise) for a Discovery reference rung, by region Savage-Dickey.

Spec task 1.5. The common-process-vs-noise Bayes factor comes from the ε = 0 (CURN) rung
alone: below an amplitude threshold c deep enough that the likelihood is flat,
p(s | d) / π(s) = Z_noise / Z_CURN. That estimator already exists for Argus
(``workflows/ng15_sgwb_demo/scripts/lnb_gw_vs_noise.py``), written for the ridge basis's
N(0, 1) latent. Here the amplitude has an exact uniform prior on [lo, hi], so it is mapped
to the same latent by the probability integral transform,

    prime = Φ⁻¹((s − lo) / (hi − lo)),

which is N(0, 1) under the prior. The estimator's prior mass below a threshold, Φ(c),
is then exactly the uniform mass (s_c − lo)/(hi − lo), and its analysis (threshold ladder,
plateau gate, γ flatness KS test, Poisson lower bound) is reused unchanged.

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
from lnb_gw_vs_noise import analyse  # noqa: E402

# (amplitude site, its config box key), (index site, its config box key), per kernel.
SITES = {
    "pl": (("gw_log10_A", "gw_log10_A"), ("gw_gamma", "gw_gamma")),
    "ou": (("gw_log10_pivot_psd", "log10_pivot_psd"), ("gw_log10_gamma_a", "log10_gamma_a")),
}


def to_latent(x, lo, hi):
    """Probability integral transform of U(lo, hi) draws to N(0, 1)."""
    u = (np.asarray(x, dtype=float) - lo) / (hi - lo)
    if np.any((u < 0) | (u > 1)):
        raise ValueError(f"draws outside the prior box [{lo}, {hi}]")
    return norm.ppf(np.clip(u, 1e-300, 1 - 1e-16))


def run(run_dir):
    import arviz as az

    run_dir = Path(run_dir)
    cfg = configparser.ConfigParser()
    cfg.read(run_dir / "resolved_config.ini")
    eps = cfg.getfloat("PriorModel", "orf_epsilon_value")
    if eps != 0.0:
        raise SystemExit(f"{run_dir}: eps = {eps}; CURN vs noise needs the eps = 0 rung")
    kernel = cfg.get("Model", "kernel")
    (amp_site, amp_key), (idx_site, idx_key) = SITES[kernel]
    box = lambda key: (cfg.getfloat("PriorModel", f"{key}_min"), cfg.getfloat("PriorModel", f"{key}_max"))  # noqa: E731

    post = az.from_netcdf(run_dir / "samples.nc").posterior
    prime = to_latent(post[amp_site].values, *box(amp_key))
    gamma_prime = to_latent(post[idx_site].values, *box(idx_key)) if idx_site in post.data_vars else None
    res = analyse(prime, gamma_prime)
    res.update(
        run=str(run_dir),
        kernel=kernel,
        amplitude_site=amp_site,
        amplitude_box=box(amp_key),
        prior_mapping="prime = Phi^-1((s - lo)/(hi - lo)); Phi(c) is the uniform prior mass below s_c",
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
        f"lower bound (95%) = {lb['ln_bayes_factor_lower']:.2f} at c={lb['c']} "
        f"({lb['n_below']} draws below, ESS {lb['ess']:.0f})"
    )
    for msg in res["failed_diagnostics"]:
        print(f"  - {msg}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
