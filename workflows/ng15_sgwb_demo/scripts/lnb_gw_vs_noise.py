#!/usr/bin/env python
"""GW+noise vs noise-only Bayes factor from one run, by Savage-Dickey on a region.

Why this exists
---------------
The literature's strong MDC2 1b detection (Hazboun et al. 2020, B = 23 for CRN and 40
for HD) is GW+noise against NOISE-ONLY, not HD against CURN. The path sampler in
``lnb_path_sampling.py`` only integrates along the correlation coordinate eps, so it
cannot produce that comparison.

The identity
------------
In ridge mode the GW amplitude enters as the pivot log-PSD
``s = log10_pivot_psd = mu + sigma * prime``, ``prime ~ N(0, 1)``. The noise-only
model is the limit s -> -inf. In any region R of s deep enough that the likelihood
no longer depends on the GW parameters (L(s, rest) = L_noise(rest)),

    p(s | d) / pi(s) = Z_noise / Z_model                   for s in R,

because the conditional prior of everything else does not depend on s. So

    ln B(model / noise) = -ln[ P_post(prime < c) / Phi(c) ]

for any threshold c inside R. This is Savage-Dickey on a region rather than at a
point; the null value sits at -inf so the point form is unavailable.

The estimate is only valid once the ratio has PLATEAUED in c: a ratio still drifting
as c deepens means the likelihood is not yet flat there. The plateau gate checks
that, and no headline number is reported without it. Like every density-ratio
estimator, this is reliable when the signal is weak (the posterior keeps mass at low
amplitude) and fails, visibly, when the detection is strong (no draws below c).

Run (CPU, no JAX):
    python workflows/ng15_sgwb_demo/scripts/lnb_gw_vs_noise.py \
        --run outputs/mdc2_d1_flat_uprior_eps000 --run outputs/mdc2_d1_flat_uprior_eps100 \
        --out outputs/lnb_gw_vs_noise_mdc2_d1_flat_uprior.json
"""

import argparse
import configparser
import glob
import json
import math
import os
import sys

import numpy as np
from scipy.stats import kstest, norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lnb_path_sampling import effective_sample_size  # noqa: E402

PIVOT_SITE = "log10_pivot_psd_prime"
GAMMA_SITE = "log10_gamma_a_prime"
DEFAULT_THRESHOLDS = tuple(np.round(np.arange(-0.5, -2.51, -0.25), 2))

# A threshold counts only if this many EFFECTIVE draws sit below it.
MIN_EFFECTIVE_BELOW = 50.0
# Number of deepest eligible thresholds that must agree for a plateau.
PLATEAU_WIDTH = 3
# Agreement tolerance between plateau thresholds, in combined standard errors.
PLATEAU_NSIGMA = 2.0
# KS p-value below which the GW nuisance parameter is "not prior-like" in the region.
FLATNESS_P = 0.01


def threshold_estimate(prime, c):
    """ln B(model/noise) at one threshold c, with its MCMC standard error.

    ``prime`` is (n_chain, n_draw). The posterior fraction below c is an MCMC mean of
    an indicator, so its error uses the indicator's ESS, not the raw draw count.
    """
    indicator = (prime < c).astype(float)
    frac = float(indicator.mean())
    prior_frac = float(norm.cdf(c))
    ess = effective_sample_size(indicator)
    out = {
        "c": float(c),
        "post_frac": frac,
        "prior_frac": prior_frac,
        "n_below": int(indicator.sum()),
        "ess": ess,
        "effective_below": frac * ess,
        "per_chain_lnb": [
            float(-math.log(f / prior_frac)) if f > 0 else None
            for f in indicator.mean(axis=1)
        ],
    }
    if frac <= 0:
        out.update(lnb=None, uncert=None)
        return out
    sigma_frac = math.sqrt(frac * (1.0 - frac) / max(ess, 1.0))
    out.update(lnb=float(-math.log(frac / prior_frac)), uncert=sigma_frac / frac)
    return out


def flatness_check(prime, gamma_prime, c):
    """KS test of the GW nuisance parameter against its N(0,1) prior inside the region.

    If the likelihood is flat in the GW parameters below c, gamma_a there is
    prior-distributed. A rejection means the region is not yet "noise-only".
    """
    if gamma_prime is None:
        return None
    sel = gamma_prime[prime < c]
    if sel.size < 20:
        return None
    # Thin so draws are closer to independent; KS assumes i.i.d.
    n_chain, n_draw = prime.shape
    tau = max(prime.size / max(effective_sample_size(gamma_prime), 1.0), 1.0)
    step = max(int(round(tau)), 1)
    thinned = gamma_prime[:, ::step][prime[:, ::step] < c]
    if thinned.size < 10:
        thinned = sel
    res = kstest(thinned, "norm")
    return {
        "c": float(c),
        "n": int(thinned.size),
        "mean": float(sel.mean()),
        "std": float(sel.std()),
        "ks_p": float(res.pvalue),
    }


def analyse(prime, gamma_prime=None, thresholds=DEFAULT_THRESHOLDS):
    """Region Savage-Dickey over a threshold ladder, with the plateau gate."""
    prime = np.atleast_2d(np.asarray(prime, dtype=float))
    if gamma_prime is not None:
        gamma_prime = np.atleast_2d(np.asarray(gamma_prime, dtype=float))
    rows = [threshold_estimate(prime, c) for c in sorted(thresholds, reverse=True)]
    eligible = [
        r
        for r in rows
        if r["lnb"] is not None and r["effective_below"] >= MIN_EFFECTIVE_BELOW
    ]

    failed = []
    plateau = eligible[-PLATEAU_WIDTH:]
    if len(plateau) < PLATEAU_WIDTH:
        failed.append(
            f"only {len(eligible)} threshold(s) with >= {MIN_EFFECTIVE_BELOW:.0f} "
            f"effective draws below; need {PLATEAU_WIDTH} for a plateau "
            "(signal too strong for a density-ratio estimate)"
        )
    else:
        for a in plateau:
            for b in plateau:
                if a["c"] <= b["c"]:
                    continue
                tol = PLATEAU_NSIGMA * math.hypot(a["uncert"], b["uncert"])
                if abs(a["lnb"] - b["lnb"]) > tol:
                    failed.append(
                        f"not plateaued: lnB({a['c']}) = {a['lnb']:.3f} vs "
                        f"lnB({b['c']}) = {b['lnb']:.3f} (tol {tol:.3f}) -- "
                        "likelihood not flat at this depth"
                    )

    headline = eligible[-1] if eligible else None
    flat = None
    if headline is not None:
        flat = flatness_check(prime, gamma_prime, headline["c"])
        if flat is not None and flat["ks_p"] < FLATNESS_P:
            failed.append(
                f"gamma_a not prior-like below c={headline['c']} "
                f"(KS p = {flat['ks_p']:.3g})"
            )

    reliable = not failed
    return {
        "estimator": "region_savage_dickey",
        "ln_bayes_factor": headline["lnb"] if (reliable and headline) else None,
        "uncert": headline["uncert"] if (reliable and headline) else None,
        "provisional_ln_bayes_factor": headline["lnb"] if headline else None,
        "provisional_uncert": headline["uncert"] if headline else None,
        "headline_threshold": headline["c"] if headline else None,
        "reliable": reliable,
        "failed_diagnostics": failed,
        "flatness": flat,
        "thresholds": rows,
    }


def load_run(run_dir):
    """Posterior latents and pivot prior (mu, sigma) of one run directory."""
    import arviz as az

    run_dir = run_dir.rstrip("/")
    name = os.path.basename(run_dir)
    nc = os.path.join(run_dir, f"{name}_results.nc")
    if not os.path.exists(nc):
        matched = sorted(glob.glob(os.path.join(run_dir, "*_results.nc")))
        if len(matched) != 1:
            raise SystemExit(f"{run_dir}: expected one *_results.nc, found {matched}")
        nc = matched[0]
    ini = sorted(glob.glob(os.path.join(run_dir, "*.ini")))
    if len(ini) != 1:
        raise SystemExit(f"{run_dir}: expected one .ini, found {ini}")
    cfg = configparser.ConfigParser()
    cfg.read(ini[0])
    mode = cfg.get("PriorModel", "gw_parameterization", fallback="direct").strip()
    if mode.lower() != "ridge":
        raise SystemExit(
            f"{ini[0]}: gw_parameterization = {mode}; the estimator needs ridge "
            "(a Gaussian prior on the pivot log-PSD)."
        )
    lo = cfg.getfloat("PriorModel", "log10_pivot_psd_min")
    hi = cfg.getfloat("PriorModel", "log10_pivot_psd_max")

    post = az.from_netcdf(nc).posterior
    eps = None
    if "orf_epsilon" in post.data_vars:
        eps = float(np.unique(np.round(post["orf_epsilon"].values, 12)).item())
    return {
        "run": name,
        "nc": os.path.abspath(nc),
        "orf_epsilon": eps,
        "pivot_mu": (lo + hi) / 2.0,
        "pivot_sigma": (hi - lo) / 6.0,
        "prime": np.asarray(post[PIVOT_SITE].values, dtype=float),
        "gamma_prime": (
            np.asarray(post[GAMMA_SITE].values, dtype=float)
            if GAMMA_SITE in post.data_vars
            else None
        ),
    }


def print_report(name, res, mu, sigma):
    print(f"\n=== {name} ===")
    print("    c  pivot<   post   prior  n_eff_below   lnB(model/noise)   per-chain")
    for r in res["thresholds"]:
        lnb = "     --      " if r["lnb"] is None else f"{r['lnb']:+.3f} ± {r['uncert']:.3f}"
        chains = ", ".join("--" if x is None else f"{x:+.2f}" for x in r["per_chain_lnb"])
        print(
            f"  {r['c']:5.2f}  {mu + r['c'] * sigma:6.2f}  {r['post_frac']:.3f}  "
            f"{r['prior_frac']:.3f}  {r['effective_below']:10.1f}   {lnb}   [{chains}]"
        )
    if res["flatness"]:
        f = res["flatness"]
        print(
            f"  gamma_a | prime<{f['c']}: mean {f['mean']:+.2f} std {f['std']:.2f} "
            f"(prior 0, 1), KS p = {f['ks_p']:.3g} (n={f['n']})"
        )
    if res["reliable"]:
        print(f"  ln B(model/noise) = {res['ln_bayes_factor']:+.3f} ± {res['uncert']:.3f}")
    else:
        print("  UNRELIABLE:")
        for msg in res["failed_diagnostics"]:
            print(f"    - {msg}")
        if res["provisional_ln_bayes_factor"] is not None:
            print(
                f"  provisional ln B = {res['provisional_ln_bayes_factor']:+.3f} ± "
                f"{res['provisional_uncert']:.3f}"
            )


def plot(results, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, (run, res) in results.items():
        rows = [r for r in res["thresholds"] if r["lnb"] is not None]
        axes[0].errorbar(
            [r["c"] for r in rows],
            [r["lnb"] for r in rows],
            yerr=[r["uncert"] for r in rows],
            marker="o",
            capsize=3,
            label=name,
        )
        s = run["pivot_mu"] + run["pivot_sigma"] * run["prime"].ravel()
        axes[1].hist(s, bins=40, density=True, histtype="step", label=name)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_xlabel("threshold c (prior sigmas of pivot log-PSD)")
    axes[0].set_ylabel("ln B(model / noise-only)")
    axes[0].invert_xaxis()
    axes[0].legend(fontsize=8)
    run0 = next(iter(results.values()))[0]
    grid = np.linspace(run0["pivot_mu"] - 4 * run0["pivot_sigma"],
                       run0["pivot_mu"] + 4 * run0["pivot_sigma"], 200)
    axes[1].plot(grid, norm.pdf(grid, run0["pivot_mu"], run0["pivot_sigma"]), "k--",
                 label="prior")
    axes[1].set_xlabel("log10 pivot PSD (two-sided)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"  wrote {path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", action="append", required=True, help="Run output dir.")
    p.add_argument(
        "--hd-curn",
        default=None,
        help="Path-sampling JSON for lnB(HD/CURN), cross-checked against the "
        "difference of the eps=1 and eps=0 runs.",
    )
    p.add_argument("--out", default=None, help="Write JSON summary here.")
    p.add_argument("--plot", default=None, help="Write a diagnostic PNG here.")
    args = p.parse_args()

    results = {}
    for run_dir in args.run:
        run = load_run(run_dir)
        res = analyse(run["prime"], run["gamma_prime"])
        res.update(
            run=run["run"],
            nc=run["nc"],
            orf_epsilon=run["orf_epsilon"],
            pivot_prior={"mu": run["pivot_mu"], "sigma": run["pivot_sigma"]},
        )
        print_report(run["run"], res, run["pivot_mu"], run["pivot_sigma"])
        results[run["run"]] = (run, res)

    summary = {"runs": [res for _, res in results.values()]}

    by_eps = {res["orf_epsilon"]: res for _, res in results.values()}
    if args.hd_curn and 0.0 in by_eps and 1.0 in by_eps:
        with open(args.hd_curn) as f:
            ps = json.load(f)
        hd, curn = by_eps[1.0], by_eps[0.0]
        diff = hd["provisional_ln_bayes_factor"] - curn["provisional_ln_bayes_factor"]
        # The two runs are independent chains, so their errors add in quadrature.
        sig = math.hypot(hd["provisional_uncert"], curn["provisional_uncert"])
        tol = math.hypot(sig, ps["uncert"])
        summary["hd_curn_crosscheck"] = {
            "region_sd_difference": diff,
            "region_sd_uncert": sig,
            "path_sampling": ps["ln_bayes_factor"],
            "path_sampling_uncert": ps["uncert"],
            "pull": (diff - ps["ln_bayes_factor"]) / tol,
        }
        print(
            f"\n  cross-check lnB(HD/CURN): region-SD {diff:+.3f} ± {sig:.3f} vs "
            f"path sampling {ps['ln_bayes_factor']:+.3f} ± {ps['uncert']:.3f} "
            f"(pull {summary['hd_curn_crosscheck']['pull']:+.2f}σ)"
        )

    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  wrote {args.out}")
    if args.plot:
        plot(results, args.plot)
    return 0 if all(res["reliable"] for _, res in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
