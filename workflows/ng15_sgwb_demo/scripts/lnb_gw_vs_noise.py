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
In ridge mode the GW amplitude enters as the pivot log-PSD ``s = log10_pivot_psd``,
sampled from the bounded Uniform U(lo, hi) of the config (issue #125). The noise-only
model is the limit of vanishing amplitude. In any region R = [lo, c) of s deep enough
that the likelihood no longer depends on the GW parameters (L(s, rest) = L_noise(rest)),

    p(s | d) / pi(s) = Z_noise / Z_model                   for s in R,

because the conditional prior of everything else does not depend on s. So

    ln B(model / noise) = -ln[ P_post(s < c) / ((c - lo) / (hi - lo)) ]

for any threshold c inside R. This is Savage-Dickey on a region rather than at a
point; the null value is not in the box, so the point form is unavailable. It needs
the bottom of the box, lo, to sit where the likelihood is already flat; the plateau
gate below is the check.

Runs from before issue #125 sampled ``log10_pivot_psd_prime ~ N(0, 1)`` mapped onto the
box, an unbounded Gaussian prior. This version refuses them; read them out with the
script as of main 7616aa7, whose prior mass was Phi(c) in prior sigmas.

The estimate is only valid once the ratio has PLATEAUED in c: a ratio still drifting
as c deepens means the likelihood is not yet flat there. The plateau gate checks
that, and no headline number is reported without it. Like every density-ratio
estimator, this is reliable when the signal is weak (the posterior keeps mass at low
amplitude) and fails, visibly, when the detection is strong (no draws below c).

For a strong detection the script still reports a one-sided LOWER BOUND. If at most
an expected mu_up effective draws fall below c (a Poisson 95% upper limit on the
observed count), then P_post(s < c) <= mu_up / ESS and

    ln B(model / noise) >= ln[ (c - lo) / (hi - lo) * ESS / mu_up ].

ESS is an effective sample size, not the draw count, because neighbouring draws are
correlated. It is the indicator's ESS, or the pivot's tail ESS when no draw falls below
c (an all-zero indicator has no ESS of its own), whichever is smaller. This also needs the GW
not to be DISFAVOURED below c (L >= L_noise there), which holds whenever the
likelihood is flat or rising in s at that depth.

Run (CPU, no JAX):
    python workflows/ng15_sgwb_demo/scripts/lnb_gw_vs_noise.py \
        --run outputs/mdc2_d1_flat_uniformgw_eps000 --run outputs/mdc2_d1_flat_uniformgw_eps100 \
        --out outputs/lnb_gw_vs_noise_mdc2_d1_flat_uniformgw.json
"""

import argparse
import configparser
import glob
import json
import math
import os
import sys

import numpy as np
from scipy.stats import chi2, kstest, uniform

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lnb_path_sampling import effective_sample_size  # noqa: E402

PIVOT_SITE = "log10_pivot_psd"
GAMMA_SITE = "log10_gamma_a"
# The site of the pre-#125 Gaussian prior; a posterior carrying it is refused.
LEGACY_PIVOT_SITE = "log10_pivot_psd_prime"
# Default thresholds, as depths below the box centre in units of (hi - lo) / 6. These
# are the pivot values the Gaussian-prior version used (its c in prior sigmas), so old
# and new readouts are taken at the same s.
DEFAULT_DEPTHS = tuple(np.round(np.arange(-0.5, -2.51, -0.25), 2))

# A threshold counts only if this many EFFECTIVE draws sit below it.
MIN_EFFECTIVE_BELOW = 50.0
# Number of deepest eligible thresholds that must agree for a plateau.
PLATEAU_WIDTH = 3
# Agreement tolerance between plateau thresholds, in combined standard errors.
PLATEAU_NSIGMA = 2.0
# KS p-value below which the GW nuisance parameter is "not prior-like" in the region.
FLATNESS_P = 0.01


def check_box(box):
    """(lo, hi) of a bounded pivot prior; anything else is refused."""
    if box is None:
        raise ValueError("the pivot prior must be a bounded Uniform (lo, hi)")
    lo, hi = (float(x) for x in box)
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
        raise ValueError(f"pivot prior box ({lo}, {hi}) is not a bounded interval")
    return lo, hi


def default_thresholds(box):
    """The DEFAULT_DEPTHS as pivot values in the box ``(lo, hi)``."""
    lo, hi = check_box(box)
    return tuple(float((lo + hi) / 2 + d * (hi - lo) / 6) for d in DEFAULT_DEPTHS)


def prior_mass_below(c, box):
    """U(lo, hi) prior mass below threshold c."""
    lo, hi = check_box(box)
    if not lo < c <= hi:
        raise ValueError(f"threshold {c} is outside the prior box ({lo}, {hi}]")
    return (c - lo) / (hi - lo)


def threshold_estimate(pivot, c, box):
    """ln B(model/noise) at one threshold c, with its MCMC standard error.

    ``pivot`` is (n_chain, n_draw) draws of the pivot log-PSD, whose prior is U(box).
    The posterior fraction below c is an MCMC mean of an indicator, so its error uses
    the indicator's ESS, not the raw draw count.
    """
    indicator = (pivot < c).astype(float)
    frac = float(indicator.mean())
    prior_frac = prior_mass_below(c, box)
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


def lower_bound(pivot, box, c=None, cl=0.95):
    """One-sided ln B(model/noise) bound at threshold c (see the module docstring).

    c defaults to the shallowest default threshold.
    """
    import arviz as az

    if c is None:
        c = max(default_thresholds(box))
    pivot = np.atleast_2d(np.asarray(pivot, dtype=float))
    indicator = (pivot < c).astype(float)
    frac = float(indicator.mean())
    # The pivot's tail ESS stands in for the indicator's, which is undefined when no
    # draw falls below c; take the smaller whenever the indicator has one.
    ess = float(az.ess(pivot, method="tail"))
    if 0.0 < frac < 1.0:
        ess = min(ess, effective_sample_size(indicator))
    k_eff = frac * ess
    mu_up = 0.5 * chi2.ppf(cl, 2.0 * (k_eff + 1.0))
    frac_up = min(mu_up / ess, 1.0)
    return {
        "c": float(c),
        "cl": cl,
        "n_below": int(indicator.sum()),
        "ess": ess,
        "ln_bayes_factor_lower": float(math.log(prior_mass_below(c, box) / frac_up)),
    }


def flatness_check(pivot, gamma, c, gamma_box):
    """KS test of the GW nuisance parameter against its U(gamma_box) prior in the region.

    If the likelihood is flat in the GW parameters below c, gamma_a there is
    prior-distributed. A rejection means the region is not yet "noise-only".
    """
    if gamma is None:
        return None
    sel = gamma[pivot < c]
    if sel.size < 20:
        return None
    # Thin so draws are closer to independent; KS assumes i.i.d.
    tau = max(pivot.size / max(effective_sample_size(gamma), 1.0), 1.0)
    step = max(int(round(tau)), 1)
    thinned = gamma[:, ::step][pivot[:, ::step] < c]
    if thinned.size < 10:
        thinned = sel
    g_lo, g_hi = check_box(gamma_box)
    res = kstest(thinned, uniform(g_lo, g_hi - g_lo).cdf)
    return {
        "c": float(c),
        "n": int(thinned.size),
        "mean": float(sel.mean()),
        "std": float(sel.std()),
        "ks_p": float(res.pvalue),
    }


def analyse(pivot, box, gamma=None, gamma_box=None, thresholds=None):
    """Region Savage-Dickey over a threshold ladder, with the plateau gate.

    ``pivot`` holds draws of the pivot log-PSD under a U(box) prior; ``gamma`` holds
    draws of log10_gamma_a under U(gamma_box), used for the flatness check.
    """
    box = check_box(box)
    if thresholds is None:
        thresholds = default_thresholds(box)
    pivot = np.atleast_2d(np.asarray(pivot, dtype=float))
    if gamma is not None:
        gamma = np.atleast_2d(np.asarray(gamma, dtype=float))
    rows = [threshold_estimate(pivot, c, box) for c in sorted(thresholds, reverse=True)]
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
                        f"not plateaued: lnB({a['c']:.2f}) = {a['lnb']:.3f} vs "
                        f"lnB({b['c']:.2f}) = {b['lnb']:.3f} (tol {tol:.3f}) -- "
                        "likelihood not flat at this depth"
                    )

    headline = eligible[-1] if eligible else None
    flat = None
    if headline is not None:
        flat = flatness_check(pivot, gamma, headline["c"], gamma_box)
        if flat is not None and flat["ks_p"] < FLATNESS_P:
            failed.append(
                f"gamma_a not prior-like below c={headline['c']:.2f} "
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
        "lower_bound": lower_bound(pivot, box),
        "thresholds": rows,
    }


def load_run(run_dir):
    """Posterior GW draws and their prior boxes for one run directory."""
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
            "(a Uniform prior on the pivot log-PSD)."
        )
    box = (
        cfg.getfloat("PriorModel", "log10_pivot_psd_min"),
        cfg.getfloat("PriorModel", "log10_pivot_psd_max"),
    )
    gamma_box = (
        cfg.getfloat("PriorModel", "log10_gamma_a_min"),
        cfg.getfloat("PriorModel", "log10_gamma_a_max"),
    )

    post = az.from_netcdf(nc).posterior
    if LEGACY_PIVOT_SITE in post.data_vars:
        raise SystemExit(
            f"{nc}: has {LEGACY_PIVOT_SITE}, i.e. the pre-#125 Gaussian pivot prior. "
            "This estimator assumes U(lo, hi); read the run with the script at 7616aa7."
        )
    eps = None
    if "orf_epsilon" in post.data_vars:
        eps = float(np.unique(np.round(post["orf_epsilon"].values, 12)).item())
    return {
        "run": name,
        "nc": os.path.abspath(nc),
        "orf_epsilon": eps,
        "box": box,
        "gamma_box": gamma_box,
        "pivot": np.asarray(post[PIVOT_SITE].values, dtype=float),
        "gamma": np.asarray(post[GAMMA_SITE].values, dtype=float),
    }


def print_report(name, res):
    print(f"\n=== {name} ===")
    print("  pivot<   post   prior  n_eff_below   lnB(model/noise)   per-chain")
    for r in res["thresholds"]:
        lnb = (
            "     --      "
            if r["lnb"] is None
            else f"{r['lnb']:+.3f} ± {r['uncert']:.3f}"
        )
        chains = ", ".join(
            "--" if x is None else f"{x:+.2f}" for x in r["per_chain_lnb"]
        )
        print(
            f"  {r['c']:6.2f}  {r['post_frac']:.3f}  "
            f"{r['prior_frac']:.3f}  {r['effective_below']:10.1f}   {lnb}   [{chains}]"
        )
    if res["flatness"]:
        f = res["flatness"]
        print(
            f"  gamma_a | pivot<{f['c']:.2f}: mean {f['mean']:+.2f} std {f['std']:.2f} "
            f"(vs its uniform prior), KS p = {f['ks_p']:.3g} (n={f['n']})"
        )
    if res["reliable"]:
        print(
            f"  ln B(model/noise) = {res['ln_bayes_factor']:+.3f} ± {res['uncert']:.3f}"
        )
    else:
        print("  UNRELIABLE:")
        for msg in res["failed_diagnostics"]:
            print(f"    - {msg}")
        if res["provisional_ln_bayes_factor"] is not None:
            print(
                f"  provisional ln B = {res['provisional_ln_bayes_factor']:+.3f} ± "
                f"{res['provisional_uncert']:.3f}"
            )
    lb = res["lower_bound"]
    print(
        f"  {lb['cl']:.0%} lower bound: ln B >= {lb['ln_bayes_factor_lower']:+.2f} "
        f"(c={lb['c']:.2f}, {lb['n_below']} draws below, ESS {lb['ess']:.0f})"
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
        axes[1].hist(
            run["pivot"].ravel(), bins=40, density=True, histtype="step", label=name
        )
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_xlabel("threshold c (log10 pivot PSD)")
    axes[0].set_ylabel("ln B(model / noise-only)")
    axes[0].invert_xaxis()
    axes[0].legend(fontsize=8)
    run0 = next(iter(results.values()))[0]
    lo, hi = run0["box"]
    density = 1.0 / (hi - lo)
    axes[1].plot([lo, lo, hi, hi], [0, density, density, 0], "k--", label="prior")
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
        res = analyse(run["pivot"], run["box"], run["gamma"], run["gamma_box"])
        res.update(
            run=run["run"],
            nc=run["nc"],
            orf_epsilon=run["orf_epsilon"],
            pivot_prior={"uniform": list(run["box"])},
            gamma_a_prior={"uniform": list(run["gamma_box"])},
        )
        print_report(run["run"], res)
        results[run["run"]] = (run, res)

    summary = {"runs": [res for _, res in results.values()]}

    by_eps = {res["orf_epsilon"]: res for _, res in results.values()}
    pair = (by_eps.get(1.0), by_eps.get(0.0))
    if args.hd_curn and None in pair:
        args.hd_curn = None
    elif args.hd_curn and None in [r["provisional_ln_bayes_factor"] for r in pair]:
        print(
            "\n  cross-check lnB(HD/CURN) skipped: no region estimate (signal too strong)"
        )
        args.hd_curn = None
    if args.hd_curn:
        with open(args.hd_curn) as f:
            ps = json.load(f)
        hd, curn = pair
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
