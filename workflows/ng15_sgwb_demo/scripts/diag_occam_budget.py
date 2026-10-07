#!/usr/bin/env python
"""Occam budget of the two pivot modes, and lnB under narrower red-noise priors.

Why this exists
---------------
On MDC2 1b the ridge pivot posterior is bimodal. In the low mode (s < -10) the GW is
off and per-pulsar OU red noise is on; in the high mode (s > -8) the reverse. The low
mode has the HIGHER mean log-likelihood yet no more posterior mass, so the balance --
and with it lnB(model / noise-only) -- is set by prior volume. This script measures
that volume, block by block, from the stored draws (post-processing only).

Part A -- the budget
--------------------
For a mode M with prior mass pi(M), Z_M = int_M L pi = Z P(M | d), and

    ln Z_M = <ln L>_M - KL( p(.|d,M) || pi(.|M) ) + ln pi(M),

so the difference in Occam penalty between the modes is exact up to MCMC error:

    dKL = d<ln L> + d ln pi(M) - ln[ P(low|d) / P(high|d) ]        (low - high).

The prior factorises over parameter blocks (each pulsar's red noise, each pulsar's
white noise, the GW pair), so KL_joint = sum_b KL_b + TC, where TC >= 0 is the
posterior's total correlation between blocks. Each block's marginal KL is
ln(prior volume)-type cross entropy minus the block's posterior entropy, estimated
two ways: Gaussian (Laplace) and Kozachenko-Leonenko nearest neighbours.

Part B -- prior restriction, exactly
------------------------------------
Restricting the prior to a region R of the red-noise parameters changes both
evidences by a posterior fraction (pi(R) cancels in the ratio):

    d lnB(model / noise) = ln P(R | d) - ln P(R | d, noise),

with P(. | d, noise) read from the draws in the noise-only region (pivot prime < c,
the same region ``lnb_gw_vs_noise.py`` uses). No rerun is needed; entries with too few
effective draws inside R are refused rather than reported.

    python workflows/ng15_sgwb_demo/scripts/diag_occam_budget.py \
        --run outputs/mdc2_d1_flat_uprior_eps000 --run outputs/mdc2_d1_flat_uprior_eps100 \
        --lnb outputs/lnb_gw_vs_noise_mdc2_d1_flat_uprior.json \
        --out outputs/diag_occam_budget_mdc2_d1_flat_uprior.json \
        --plot outputs/diag_occam_budget_mdc2_d1_flat_uprior.png
"""

import argparse
import configparser
import glob
import json
import math
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma, gammaln
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lnb_path_sampling import effective_sample_size  # noqa: E402

LOW, HIGH = -10.0, -8.0  # same mode split as diag_gw_noise_tradeoff.py
KNN_K = 4
MIN_EFFECTIVE = 50.0
TOP_PULSARS = 6


# ---------------------------------------------------------------------------
# Entropy and KL estimators
# ---------------------------------------------------------------------------


def gaussian_entropy(x):
    """Entropy of the Gaussian with the sample covariance of x (n, d)."""
    x = np.asarray(x, dtype=float).reshape(len(x), -1)
    d = x.shape[1]
    cov = np.atleast_2d(np.cov(x, rowvar=False))
    _, logdet = np.linalg.slogdet(cov)
    return 0.5 * (d * math.log(2.0 * math.pi * math.e) + logdet)


def knn_entropy(x, chain, k=KNN_K):
    """Kozachenko-Leonenko entropy (nats) of MCMC draws x (n, d), cross-chain.

    Consecutive draws of one chain are correlated and sit close together, which
    biases a plain nearest-neighbour estimate low. Instead each draw's neighbours
    are searched only among the OTHER chains' draws, which are independent of it;
    with m reference draws the estimator is psi(m) - psi(k) + ln V_d + d <ln r_k>.
    """
    x = np.asarray(x, dtype=float).reshape(len(x), -1)
    chain = np.asarray(chain)
    d = x.shape[1]
    # Tiny jitter breaks exact ties (repeated draws from rejected NUTS proposals).
    x = x + 1e-9 * np.random.default_rng(0).normal(size=x.shape) * x.std(axis=0)
    log_unit_ball = 0.5 * d * math.log(math.pi) - gammaln(0.5 * d + 1.0)
    terms = []
    for c in np.unique(chain):
        ref, qry = x[chain != c], x[chain == c]
        if len(ref) <= k or len(qry) == 0:
            continue
        r = cKDTree(ref).query(qry, k=k)[0][:, k - 1]
        terms.append((len(qry), digamma(len(ref)) - digamma(k) + log_unit_ball
                      + d * np.mean(np.log(r))))
    n = sum(t[0] for t in terms)
    return float(sum(w * h for w, h in terms) / n)


def block_kl(x, chain, cross_entropy):
    """KL(post || prior) of one block = cross entropy - posterior entropy."""
    h_gauss = gaussian_entropy(x)
    h_knn = knn_entropy(x, chain)
    return {"kl_gauss": cross_entropy - h_gauss, "kl_knn": cross_entropy - h_knn,
            "h_gauss": h_gauss, "h_knn": h_knn}


def uniform_cross_entropy(bounds):
    """-E_post[ln pi] for a uniform box prior: ln(volume)."""
    return float(sum(math.log(hi - lo) for lo, hi in bounds))


def std_normal_cross_entropy(z):
    """-E_post[ln pi] for a standard-normal prior on the columns of z (n, d)."""
    z = np.asarray(z, dtype=float).reshape(len(z), -1)
    return float(0.5 * z.shape[1] * math.log(2 * math.pi) + 0.5 * np.mean(np.sum(z**2, 1)))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_run(run_dir):
    import arviz as az

    run_dir = run_dir.rstrip("/")
    nc = sorted(glob.glob(os.path.join(run_dir, "*_results.nc")))
    ini = sorted(glob.glob(os.path.join(run_dir, "*.ini")))
    if len(nc) != 1 or len(ini) != 1:
        raise SystemExit(f"{run_dir}: expected one *_results.nc and one .ini")
    cfg = configparser.ConfigParser()
    cfg.read(ini[0])
    pm = cfg["PriorModel"]
    if pm.get("red_noise_prior", "").strip() != "flat":
        raise SystemExit(f"{ini[0]}: needs red_noise_prior = flat (true uniform)")
    if pm.get("gw_parameterization", "").strip().lower() != "ridge":
        raise SystemExit(f"{ini[0]}: needs gw_parameterization = ridge")

    idata = az.from_netcdf(nc[0])
    post = idata.posterior
    ll = idata.log_likelihood["likelihood"].values
    ll = ll.reshape(ll.shape[0], ll.shape[1], -1).sum(axis=-1)

    def bounds(name):
        return pm.getfloat(f"{name}_min"), pm.getfloat(f"{name}_max")

    efac_lo, efac_hi = bounds("efac")
    eq_lo, eq_hi = bounds("log10_equad")
    # White noise is a Gaussian reparam on the NUTS path: N(mid, (hi - lo)/6).
    efac_z = (post["efac"].values - (efac_lo + efac_hi) / 2) / ((efac_hi - efac_lo) / 6)
    eq_z = (np.log10(post["equad"].values) - (eq_lo + eq_hi) / 2) / ((eq_hi - eq_lo) / 6)
    eps = float(np.unique(np.round(post["orf_epsilon"].values, 12)).item())
    return {
        "run": os.path.basename(run_dir),
        "orf_epsilon": eps,
        "loglike": ll,
        "pivot": post["log10_pivot_psd"].values,
        "prime": post["log10_pivot_psd_prime"].values,
        "gamma_prime": post["log10_gamma_a_prime"].values,
        "pivot_mu": sum(bounds("log10_pivot_psd")) / 2,
        "pivot_sigma": (bounds("log10_pivot_psd")[1] - bounds("log10_pivot_psd")[0]) / 6,
        "sigma_p": post["log10_σp"].values,
        "gamma_p": post["log10_γp"].values,
        "sigma_p_bounds": bounds("log10_sigma_p"),
        "gamma_p_bounds": bounds("log10_gamma_p"),
        "efac_z": efac_z,
        "equad_z": eq_z,
    }


# ---------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------


def mode_totals(run, masks):
    """Per-mode P(M|d), <lnL>_M, ln pi(M), with chain-to-chain standard errors."""
    mu, sig = run["pivot_mu"], run["pivot_sigma"]
    prior_mass = {"low": norm.cdf((LOW - mu) / sig), "high": norm.sf((HIGH - mu) / sig)}
    out = {}
    for m, mask in masks.items():
        per_chain_frac = mask.mean(axis=1)
        per_chain_ll = np.array([run["loglike"][c][mask[c]].mean()
                                 for c in range(mask.shape[0])])
        out[m] = {
            "post_frac": float(mask.mean()),
            "mean_loglike": float(run["loglike"][mask].mean()),
            "ln_prior_mass": float(math.log(prior_mass[m])),
            "per_chain_frac": per_chain_frac.tolist(),
            "per_chain_mean_loglike": per_chain_ll.tolist(),
        }
    n_chain = masks["low"].shape[0]
    lo_c, hi_c = np.array(out["low"]["per_chain_frac"]), np.array(out["high"]["per_chain_frac"])
    ln_ratio_c = np.log(lo_c / hi_c)
    dll_c = (np.array(out["low"]["per_chain_mean_loglike"])
             - np.array(out["high"]["per_chain_mean_loglike"]))
    ln_ratio = math.log(out["low"]["post_frac"] / out["high"]["post_frac"])
    dll = out["low"]["mean_loglike"] - out["high"]["mean_loglike"]
    dlnpi = out["low"]["ln_prior_mass"] - out["high"]["ln_prior_mass"]
    out["ln_post_ratio"] = ln_ratio
    out["ln_post_ratio_se"] = float(ln_ratio_c.std(ddof=1) / math.sqrt(n_chain))
    out["delta_mean_loglike"] = dll
    out["delta_mean_loglike_se"] = float(dll_c.std(ddof=1) / math.sqrt(n_chain))
    out["delta_ln_prior_mass"] = dlnpi
    out["delta_kl_total"] = dll + dlnpi - ln_ratio
    out["delta_kl_total_se"] = float(math.hypot(out["ln_post_ratio_se"],
                                                out["delta_mean_loglike_se"]))
    return out


def mode_blocks(run, mask, mode):
    """Marginal KL of every sampled prior block within one mode.

    Blocks held fixed in the run (zero posterior spread, e.g. white noise fixed at
    the MDC2 truth) carry no prior volume and are skipped.
    """
    chain = np.broadcast_to(np.arange(mask.shape[0])[:, None], mask.shape)[mask]
    n_psr = run["sigma_p"].shape[-1]
    blocks = {}
    rn_box = [run["sigma_p_bounds"], run["gamma_p_bounds"]]
    for i in range(n_psr):
        rn = np.column_stack([run["sigma_p"][..., i][mask], run["gamma_p"][..., i][mask]])
        blocks[f"rn_{i}"] = block_kl(rn, chain, uniform_cross_entropy(rn_box))
        # 1-D marginals say whether the cost sits in amplitude or in corner frequency.
        for j, par in enumerate(("sigma", "gamma")):
            blocks[f"_rn{par}_{i}"] = knn_entropy(rn[:, j], chain)
        wn = np.column_stack([run["efac_z"][..., i][mask], run["equad_z"][..., i][mask]])
        if np.all(wn.std(axis=0) > 1e-6):
            blocks[f"wn_{i}"] = block_kl(wn, chain, std_normal_cross_entropy(wn))
    # GW pair: standard-normal prior truncated to the mode, so add ln pi(M) back.
    mu, sig = run["pivot_mu"], run["pivot_sigma"]
    ln_pm = math.log(norm.cdf((LOW - mu) / sig) if mode == "low"
                     else norm.sf((HIGH - mu) / sig))
    gw = np.column_stack([run["prime"][mask], run["gamma_prime"][mask]])
    blocks["gw"] = block_kl(gw, chain, std_normal_cross_entropy(gw) + ln_pm)
    blocks["_n"] = int(mask.sum())
    return blocks


def part_a(run):
    masks = {"low": run["pivot"] < LOW, "high": run["pivot"] > HIGH}
    totals = mode_totals(run, masks)
    blocks = {m: mode_blocks(run, masks[m], m) for m in masks}
    names = [k for k in blocks["low"] if not k.startswith("_")]
    marg = {}
    for par, (lo_b, hi_b) in (("sigma", run["sigma_p_bounds"]), ("gamma", run["gamma_p_bounds"])):
        n_psr = run["sigma_p"].shape[-1]
        # KL_low - KL_high of a 1-D uniform-prior marginal = H_high - H_low.
        marg[par] = [blocks["high"][f"_rn{par}_{i}"] - blocks["low"][f"_rn{par}_{i}"]
                     for i in range(n_psr)]
    table = []
    for name in names:
        lo, hi = blocks["low"][name], blocks["high"][name]
        table.append({
            "block": name,
            "kl_low_gauss": lo["kl_gauss"], "kl_high_gauss": hi["kl_gauss"],
            "kl_low_knn": lo["kl_knn"], "kl_high_knn": hi["kl_knn"],
            "dkl_gauss": lo["kl_gauss"] - hi["kl_gauss"],
            "dkl_knn": lo["kl_knn"] - hi["kl_knn"],
        })
    table.sort(key=lambda r: -r["dkl_knn"])

    def total(prefix, key):
        return float(sum(r[key] for r in table if r["block"].startswith(prefix)))

    sums = {f"{p}_{k}": total(p, f"dkl_{k}") for p in ("rn", "wn", "gw") for k in ("gauss", "knn")}
    sums["rn_sigma_marginals_knn"] = float(sum(marg["sigma"]))
    sums["rn_gamma_marginals_knn"] = float(sum(marg["gamma"]))
    sums["blocks_knn"] = sums["rn_knn"] + sums["wn_knn"] + sums["gw_knn"]
    sums["blocks_gauss"] = sums["rn_gauss"] + sums["wn_gauss"] + sums["gw_gauss"]
    # dKL_total = sum of block dKL + d(total correlation).
    sums["delta_total_correlation_knn"] = totals["delta_kl_total"] - sums["blocks_knn"]
    return {"totals": totals,
            "n_draws": {m: blocks[m]["_n"] for m in blocks},
            "blocks": table, "rn_marginal_dkl_knn": marg, "sums": sums}


# ---------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------


def restricted_lnb(in_r, below):
    """d lnB(model/noise) = ln P(R|d) - ln P(R|d, noise) with MCMC errors.

    ``in_r`` and ``below`` are boolean (n_chain, n_draw).
    """
    in_r, below = in_r.astype(float), below.astype(float)
    p_all = in_r.mean()
    ess_all = effective_sample_size(in_r) if 0 < p_all < 1 else float(in_r.size)
    n_below = below.mean() * (effective_sample_size(below) if below.mean() < 1 else below.size)
    joint = in_r * below
    p_noise = joint.sum() / max(below.sum(), 1.0)
    # Effective draws inside R: overall, and inside the noise region.
    eff_all = p_all * ess_all
    eff_noise = p_noise * n_below
    out = {"p_r_post": float(p_all), "p_r_noise": float(p_noise),
           "eff_in_r_post": float(eff_all), "eff_in_r_noise": float(eff_noise)}
    if min(eff_all, eff_noise) < MIN_EFFECTIVE or p_noise == 0 or p_all == 0:
        out.update(dlnb=None, dlnb_se=None)
        return out
    se = math.sqrt((1 - p_all) / max(eff_all, 1) + (1 - p_noise) / max(eff_noise, 1))
    out.update(dlnb=float(math.log(p_all) - math.log(p_noise)), dlnb_se=float(se))
    return out


def part_b(run, c, top):
    below = run["prime"] < c
    sp, gp = run["sigma_p"], run["gamma_p"]
    groups = {f"psr{i}": [i] for i in top}
    groups["top3"] = list(top[:3])
    groups[f"top{len(top)}"] = list(top)
    groups["all"] = list(range(sp.shape[-1]))
    g_lo, g_hi = run["gamma_p_bounds"]
    s_lo, s_hi = run["sigma_p_bounds"]
    cuts = ([("gamma_p_max", u) for u in np.arange(g_hi - 0.5, g_lo, -0.5)]
            + [("gamma_p_min", v) for v in np.arange(g_lo + 0.5, g_hi, 0.5)]
            + [("sigma_p_max", w) for w in np.arange(s_hi - 1.0, s_lo, -1.0)])
    rows = [{"group": "none", "cut": "full prior", "value": None,
             **restricted_lnb(np.ones_like(below), below)}]
    for gname, idx in groups.items():
        for kind, val in cuts:
            if kind == "gamma_p_max":
                in_r = np.all(gp[..., idx] <= val, axis=-1)
            elif kind == "gamma_p_min":
                in_r = np.all(gp[..., idx] >= val, axis=-1)
            else:
                in_r = np.all(sp[..., idx] <= val, axis=-1)
            rows.append({"group": gname, "pulsars": idx, "cut": kind,
                         "value": float(val), **restricted_lnb(in_r, below)})
    return rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_report(name, a, b, base_lnb, c):
    t = a["totals"]
    print(f"\n=== {name} ===")
    print(f"  P(low|d) {t['low']['post_frac']:.3f}   P(high|d) {t['high']['post_frac']:.3f}"
          f"   ln ratio {t['ln_post_ratio']:+.2f} ± {t['ln_post_ratio_se']:.2f}")
    print(f"  <lnL> low - high      {t['delta_mean_loglike']:+.2f} ± "
          f"{t['delta_mean_loglike_se']:.2f}")
    print(f"  ln pi(M) low - high   {t['delta_ln_prior_mass']:+.2f}")
    print(f"  => dKL (low - high)   {t['delta_kl_total']:+.2f} ± {t['delta_kl_total_se']:.2f}"
          "   (extra Occam penalty paid by the red-noise mode)")
    s = a["sums"]
    print(f"  block sums dKL (kNN / Gauss), n low {a['n_draws']['low']}, "
          f"high {a['n_draws']['high']}:")
    for p, lab in (("rn", "red noise (33 psr)"), ("wn", "white noise (33 psr)"),
                   ("gw", "GW pair")):
        print(f"    {lab:22s} {s[p + '_knn']:+7.2f} / {s[p + '_gauss']:+7.2f}")
    print(f"      of which 1-D marginals: sigma_p {s['rn_sigma_marginals_knn']:+.2f}, "
          f"gamma_p {s['rn_gamma_marginals_knn']:+.2f}")
    print(f"    {'sum of blocks':22s} {s['blocks_knn']:+7.2f} / {s['blocks_gauss']:+7.2f}")
    print(f"    {'d total correlation':22s} {s['delta_total_correlation_knn']:+7.2f} (kNN)")
    print("  top blocks by dKL (kNN):  block   KL_low   KL_high   dKL (Gauss)")
    for r in a["blocks"][:8]:
        print(f"    {r['block']:8s} {r['kl_low_knn']:7.2f} {r['kl_high_knn']:8.2f} "
              f"{r['dkl_knn']:+7.2f} ({r['dkl_gauss']:+.2f})")
    for r in a["blocks"][-3:]:
        print(f"    {r['block']:8s} {r['kl_low_knn']:7.2f} {r['kl_high_knn']:8.2f} "
              f"{r['dkl_knn']:+7.2f} ({r['dkl_gauss']:+.2f})")
    print_part_b(b, base_lnb, c)


def print_part_b(b, base_lnb, c):
    print(f"  Part B: lnB(model/noise) under restricted red-noise priors "
          f"(base {base_lnb:+.3f}, noise region prime < {c})")
    shown = 0
    for r in b:
        if r["dlnb"] is None:
            continue
        print(f"    {r['group']:6s} {r['cut']:12s} {'' if r['value'] is None else r['value']:>6}"
              f"  dlnB {r['dlnb']:+.2f} ± {r['dlnb_se']:.2f} -> {base_lnb + r['dlnb']:+.2f}"
              f"   (eff in R: post {r['eff_in_r_post']:.0f}, noise {r['eff_in_r_noise']:.0f})")
        shown += 1
    n_ref = sum(r["dlnb"] is None for r in b)
    print(f"    ({shown} shown; {n_ref} refused for < "
          f"{MIN_EFFECTIVE:.0f} effective draws)")


def plot(results, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(results), 2, figsize=(13, 4 * len(results)), squeeze=False)
    for row, (name, res) in enumerate(results.items()):
        blocks = [r for r in res["part_a"]["blocks"] if r["block"].startswith("rn_")]
        blocks.sort(key=lambda r: int(r["block"].split("_")[1]))
        x = np.arange(len(blocks))
        ax = axes[row, 0]
        ax.bar(x - 0.2, [r["dkl_knn"] for r in blocks], 0.4, label="red noise (kNN)")
        wn = sorted([r for r in res["part_a"]["blocks"] if r["block"].startswith("wn_")],
                    key=lambda r: int(r["block"].split("_")[1]))
        if len(wn) == len(blocks):
            ax.bar(x + 0.2, [r["dkl_knn"] for r in wn], 0.4, label="white noise (kNN)")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("pulsar index")
        ax.set_ylabel("KL(low) - KL(high)  [nats]")
        ax.set_title(f"{name}: per-pulsar Occam shift")
        ax.legend(fontsize=8)
        ax = axes[row, 1]
        base = res["base_lnb"]
        for grp in ("top3", "all"):
            for cut, mk in (("gamma_p_max", "o"), ("gamma_p_min", "s")):
                pts = [r for r in res["part_b"] if r["group"] == grp and r["cut"] == cut
                       and r["dlnb"] is not None]
                if pts:
                    ax.errorbar([r["value"] for r in pts], [base + r["dlnb"] for r in pts],
                                yerr=[r["dlnb_se"] for r in pts], marker=mk, capsize=2,
                                label=f"{grp} {cut}")
        ax.axhline(base, color="k", lw=0.5, ls="--", label="full prior")
        ax.set_xlabel("cut on log10 gamma_p")
        ax.set_ylabel("lnB(model / noise)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"  wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True)
    p.add_argument("--lnb", required=True,
                   help="lnb_gw_vs_noise.py JSON (headline threshold and base lnB).")
    p.add_argument("--noise-c", type=float, action="append", default=[],
                   help="Extra noise-region thresholds (pivot prime) for Part B, beyond "
                   "the headline one. Shallower c gives more draws but assumes the "
                   "likelihood is already flat there.")
    p.add_argument("--out", default=None)
    p.add_argument("--plot", default=None)
    args = p.parse_args()

    with open(args.lnb) as f:
        lnb = {r["run"]: r for r in json.load(f)["runs"]}

    results = {}
    for run_dir in args.run:
        run = load_run(run_dir)
        ref = lnb[run["run"]]
        c, base = ref["headline_threshold"], ref["provisional_ln_bayes_factor"]
        a = part_a(run)
        top = [int(r["block"].split("_")[1]) for r in a["blocks"]
               if r["block"].startswith("rn_")][:TOP_PULSARS]
        b = part_b(run, c, top)
        print_report(run["run"], a, b, base, c)
        extra = {}
        for c2 in args.noise_c:
            # The base lnB also moves with c; re-read it from the same posterior.
            base2 = -math.log((run["prime"] < c2).mean() / norm.cdf(c2))
            extra[str(c2)] = {"base_lnb": base2, "rows": part_b(run, c2, top)}
            print_part_b(extra[str(c2)]["rows"], base2, c2)
        results[run["run"]] = {"orf_epsilon": run["orf_epsilon"], "threshold": c,
                               "base_lnb": base, "top_pulsars": top, "part_a": a,
                               "part_b": b, "part_b_extra_thresholds": extra}

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  wrote {args.out}")
    if args.plot:
        plot(results, args.plot)


if __name__ == "__main__":
    main()
