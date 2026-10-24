#!/usr/bin/env python
"""Spec Phase 4, task 4.3: the GW ridge prior, Gaussian vs bounded Uniform, on MDC2.

Each uniform-prior rung (``*_uniformgw_*``) is paired with its Gaussian-prior twin
(``*_gaussgw_*``), run on the same likelihood (slurm_scripts/phase4_gw_prior.sh). Per rung
this reports:

- lnB(GW+noise / noise) by region Savage-Dickey, with its 95% lower bound. The uniform
  rungs use ``lnb_gw_vs_noise.py`` here; the Gaussian rungs use the same script as of
  main 7616aa7, loaded from the old-library worktree (``--old-lib``; create it with
  ``git worktree add --detach /fred/oz022/tkimpson/Argus-phase4-old 7616aa7``, see the
  header of slurm_scripts/phase4_gw_prior.sh), whose prior mass is Phi(c).
- percentiles (2.5, 16, 50, 84, 97.5) of the two-sided pivot log-PSD, the derived
  log10_ha and log10_gamma_a;
- sampling health: max r-hat and min bulk/tail ESS over the sampled sites, divergences;
- wall time of run_analysis.py (``time`` in the SLURM log, so data loading and
  compilation included) and seconds per iteration, with the GPU count and chain
  method. The Gaussian twins were resubmitted on one A100 (queue time),
  so their chains ran sequentially: s/iteration compares, wall time does not.

and, per pair, the change (uniform minus Gaussian). For 1b, where both eps = 0 and eps = 1
exist, the difference of their region estimates is a cross-check of lnB(HD/CURN).

Run (CPU, from the repo root, Argus env):
    python workflows/ng15_sgwb_demo/scripts/phase4_compare_gw_prior.py \
        --out workflows/ng15_sgwb_demo/results/phase4/phase4_uniform_gw_comparison.json
"""

import argparse
import glob
import importlib.util
import json
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OLD_LIB = "/fred/oz022/tkimpson/Argus-phase4-old"
PAIRS = [
    ("1b", 0.0, "mdc2_d1_flat_uniformgw_eps000", "mdc2_d1_flat_gaussgw_eps000"),
    ("1b", 1.0, "mdc2_d1_flat_uniformgw_eps100", "mdc2_d1_flat_gaussgw_eps100"),
    ("2b", 0.0, "mdc2_flat_uniformgw_eps000", "mdc2_flat_gaussgw_eps000"),
]
PCTS = (2.5, 16, 50, 84, 97.5)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(path))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return mod


def _percentiles(x):
    return dict(zip([str(p) for p in PCTS], np.percentile(x, PCTS).round(4).tolist()))


def _timing(run_id, logdir):
    """run_analysis.py wall time (s) and s/iteration from the newest SLURM log of a rung."""
    logs = []
    for path in glob.glob(os.path.join(logdir, "p4_gwprior_*.out")):
        with open(path, errors="replace") as f:
            text = f.read()
        if f"=== env check ({run_id}) ===" in text:
            logs.append((os.path.getmtime(path), path, text))
    if not logs:
        return None
    _, path, text = max(logs)
    real = re.findall(r"^real\s+(\d+)m([\d.]+)s", text, flags=re.M)
    rates = [float(r) for r in re.findall(r"([\d.]+)s/it[\],]", text)]
    out = {
        "log": os.path.basename(path),
        "n_gpus": len(re.findall(r"^GPU \d+: ", text, flags=re.M)),
        # chains run in parallel only with >= num_chains devices (bayesian_inference)
        "chain_method": "sequential" if "chains sequentially" in text else "parallel",
    }
    if real:
        m, s = real[-1]
        out["wall_s"] = 60 * int(m) + float(s)
    if rates:
        out["median_s_per_it"] = float(np.median(rates))
    return out


def summarise(run_id, gvn, outputs, logdir):
    import arviz as az

    run_dir = os.path.join(outputs, run_id)
    nc = os.path.join(run_dir, f"{run_id}_results.nc")
    idata = az.from_netcdf(nc)
    post = idata.posterior
    run = gvn.load_run(run_dir)
    if "box" in run:  # uniform-prior script; thresholds are pivot values
        res = gvn.analyse(run["pivot"], run["box"], run["gamma"], run["gamma_box"])
        to_pivot = float
    else:  # the 7616aa7 script (Gaussian pivot prior); thresholds in prior sigmas
        res = gvn.analyse(run["prime"], run["gamma_prime"])

        def to_pivot(c):
            return run["pivot_mu"] + c * run["pivot_sigma"]

    sampled = [
        v
        for v in post.data_vars
        if v in ("log10_pivot_psd", "log10_gamma_a")
        or v.endswith("_prime")
        or v in ("log10_γp", "log10_σp")
    ]
    summ = az.summary(idata, var_names=sampled, kind="diagnostics")
    with open(os.path.join(run_dir, "provenance.json")) as f:
        prov = json.load(f)
    return {
        "run": run_id,
        "nc": os.path.relpath(nc, ROOT),
        "library_sha": prov.get("git_sha"),
        "config_sha256": prov.get("config_sha256"),
        "lnb_gw_vs_noise": res["ln_bayes_factor"],
        "lnb_gw_vs_noise_uncert": res["uncert"],
        "lnb_reliable": res["reliable"],
        "lnb_failed_diagnostics": res["failed_diagnostics"],
        "lnb_provisional": res["provisional_ln_bayes_factor"],
        "lnb_provisional_uncert": res["provisional_uncert"],
        "lnb_lower_95": res["lower_bound"]["ln_bayes_factor_lower"],
        "lnb_threshold_pivot": (
            to_pivot(res["headline_threshold"])
            if res["headline_threshold"] is not None
            else None
        ),
        "lnb_lower_threshold_pivot": to_pivot(res["lower_bound"]["c"]),
        "pivot_two_sided": _percentiles(post["log10_pivot_psd"].values),
        "log10_ha": _percentiles(post["log10_ha"].values),
        "log10_gamma_a": _percentiles(post["log10_gamma_a"].values),
        "rhat_max": float(summ["r_hat"].max()),
        "ess_bulk_min": float(summ["ess_bulk"].min()),
        "ess_tail_min": float(summ["ess_tail"].min()),
        "divergences": int(idata.sample_stats["diverging"].values.sum()),
        "n_draws": int(post.sizes["chain"] * post.sizes["draw"]),
        "timing": _timing(run_id, logdir),
    }


def _delta(new, old):
    out = {}
    for key in ("pivot_two_sided", "log10_ha", "log10_gamma_a"):
        out[key] = {
            p: round(new[key][p] - old[key][p], 4) for p in new[key] if p in old[key]
        }
    for key in ("lnb_gw_vs_noise", "lnb_provisional", "lnb_lower_95"):
        if new[key] is not None and old[key] is not None:
            out[key] = new[key] - old[key]
    tn, to = new["timing"] or {}, old["timing"] or {}
    if "median_s_per_it" in tn and "median_s_per_it" in to:
        out["s_per_it_ratio"] = tn["median_s_per_it"] / to["median_s_per_it"]
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--outputs", default=os.path.join(ROOT, "outputs"))
    p.add_argument("--old-lib", default=OLD_LIB)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    gvn_new = _load(os.path.join(HERE, "lnb_gw_vs_noise.py"), "gvn_new")
    old_script = os.path.join(
        args.old_lib, "workflows/ng15_sgwb_demo/scripts/lnb_gw_vs_noise.py"
    )
    gvn_old = _load(old_script, "gvn_old")
    logdir = os.path.join(args.outputs, "logfiles")

    rungs = []
    for dataset, eps, new_id, old_id in PAIRS:
        new = summarise(new_id, gvn_new, args.outputs, logdir)
        old = summarise(old_id, gvn_old, args.outputs, logdir)
        rungs.append(
            {
                "dataset": dataset,
                "orf_epsilon": eps,
                "uniform": new,
                "gaussian": old,
                "delta_uniform_minus_gaussian": _delta(new, old),
            }
        )
        print(f"\n=== {dataset} eps={eps} ===")
        for tag, r in (("gaussian", old), ("uniform", new)):
            lnb = (
                f"{r['lnb_gw_vs_noise']:+.3f} ± {r['lnb_gw_vs_noise_uncert']:.3f}"
                if r["lnb_reliable"]
                else "unreliable"
            )
            pv = r["pivot_two_sided"]
            print(
                f"  {tag:8s} lnB(GW/noise) {lnb:>16s}  lower95 {r['lnb_lower_95']:+.2f}  "
                f"pivot 2.5/50/97.5 {pv['2.5']:+.2f}/{pv['50']:+.2f}/{pv['97.5']:+.2f}  "
                f"rhat {r['rhat_max']:.3f} div {r['divergences']}"
            )

    hd_curn = {}
    for tag in ("uniform", "gaussian"):
        ends = {
            r["orf_epsilon"]: r[tag]
            for r in rungs
            if r["dataset"] == "1b" and r[tag]["lnb_provisional"] is not None
        }
        if 0.0 in ends and 1.0 in ends:
            hd_curn[tag] = {
                "region_sd_difference": ends[1.0]["lnb_provisional"]
                - ends[0.0]["lnb_provisional"],
                "uncert": float(
                    np.hypot(
                        ends[1.0]["lnb_provisional_uncert"],
                        ends[0.0]["lnb_provisional_uncert"],
                    )
                ),
            }

    summary = {
        "spec_task": "4.3",
        "issue": 125,
        "script_sha": subprocess.run(
            ["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "old_script": old_script,
        "pivot_convention": "two-sided OU density at f = 1/(5 yr); one-sided is +0.301",
        "rungs": rungs,
        "hd_curn_region_sd_1b": hd_curn,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
