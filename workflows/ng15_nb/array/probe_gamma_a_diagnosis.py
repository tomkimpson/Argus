#!/usr/bin/env python
"""Why does log10 γa sit on its −6 ceiling in the Phase 5 probes? (spec §12, 2026-11-13)

    python workflows/ng15_nb/array/probe_gamma_a_diagnosis.py <output_dir> \
        [--out workflows/ng15_nb/results/array/<output_id>_gamma_a_diagnosis.json]

Reads `<id>_results.nc` or `<id>_partial.nc` with h5py (arviz on the login node times
out on these files) and the checkpoint's adapted mass matrix. Reports, per chain and
pooled:
- the γa trace shape: median, sd, range, lag-1 autocorrelation, share of draws within
  0.5 dex of the ceiling, and the first/last ten draws (is it two modes, or one spike
  with a tail?);
- lnL against γa: per-chain correlation and the mean lnL in γa bins (does the ceiling fit
  better?);
- which pulsars' red noise co-moves with γa: the top pulsars by max-over-chains
  |corr(γa, white-like state)|, where the state is σp above `--on-threshold` and γp above
  `--flat-threshold`, each with per-chain state fraction, switch count, and γa conditional
  on the state (per chain, since a chain that stays in one state cannot show the coupling);
- the split of per-pulsar medians between the chain groups given by `--groups`
  (default 0,3 vs 1,2: the γa-high and γa-low chains of probe 2). This ranking mixes chain
  identity with the effect; use the per-chain correlations above to name drivers;
- classic split r̂ (two halves per chain) for γa, the pivot, and every γp/σp site, with
  the worst sites named;
- a preview of a lower γa ceiling (`--preview-ceiling`): the draws with γa below it, who
  owns the excess there, and the top pulsar's on-fraction by γa bin;
- what the probe ran: target_accept, max_tree_depth, dense blocks (from the ini and the
  checkpoint's inverse-mass shapes), the final iteration's step count per chain, and
  whether EFAC/EQUAD stayed fixed.
"""

import argparse
import configparser
import json
import pickle
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "single_psr"))
import common  # noqa: E402  (manifest order = the array's pulsar order)


def git_sha(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


def split_rhat(x):
    """Classic split r̂ on a (chains, draws) array: each chain cut into two halves."""
    c, n = x.shape
    h = n // 2
    xs = np.concatenate([x[:, :h], x[:, h : 2 * h]], 0)
    W = xs.var(1, ddof=1).mean()
    B = h * xs.mean(1).var(ddof=1)
    return float(np.sqrt(((h - 1) / h * W + B / h) / W))


def lag1(x):
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--out")
    ap.add_argument(
        "--ceiling", type=float, default=-6.0, help="log10 γa prior ceiling"
    )
    ap.add_argument(
        "--near", type=float, default=0.5, help="dex counted as 'at the ceiling'"
    )
    ap.add_argument(
        "--on-threshold",
        type=float,
        default=-16.0,
        help="log10 σp above which a pulsar's red noise counts as 'on'",
    )
    ap.add_argument(
        "--flat-threshold",
        type=float,
        default=-6.0,
        help="log10 γp above which a pulsar's OU is flat (f⁻²) across the band",
    )
    ap.add_argument("--groups", default="0,3:1,2", help="chain groups, e.g. 0,3:1,2")
    ap.add_argument(
        "--preview-ceiling",
        type=float,
        default=-7.2,
        help="preview a lower γa ceiling by restricting the draws to γa ≤ this",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    run_id = out_dir.name
    nc = out_dir / f"{run_id}_results.nc"
    if not nc.exists():
        nc = out_dir / f"{run_id}_partial.nc"
    if not nc.exists():
        raise SystemExit(f"no results or partial .nc in {out_dir}")
    names = common.pulsars()

    with h5py.File(nc, "r") as f:
        post = f["posterior"]
        ga = post["log10_gamma_a"][...]
        pv = post["log10_pivot_psd"][...]
        gp = post["log10_γp"][...]
        sp = post["log10_σp"][...]
        lnl = f["log_likelihood"]["likelihood"][...]
        efac = post["efac"][...]
        equad = post["equad"][...]
    n_chains, n_draws, n_psr = gp.shape
    assert n_psr == len(names), (n_psr, len(names))
    if lnl.ndim > 2:
        lnl = lnl.reshape(n_chains, n_draws, -1).sum(-1)

    # --- γa trace shape -------------------------------------------------------------
    near = ga > args.ceiling - args.near
    chains = []
    for c in range(n_chains):
        x = ga[c]
        chains.append(
            {
                "chain": c,
                "gamma_a_median": float(np.median(x)),
                "gamma_a_sd": float(x.std()),
                "gamma_a_min": float(x.min()),
                "gamma_a_max": float(x.max()),
                "gamma_a_lag1_autocorr": lag1(x),
                "gamma_a_frac_near_ceiling": float(near[c].mean()),
                "gamma_a_first10": np.round(x[:10], 2).tolist(),
                "gamma_a_last10": np.round(x[-10:], 2).tolist(),
                "corr_gamma_a_lnL": float(np.corrcoef(x, lnl[c])[0, 1]),
                "corr_gamma_a_pivot": float(np.corrcoef(x, pv[c])[0, 1]),
                "lnL_median": float(np.median(lnl[c])),
                "lnL_sd": float(lnl[c].std()),
            }
        )
    edges = [-11, -9, -7.5, -6.5, args.ceiling]
    lnl_bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ga >= lo) & (ga < hi)
        lnl_bins.append(
            {
                "gamma_a_bin": [lo, hi],
                "n": int(m.sum()),
                "lnL_mean": float(lnl[m].mean()) if m.any() else None,
            }
        )

    # --- which pulsars co-move with γa -------------------------------------------
    # A pulsar's red noise is in its "white-like" state when it is on (σp above the
    # threshold) and flat in band (γp above `--flat-threshold`). The coupling to γa is
    # reported PER CHAIN: a chain that stays in one state cannot show it, so pooled numbers
    # mix chain identity with the draw-level effect.
    white = (sp > args.on_threshold) & (gp > args.flat_threshold)

    def chain_corr(i):
        out = []
        for c in range(n_chains):
            w = white[c, :, i].astype(float)
            out.append(float(np.corrcoef(ga[c], w)[0, 1]) if w.std() > 0 else 0.0)
        return out

    corr_w = np.array([chain_corr(i) for i in range(n_psr)])  # (n_psr, n_chains)
    score = np.abs(corr_w).max(1)
    picks = [int(i) for i in np.argsort(-score)[:4]]

    def cond(c_mask, w_mask):
        m = c_mask & w_mask
        return {
            "n": int(m.sum()),
            "gamma_a_median": float(np.median(ga[m])) if m.any() else None,
            "frac_near_ceiling": float(near[m].mean()) if m.any() else None,
        }

    def driver_of(i):
        w = white[:, :, i]
        per_chain = []
        for c in range(n_chains):
            cm = np.zeros_like(w, dtype=bool)
            cm[c] = True
            per_chain.append(
                {
                    "chain": c,
                    "white_frac": float(w[c].mean()),
                    "switches": int(np.abs(np.diff(w[c].astype(int))).sum()),
                    "corr_gamma_a_white": float(corr_w[i, c]),
                    "sigma_p_median": float(np.median(sp[c, :, i])),
                    "gamma_p_median": float(np.median(gp[c, :, i])),
                    "gamma_a_given_white": cond(cm, w),
                    "gamma_a_given_not_white": cond(cm, ~w),
                }
            )
        return {
            "pulsar": names[i],
            "index": i,
            "max_abs_corr_over_chains": float(score[i]),
            "per_chain": per_chain,
            "pooled_gamma_a_given_white": cond(np.ones_like(w, bool), w),
            "pooled_gamma_a_given_not_white": cond(np.ones_like(w, bool), ~w),
            "pooled_note": "pooled numbers mix chain identity with the draw-level effect",
        }

    drivers = [driver_of(i) for i in picks]
    drivers_joint = {
        "white_state": f"log10 σp > {args.on_threshold} and log10 γp > {args.flat_threshold}",
        "per_chain_frac_near_ceiling": near.mean(1).tolist(),
    }

    # --- lever A preview: the posterior restricted to γa ≤ --preview-ceiling -------------
    # The γa prior is uniform, so a run that only lowers the ceiling has v2's posterior
    # restricted to the subset. Report who owns the excess there and how the top pulsar's
    # on-fraction depends on γa.
    top = picks[0]
    w_top = white[:, :, top]
    sub = ga <= args.preview_ceiling
    edges = [-11, -9, -8.5, -8, -7.5, -7.2, -7, -6.5, -6.25, args.ceiling + 1e-9]
    preview = {
        "ceiling": args.preview_ceiling,
        "n_draws": int(sub.sum()),
        "share_from_chains": [
            float(sub[c].sum() / max(sub.sum(), 1)) for c in range(n_chains)
        ],
        "per_chain": [
            {
                "chain": c,
                "n": int(sub[c].sum()),
                "top_on_frac": float(w_top[c][sub[c]].mean()) if sub[c].any() else None,
                "lnL_subset_minus_rest": (
                    float(lnl[c][sub[c]].mean() - lnl[c][~sub[c]].mean())
                    if sub[c].any() and (~sub[c]).any()
                    else None
                ),
            }
            for c in range(n_chains)
        ],
        "top_pulsar": names[top],
        "top_on_frac_in_subset": float(w_top[sub].mean()) if sub.any() else None,
        "top_on_frac_outside": float(w_top[~sub].mean()) if (~sub).any() else None,
        "top_off_in_subset": {
            "n": int((sub & ~w_top).sum()),
            "lnL_mean": (
                float(lnl[sub & ~w_top].mean()) if (sub & ~w_top).any() else None
            ),
        },
        "top_on_in_subset": {
            "n": int((sub & w_top).sum()),
            "lnL_mean": float(lnl[sub & w_top].mean()) if (sub & w_top).any() else None,
        },
        "top_on_frac_by_gamma_a_bin": [
            {
                "bin": [lo, hi],
                "n": int(m.sum()),
                "on_frac": float(w_top[m].mean()) if m.any() else None,
            }
            for lo, hi in zip(edges[:-1], edges[1:])
            for m in [(ga >= lo) & (ga < hi)]
        ],
        "sigma_p_in_subset": [
            {
                "pulsar": names[i],
                "q05_50_95": np.round(
                    np.quantile(sp[:, :, i][sub], [0.05, 0.5, 0.95]), 1
                ).tolist(),
                "frac_on": float((sp[:, :, i][sub] > args.on_threshold).mean()),
                "frac_on_all_draws": float((sp[:, :, i] > args.on_threshold).mean()),
            }
            for i in picks
        ],
    }

    # --- chain-group split of per-pulsar medians ------------------------------------
    g1, g2 = [[int(k) for k in g.split(",")] for g in args.groups.split(":")]
    splits = {}
    for arr, lab in ((sp, "sigma_p"), (gp, "gamma_p")):
        d = np.abs(
            np.median(arr[g1].reshape(-1, n_psr), 0)
            - np.median(arr[g2].reshape(-1, n_psr), 0)
        )
        splits[lab] = [
            {"pulsar": names[i], "abs_median_diff": float(d[i])}
            for i in np.argsort(-d)[:8]
        ]

    # --- split r̂ ---------------------------------------------------------------------
    r_gp = np.array([split_rhat(gp[:, :, i]) for i in range(n_psr)])
    r_sp = np.array([split_rhat(sp[:, :, i]) for i in range(n_psr)])
    rhat = {
        "method": f"classic split r-hat on {n_draws // 2}-draw halves (not the readout's)",
        "gamma_a": split_rhat(ga),
        "pivot": split_rhat(pv),
        "n_gamma_p_over_1.1": int((r_gp > 1.1).sum()),
        "n_sigma_p_over_1.1": int((r_sp > 1.1).sum()),
        "worst_gamma_p": [
            {
                "pulsar": names[i],
                "rhat": float(r_gp[i]),
                "chain_medians": np.round(np.median(gp[:, :, i], 1), 2).tolist(),
            }
            for i in np.argsort(-r_gp)[:5]
        ],
        "worst_sigma_p": [
            {
                "pulsar": names[i],
                "rhat": float(r_sp[i]),
                "chain_medians": np.round(np.median(sp[:, :, i], 1), 2).tolist(),
            }
            for i in np.argsort(-r_sp)[:5]
        ],
    }

    # --- what ran ---------------------------------------------------------------------
    ran = {}
    inis = sorted(out_dir.glob("*.ini"))
    if inis:
        cfg = configparser.ConfigParser()
        cfg.read(inis[0])
        for sec in cfg.sections():
            for k in (
                "target_accept_prob",
                "max_tree_depth",
                "dense_mass",
                "dense_mass_blocks",
                "num_warmup",
                "num_samples",
            ):
                if cfg.has_option(sec, k):
                    ran[k] = cfg.get(sec, k)
        ran["ini"] = str(inis[0])
    ck = out_dir / f"{run_id}_checkpoint.pkl"
    if ck.exists():
        try:
            state = pickle.load(open(ck, "rb"))["sampler_state"]
            imm = state.adapt_state.inverse_mass_matrix
            ran["inverse_mass_blocks"] = {
                str(k): list(np.asarray(v).shape) for k, v in imm.items()
            }
            # The final iteration's tree size and running mean acceptance, per chain.
            ran["final_num_steps"] = np.asarray(state.num_steps).tolist()
            ran["mean_accept_prob"] = np.round(
                np.asarray(state.mean_accept_prob), 3
            ).tolist()
        except Exception as e:  # noqa: BLE001
            ran["inverse_mass_blocks"] = f"unreadable: {e!r}"
    ran["efac_fixed"] = bool(np.all(efac == efac.flat[0]))
    ran["equad_fixed"] = bool(np.all(equad == equad.flat[0]))
    ran["efac_value"] = float(efac.flat[0])
    ran["equad_value"] = float(equad.flat[0])

    report = {
        "run": run_id,
        "nc": str(nc),
        "diagnosis_sha": git_sha(HERE),
        "n_chains": n_chains,
        "n_draws_per_chain": n_draws,
        "gamma_a_ceiling": args.ceiling,
        "gamma_a_pooled_quantiles_05_25_50_75_95": np.round(
            np.quantile(ga, [0.05, 0.25, 0.5, 0.75, 0.95]), 2
        ).tolist(),
        "gamma_a_pooled_frac_near_ceiling": float(near.mean()),
        "chains": chains,
        "lnL_by_gamma_a_bin": lnl_bins,
        "drivers": drivers,
        "drivers_joint": drivers_joint,
        "chain_group_splits": {"groups": [g1, g2], **splits},
        "lever_a_preview": preview,
        "split_rhat": rhat,
        "ran": ran,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
