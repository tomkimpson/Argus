#!/usr/bin/env python
"""Static CURN lnB(GW / noise-only) on MDC2 1b for GW-model x red-noise-prior pairs. (CPU)

Why this exists
---------------
``diag_spectral_prior_volume.py`` showed the red-noise prior is not why Argus gets
lnB(GW / noise) ~ 0 on MDC2 1b where Hazboun et al. get +3.1: with the GW fixed at the
truth, auto-power alone gives about +8 under either red-noise prior. The deficit is on the
GW side. This script puts the GW's own parameters back in and integrates them over their
prior, for each combination of

    GW model:  pl13  enterprise power law, gamma = 13/3,     log10 A ~ U[-18, -11]
               pl    enterprise power law,                   log10 A ~ U[-18, -11], gamma ~ U[0, 7]
               ou    Argus OU (ridge), uniform prior         pivot ~ U[-13, -5], log10 gamma_a ~ U[-11, -6]
               ou_nuts  the same grid under the prior NUTS actually uses: Gaussians
                        N(mid, (hi - lo)/6) on both ridge coordinates
    red noise: ou (Argus) or pl (enterprise), as in diag_spectral_prior_volume.py

The CURN evidence factorises over pulsars given the GW parameters, so

    lnB = ln sum_g w_g exp( sum_i ln Z_i(g) ) - sum_i ln Z_i(no GW),

with Z_i(g) the pulsar's red-noise-marginal evidence (2-D grid) for GW point g. This is
CURN, not HD: only auto-power enters. Compare with Hazboun +3.1 (CURN) and Argus -0.47.

If pl/pl gives ~+3 and ou_nuts/ou gives ~0, this static calculation reproduces both
numbers and the GW model (shape or prior) is the gap; switching one factor at a time
then attributes it. If every combination gives ~+3, the gap is in Argus's likelihood or
sampler, not the model.

    python scripts/diag_curn_static_lnb.py --data data/mdc2_d1_all \
        --truth-noise ../data/IPTA_MockDataChallenge2/group1_psr_noise.json \
        --log10-a -15.18045606445813 --workers 32 \
        --out outputs/diag_curn_static_lnb_mdc2_d1.json
"""

import argparse
import glob
import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_spectral_prior_volume as spv  # noqa: E402
from inject_powerlaw_gwb import powerlaw_psd  # noqa: E402

SEC_PER_YEAR = 365.25 * 86400.0
PIVOT_W = 2 * math.pi / (5.0 * SEC_PER_YEAR)  # Argus ridge pivot, f = 1/(5 yr)
PL_GW = {"amp": (-18.0, -11.0), "gamma": (0.0, 7.0)}
OU_GW = {"pivot": (-13.0, -5.0), "lga": (-11.0, -6.0)}
OU_PIVOT_GRID = (-16.0, -5.0)
FAILED = [0]  # per-process count of non-PD GW points  # wider than the uniform box, for the Gaussian prior's tail


def rng(lo, hi, step):
    return np.linspace(lo, hi, int(round((hi - lo) / step)) + 1)


def gw_grids(step_amp, step_shape):
    amp = rng(*PL_GW["amp"], step_amp)
    gam = rng(*PL_GW["gamma"], step_shape)
    piv = rng(*OU_PIVOT_GRID, step_amp)
    lga = rng(*OU_GW["lga"], step_shape)
    return {"pl13": {"amp": amp}, "pl": {"amp": amp, "gamma": gam},
            "ou": {"pivot": piv, "lga": lga}}


def trapezoid(x):
    w = np.ones_like(x)
    w[0] = w[-1] = 0.5
    return w * (x[1] - x[0])


def gw_log_weights(grids):
    """Normalised log prior weights on each GW grid, keyed by GW model."""
    out = {}
    a = grids["pl13"]["amp"]
    out["pl13"] = np.log(trapezoid(a) / trapezoid(a).sum())
    g = grids["pl"]["gamma"]
    w = np.outer(trapezoid(g), trapezoid(a))
    out["pl"] = np.log(w / w.sum())
    piv, lga = grids["ou"]["pivot"], grids["ou"]["lga"]
    # Uniform pivot box inside the wider grid: trapezoid weights with half-weight ends.
    idx = np.flatnonzero((piv >= OU_GW["pivot"][0] - 1e-9) & (piv <= OU_GW["pivot"][1] + 1e-9))
    wp = np.zeros_like(piv)
    wp[idx] = trapezoid(piv[idx])
    w = np.outer(trapezoid(lga), wp)
    with np.errstate(divide="ignore"):
        out["ou"] = np.log(w / w.sum())
    # NUTS prior: N(mid, (hi - lo)/6) on both ridge coordinates (prior_models._reparam).
    def gauss(x, lo, hi):
        return norm.pdf(x, 0.5 * (lo + hi), (hi - lo) / 6.0) * trapezoid(x)
    w = np.outer(gauss(lga, *OU_GW["lga"]), gauss(piv, *OU_GW["pivot"]))
    out["ou_nuts"] = np.log(w / w.sum())
    out["ou_nuts_mass_on_grid"] = float(
        (norm.cdf(OU_PIVOT_GRID[1], -9.0, 8 / 6) - norm.cdf(OU_PIVOT_GRID[0], -9.0, 8 / 6)))
    return out


def ou_gw_unit(t, lga):
    """Argus OU GW covariance per unit TWO-sided pivot PSD (10**pivot = 1)."""
    ga = 10.0**lga
    norm_ = PIVOT_W**2 * (ga**2 + PIVOT_W**2)
    return spv.ou_covariance(t, lga, 1.0) * norm_


def psd_clip(c):
    """Symmetrise and zero negative eigenvalues: roundoff in a projected steep-spectrum
    covariance leaves negative eigenvalues of order 1e-16 x its largest, which a large
    amplitude scales past the white noise and makes the base non-positive-definite."""
    lam, v = np.linalg.eigh(0.5 * (c + c.T))
    return (v * np.clip(lam, 0.0, None)) @ v.T


def rn_lnz(base, shapes, amp_grid, y):
    """Red-noise-marginal ln Z of y for one base covariance (data likelihood only)."""
    amps = 10.0 ** (2.0 * amp_grid)
    lnl = np.empty((len(shapes), amp_grid.size))
    try:
        for k, u in enumerate(shapes):
            lnl[k], _ = spv.amplitude_line(base, u, amps, y, None)
    except np.linalg.LinAlgError:
        # Only at GW power ~1e10 x white, where roundoff beats the white floor; the data
        # exclude such points by hundreds of nats, so scoring them -inf changes nothing.
        FAILED[0] += 1
        return -np.inf
    return spv.log_evidence(lnl)


def analyse_pulsar(job):
    FAILED[0] = 0
    psr, t_array, args, grids, rn_grids = job
    G, t = psr["G"], psr["toas"]

    def proj(c):
        return G.T @ c @ G

    w = proj(np.diag(psr["white"]))
    y = G.T @ psr["residuals"]
    span = t.max() - t.min()
    kern = spv.StationaryKernel(t, span)
    fmat, ff = spv.fourier_basis(t, t_array, args.n_freqs)
    rn_shapes = {
        "ou": [proj(spv.ou_covariance(t, s, psr["f0"])) for s in rn_grids["ou"]["shape"]],
        "pl": [proj(spv.pl_covariance(fmat, ff, t_array, s)) for s in rn_grids["pl"]["shape"]],
    }
    out = {"name": psr["name"], "noise": {}, "pl13": {}, "pl": {}, "ou": {}}
    for rp in ("ou", "pl"):
        out["noise"][rp] = rn_lnz(w, rn_shapes[rp], rn_grids[rp]["amp"], y)

    # GW unit covariances. The stationary GW integral is linear in A^2 at fixed gamma.
    pl_units = {g: psd_clip(proj(kern(lambda f, g=g: powerlaw_psd(f, 0.0, g))))
                for g in np.append(grids["pl"]["gamma"], 13.0 / 3.0)}
    ou_units = {lg: psd_clip(proj(ou_gw_unit(t, lg))) for lg in grids["ou"]["lga"]}

    for rp in ("ou", "pl"):
        sh, ag = rn_shapes[rp], rn_grids[rp]["amp"]
        u13 = pl_units[13.0 / 3.0]
        out["pl13"][rp] = [rn_lnz(w + 10.0 ** (2 * a) * u13, sh, ag, y)
                           for a in grids["pl13"]["amp"]]
        out["pl"][rp] = [[rn_lnz(w + 10.0 ** (2 * a) * pl_units[g], sh, ag, y)
                          for a in grids["pl"]["amp"]] for g in grids["pl"]["gamma"]]
        out["ou"][rp] = [[rn_lnz(w + 10.0**p * ou_units[lg], sh, ag, y)
                          for p in grids["ou"]["pivot"]] for lg in grids["ou"]["lga"]]
    out["n_non_pd"] = FAILED[0]
    print(f"  done {psr['name']} (non-PD GW points scored -inf: {FAILED[0]})", flush=True)
    return out


def combine(per_psr, logw):
    """lnB(GW / noise) and the GW posterior for each GW model x red-noise prior."""
    res = {}
    for rp in ("ou", "pl"):
        noise = sum(p["noise"][rp] for p in per_psr)
        for gm, key in (("pl13", "pl13"), ("pl", "pl"), ("ou", "ou"), ("ou_nuts", "ou")):
            tot = sum(np.asarray(p[key][rp]) for p in per_psr)
            lp = logw[gm] + tot
            lnz = float(logsumexp(lp[np.isfinite(lp)]))
            post = np.exp(lp - lnz)
            res[f"{gm}|{rp}"] = {"lnB": lnz - noise, "max_lnl_gain": float(tot.max() - noise),
                                 "post": post}
    return res


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--truth-noise", required=True)
    p.add_argument("--log10-a", type=float, required=True, help="Injected GWB log10 A.")
    p.add_argument("--n-freqs", type=int, default=30)
    p.add_argument("--gw-step-amp", type=float, default=0.2)
    p.add_argument("--gw-step-shape", type=float, default=0.5)
    p.add_argument("--rn-step-amp", type=float, default=0.1)
    p.add_argument("--rn-step-shape", type=float, default=0.25)
    p.add_argument("--max-pulsars", type=int, default=None, help="Smoke test only.")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    with open(args.truth_noise) as f:
        truth = json.load(f)
    files = sorted(glob.glob(os.path.join(args.data, "*.feather")))
    psrs = [spv.load_pulsar(f, truth) for f in files]
    t_array = max(q["toas"].max() for q in psrs) - min(q["toas"].min() for q in psrs)
    if args.max_pulsars:
        psrs = psrs[: args.max_pulsars]
    grids = gw_grids(args.gw_step_amp, args.gw_step_shape)
    rn_grids = spv.make_grids(args.rn_step_amp, args.rn_step_shape, args.rn_step_shape)
    logw = gw_log_weights(grids)

    jobs = [(q, t_array, args, grids, rn_grids) for q in psrs]
    if args.workers > 1:
        with Pool(args.workers) as pool:
            per_psr = pool.map(analyse_pulsar, jobs, chunksize=1)
    else:
        per_psr = [analyse_pulsar(j) for j in jobs]

    res = combine(per_psr, logw)
    print(f"\nStatic CURN lnB(GW / noise-only), {len(psrs)} pulsars "
          f"(Hazboun CURN +3.1, Argus CURN -0.47):")
    print("  GW model   red noise   lnB      max lnL gain   posterior peak")
    summary = {}
    for k, r in res.items():
        gm, rp = k.split("|")
        post = r["post"]
        if gm == "pl13":
            peak = f"log10A {grids['pl13']['amp'][np.argmax(post)]:.1f}"
        elif gm == "pl":
            i, j = np.unravel_index(np.argmax(post), post.shape)
            peak = f"gamma {grids['pl']['gamma'][i]:.1f}, log10A {grids['pl']['amp'][j]:.1f}"
        else:
            i, j = np.unravel_index(np.argmax(post), post.shape)
            peak = f"lga {grids['ou']['lga'][i]:.1f}, pivot {grids['ou']['pivot'][j]:.1f}"
        print(f"  {gm:8s}   {rp:9s}   {r['lnB']:+6.2f}   {r['max_lnl_gain']:+8.2f}       {peak}")
        summary[k] = {"lnB": r["lnB"], "max_lnl_gain": r["max_lnl_gain"], "peak": peak}

    if args.out:
        out = {
            "log10_a_injected": args.log10_a, "n_pulsars": len(psrs),
            "priors": {"pl_gw": PL_GW, "ou_gw": OU_GW, "ou_pivot_grid": OU_PIVOT_GRID,
                       "rn_ou": spv.OU_PRIOR, "rn_pl": spv.PL_PRIOR},
            "steps": {"gw_amp": args.gw_step_amp, "gw_shape": args.gw_step_shape,
                      "rn_amp": args.rn_step_amp, "rn_shape": args.rn_step_shape},
            "ou_nuts_mass_on_grid": logw["ou_nuts_mass_on_grid"],
            "summary": summary,
            "per_pulsar_noise_lnz": {q["name"]: q["noise"] for q in per_psr},
            "per_pulsar_non_pd_points": {q["name"]: q["n_non_pd"] for q in per_psr},
        }
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        np.savez(os.path.splitext(args.out)[0] + ".npz",
                 **{k.replace("|", "__"): r["post"] for k, r in res.items()},
                 **{f"grid_{gm}_{ax}": v for gm, g in grids.items() for ax, v in g.items()})
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
