#!/usr/bin/env python
"""Is "red noise imitates the GW" cheaper under the OU prior than the power-law one? (CPU)

Why this exists
---------------
On MDC2 1b Argus gets lnB(GW+noise / noise-only) ~ 0 (CURN -0.47, HD +0.14) where
Hazboun et al. get +3.1 / +3.7. The Occam budget (``diag_occam_budget.py``) showed the
two pivot modes balance: the red-noise mode fits ~6-8 nats better and pays the same in
prior volume, mostly in sigma_p, on J1909, J1939, J1600 and J1713. The hypothesis
tested here: almost every OU red-noise prior draw is a red, GW-like spectrum, so
explaining the GWB as red noise is cheap; enterprise's power law with gamma ~ U[0, 7]
spends most of its volume on flatter spectra, so the same imitation is expensive.

The test is each pulsar's contribution to lnB(GW / noise) under each red-noise prior,
everything else fixed. With white noise at MDC2 truth and, in the GW model, the GWB
auto-power fixed at the injected power law,

    B_i(P) = ln Z_i(W + GW + R_P) - ln Z_i(W + R_P),
    Z_i(C) = int L_i(theta) pi_P(theta) dtheta    (2-D red-noise prior, on a grid),

and the headline is Delta = sum_i [B_i(PL) - B_i(OU)]: about +3 if the red-noise prior
explains the gap to Hazboun. Each evidence is computed two ways:

* ``data``   -- the Gaussian likelihood of the pulsar's real residuals after projecting
                out the timing model (G^T M = 0). This is what the samplers see.
* ``asimov`` -- the expected log-likelihood, -KL(C_true || C) up to a constant, with
                C_true = white + injected GWB + injected intrinsic (power-law) red noise.
                This removes the luck of the noise realisation.

Models. The OU red noise is Argus's: an OU spin-frequency process, whose phase residual
is its integral divided by f0; its covariance is computed in closed form (no frequency
truncation). The power-law red noise is enterprise's: ``n_freqs`` Fourier pairs at k/T
(T the array span), each with variance P(f_k) / T. The GWB and the injected red noise
are stationary power laws integrated on a fine frequency grid, as in
``diag_ou_powerlaw_capture.py``.

Speed. For one spectral shape U the covariance is B + a U, so one eigendecomposition of
L^-1 U L^-T (B = L L^T) gives the log-determinant, quadratic form and trace for every
amplitude a in O(n): the amplitude grids are effectively free.

Caveats: the GW is conditioned at the truth (its own Occam factor is common to both
priors), only auto-power enters (CURN-like), and G-projection stands in for Argus's
Kalman timing marginalisation.

    python scripts/diag_spectral_prior_volume.py --data data/mdc2_d1_all \
        --truth-noise ../data/IPTA_MockDataChallenge2/group1_psr_noise.json \
        --log10-a -15.18045606445813 --pulsars J1909-3744,J1939+2134,J1600-3053,J1713+0747 \
        --out outputs/diag_spectral_prior_volume_mdc2_d1.json \
        --plot outputs/diag_spectral_prior_volume_mdc2_d1.png
"""

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_ou_powerlaw_capture import projector  # noqa: E402
from inject_powerlaw_gwb import powerlaw_psd  # noqa: E402

# Priors (log10 sigma_p, log10 gamma_p) as in configs/mdc2_d1_flat_uprior_*.ini, and
# enterprise's standard power-law red-noise prior (log10 A, gamma).
OU_PRIOR = {"amp": (-20.0, -12.0), "shape": (-12.0, -6.0)}
PL_PRIOR = {"amp": (-20.0, -11.0), "shape": (0.0, 7.0)}
GW_GAMMA = 13.0 / 3.0
GW_LIKE_NATS = 1.0
SUPPORTED, REFUTED = 2.0, 1.0


# ---------------------------------------------------------------------------
# Covariances (seconds^2)
# ---------------------------------------------------------------------------


def _g(x):
    """e^-x - 1 + x, accurate for small x."""
    x = np.asarray(x, dtype=float)
    small = x < 1e-3
    xs = x[small]
    out = np.empty_like(x)
    out[small] = xs**2 / 2 - xs**3 / 6 + xs**4 / 24 - xs**5 / 120
    out[~small] = np.expm1(-x[~small]) + x[~small]
    return out


def ou_covariance(toas, log10_gamma_p, f0):
    """Phase-residual covariance of an OU spin-frequency process with sigma_p = 1.

    d(delta f) = -gamma delta f dt + sigma_p dW, residual = int delta f dt / f0, started
    stationary at the first TOA. Cov = (1/f0^2) sigma_p^2/(2 gamma^3) *
    [g(gamma s) + g(gamma t) - g(gamma |t - s|)], g(x) = e^-x - 1 + x. Its spectrum is
    the one-sided 2 (sigma_p/f0)^2 / (w^2 (gamma^2 + w^2)) of ng15_red_noise_budget.py.
    """
    gam = 10.0**log10_gamma_p
    t = toas - toas[0]
    s, u = np.meshgrid(t, t, indexing="ij")
    bracket = _g(gam * s) + _g(gam * u) - _g(gam * np.abs(s - u))
    return bracket / (2.0 * gam**3 * f0**2)


class StationaryKernel:
    """C(tau) = int S(f) cos(2 pi f tau) df on a fixed grid, trig tables cached."""

    def __init__(self, toas, t_span):
        f_lo = 1.0 / (20.0 * t_span)
        f_hi = 1.0 / (2.0 * 7.0 * 86400.0)
        self.df = f_lo / 4.0
        self.f = np.arange(f_lo, f_hi, self.df)
        ph = 2 * math.pi * np.outer(toas - toas[0], self.f)
        self.c, self.s = np.cos(ph), np.sin(ph)

    def __call__(self, psd):
        w = psd(self.f) * self.df
        return (self.c * w) @ self.c.T + (self.s * w) @ self.s.T


def fourier_basis(toas, t_span, n_freqs):
    """Enterprise Fourier design matrix (sin, cos at k/T) and its frequencies."""
    f = np.arange(1, n_freqs + 1) / t_span
    ph = 2 * math.pi * np.outer(toas, f)
    return np.hstack([np.sin(ph), np.cos(ph)]), f


def pl_covariance(fmat, f, t_span, gamma):
    """Enterprise power-law red noise with log10 A = 0: F diag(P(f_k)/T) F^T."""
    phi = powerlaw_psd(f, 0.0, gamma) / t_span
    return (fmat * np.concatenate([phi, phi])) @ fmat.T


# ---------------------------------------------------------------------------
# Likelihood over an amplitude line, via one eigendecomposition
# ---------------------------------------------------------------------------


def amplitude_line(base, shape, amps, y=None, c_true=None):
    """ln L of covariance base + a * shape for every a in amps.

    Returns (data, asimov): the Gaussian log-density of y, and the expected one
    E_{y ~ N(0, c_true)}[ln N(y; 0, C)]; either is None when its input is.
    """
    n = base.shape[0]
    lb = np.linalg.cholesky(base)
    a = np.linalg.solve(lb, np.linalg.solve(lb, shape).T)  # L^-1 U L^-T
    lam, v = np.linalg.eigh(0.5 * (a + a.T))
    lam = np.clip(lam, 0.0, None)
    logdet_b = 2.0 * np.sum(np.log(np.diag(lb)))
    one_p = 1.0 + np.outer(amps, lam)  # (n_amp, n)
    logdet = logdet_b + np.sum(np.log(one_p), axis=1)
    const = n * math.log(2 * math.pi)
    data = asimov = None
    if y is not None:
        z = v.T @ np.linalg.solve(lb, y)
        data = -0.5 * (np.sum(z**2 / one_p, axis=1) + logdet + const)
    if c_true is not None:
        m = np.linalg.solve(lb, np.linalg.solve(lb, c_true).T)  # L^-1 C_true L^-T
        d = np.einsum("ik,ij,jk->k", v, m, v)  # diag(V^T M V)
        asimov = -0.5 * (np.sum(d / one_p, axis=1) + logdet + const)
    return data, asimov


def grid_map(base, shapes, amp_grid, y, c_true):
    """(n_shape, n_amp) ln L maps for data and asimov; amp_grid in log10 amplitude."""
    amps = 10.0 ** (2.0 * amp_grid)
    out_d = np.empty((len(shapes), amp_grid.size))
    out_a = np.empty_like(out_d)
    for k, u in enumerate(shapes):
        out_d[k], out_a[k] = amplitude_line(base, u, amps, y, c_true)
    return out_d, out_a


def log_evidence(lnl):
    """ln of the uniform-prior average of exp(lnL) over a (shape, amp) grid (trapezoid)."""
    w = np.ones(lnl.shape)
    w[0, :] *= 0.5
    w[-1, :] *= 0.5
    w[:, 0] *= 0.5
    w[:, -1] *= 0.5
    return float(logsumexp(lnl, b=w) - math.log(w.sum()))


def gw_like_fraction(lnl, nats=GW_LIKE_NATS):
    """Prior mass (grid fraction) within ``nats`` of the maximum log-likelihood."""
    return float(np.mean(lnl >= lnl.max() - nats))


# ---------------------------------------------------------------------------
# Per pulsar
# ---------------------------------------------------------------------------


def load_pulsar(path, truth):
    import pyarrow.feather as pf

    t = pf.read_table(path)
    meta = json.loads(t.schema.metadata[b"json"])
    toas = t.column("toas").to_numpy().astype(float)
    sig = t.column("toaerrs").to_numpy().astype(float)
    mcols = [c for c in t.column_names if c.startswith("Mmat_")]
    mmat = np.column_stack([t.column(c).to_numpy() for c in mcols]).astype(float)
    mmat = mmat / np.linalg.norm(mmat, axis=0)
    tr = truth[meta["name"]]
    white = (tr["efac"] * sig) ** 2 + 10.0 ** (2.0 * tr["equad"])
    return {
        "name": meta["name"], "toas": toas, "f0": float(meta["F0"]), "white": white,
        "residuals": t.column("residuals").to_numpy().astype(float),
        "G": projector(mmat), "rn_log10_a": tr["rn_log10_A"], "rn_gamma": tr["rn_spec_ind"],
    }


def analyse(psr, t_array, args, grids):
    G = psr["G"]
    t = psr["toas"]
    span = t.max() - t.min()
    kern = StationaryKernel(t, span)

    def proj(c):
        return G.T @ c @ G

    white = np.diag(psr["white"])
    gw = proj(kern(lambda f: powerlaw_psd(f, args.log10_a, GW_GAMMA)))
    rn_true = proj(kern(lambda f: powerlaw_psd(f, psr["rn_log10_a"], psr["rn_gamma"])))
    w = proj(white)
    c_true = w + gw + (rn_true if args.asimov_intrinsic_rn else 0.0)
    y = G.T @ psr["residuals"]

    fmat, ff = fourier_basis(t, t_array, args.n_freqs)
    shapes = {
        "ou": [proj(ou_covariance(t, s, psr["f0"])) for s in grids["ou"]["shape"]],
        "pl": [proj(pl_covariance(fmat, ff, t_array, s)) for s in grids["pl"]["shape"]],
    }
    res = {"name": psr["name"], "f0": psr["f0"]}
    maps = {}
    for prior in ("ou", "pl"):
        amp = grids[prior]["amp"]
        noise_d, noise_a = grid_map(w, shapes[prior], amp, y, c_true)
        gwm_d, gwm_a = grid_map(w + gw, shapes[prior], amp, y, c_true)
        r = {}
        for kind, ln_n, ln_g in (("data", noise_d, gwm_d), ("asimov", noise_a, gwm_a)):
            z_n, z_g = log_evidence(ln_n), log_evidence(ln_g)
            r[kind] = {
                "lnz_noise": z_n, "lnz_gw": z_g, "b_gw_vs_noise": z_g - z_n,
                "max_lnl_noise": float(ln_n.max()), "max_lnl_gw": float(ln_g.max()),
                "gw_like_fraction": gw_like_fraction(ln_n),
                "ln_gw_like_fraction": math.log(max(gw_like_fraction(ln_n), 1e-300)),
            }
        res[prior] = r
        maps[prior] = {"noise_data": noise_d, "noise_asimov": noise_a}
    for kind in ("data", "asimov"):
        res[f"delta_pl_minus_ou_{kind}"] = (
            res["pl"][kind]["b_gw_vs_noise"] - res["ou"][kind]["b_gw_vs_noise"])
    return res, maps


def verdict(delta):
    if delta >= SUPPORTED:
        return "SUPPORTED: the red-noise prior explains most of the gap to Hazboun"
    if abs(delta) < REFUTED:
        return "REFUTED: the red-noise prior does not explain the gap"
    return "PARTIAL: the red-noise prior explains part of the gap"


def make_grids(step_amp, step_ou, step_pl):
    def rng(lo, hi, step):
        return np.linspace(lo, hi, int(round((hi - lo) / step)) + 1)

    return {
        "ou": {"amp": rng(*OU_PRIOR["amp"], step_amp), "shape": rng(*OU_PRIOR["shape"], step_ou)},
        "pl": {"amp": rng(*PL_PRIOR["amp"], step_amp), "shape": rng(*PL_PRIOR["shape"], step_pl)},
    }


def plot(results, maps, grids, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(n + 1, 2, figsize=(9, 2.6 * (n + 1)), squeeze=False)
    for i, (r, m) in enumerate(zip(results, maps)):
        for j, (prior, xl, yl) in enumerate((("ou", "log10 sigma_p", "log10 gamma_p"),
                                             ("pl", "log10 A", "gamma"))):
            ax = axes[i, j]
            z = m[prior]["noise_data"]
            g = grids[prior]
            ax.pcolormesh(g["amp"], g["shape"], np.clip(z - z.max(), -30, 0), shading="auto")
            ax.contour(g["amp"], g["shape"], z - z.max(), levels=[-GW_LIKE_NATS], colors="w")
            ax.set_title(f"{r['name']} {prior.upper()}: GW-like frac "
                         f"{r[prior]['data']['gw_like_fraction']:.2e}", fontsize=8)
            ax.set_xlabel(xl, fontsize=8)
            ax.set_ylabel(yl, fontsize=8)
    ax = axes[n, 0]
    x = np.arange(n)
    for off, kind in ((-0.2, "data"), (0.2, "asimov")):
        ax.bar(x + off, [r[f"delta_pl_minus_ou_{kind}"] for r in results], 0.4, label=kind)
    ax.set_xticks(x, [r["name"] for r in results], rotation=60, fontsize=7)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("B(PL) - B(OU) [nats]")
    ax.legend(fontsize=7)
    axes[n, 1].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="Directory of per-pulsar feathers.")
    p.add_argument("--truth-noise", required=True, help="MDC2 group1_psr_noise.json.")
    p.add_argument("--log10-a", type=float, required=True, help="Injected GWB log10 A.")
    p.add_argument("--pulsars", default="all", help="Comma-separated names, or 'all'.")
    p.add_argument("--headline", default="J1909-3744,J1939+2134,J1600-3053,J1713+0747",
                   help="Pulsars summed for the headline Delta.")
    p.add_argument("--n-freqs", type=int, default=30, help="Enterprise Fourier pairs.")
    p.add_argument("--step-amp", type=float, default=0.05)
    p.add_argument("--step-ou", type=float, default=0.1)
    p.add_argument("--step-pl", type=float, default=0.1)
    p.add_argument("--no-asimov-intrinsic-rn", dest="asimov_intrinsic_rn",
                   action="store_false", help="Leave injected red noise out of C_true.")
    p.add_argument("--out", default=None)
    p.add_argument("--plot", default=None)
    args = p.parse_args()

    with open(args.truth_noise) as f:
        truth = json.load(f)
    files = sorted(glob.glob(os.path.join(args.data, "*.feather")))
    psrs_all = [load_pulsar(f, truth) for f in files]
    t_array = (max(p_["toas"].max() for p_ in psrs_all)
               - min(p_["toas"].min() for p_ in psrs_all))
    want = None if args.pulsars == "all" else set(args.pulsars.split(","))
    psrs = [p_ for p_ in psrs_all if want is None or p_["name"] in want]
    grids = make_grids(args.step_amp, args.step_ou, args.step_pl)

    results, maps = [], []
    for psr in psrs:
        r, m = analyse(psr, t_array, args, grids)
        results.append(r)
        maps.append(m)
        print(f"  {r['name']:12s}  B_OU data {r['ou']['data']['b_gw_vs_noise']:+7.2f}"
              f"  B_PL data {r['pl']['data']['b_gw_vs_noise']:+7.2f}"
              f"  Delta data {r['delta_pl_minus_ou_data']:+6.2f}"
              f"  asimov {r['delta_pl_minus_ou_asimov']:+6.2f}"
              f"  GW-like frac OU {r['ou']['data']['gw_like_fraction']:.1e}"
              f" PL {r['pl']['data']['gw_like_fraction']:.1e}", flush=True)

    head = set(args.headline.split(","))
    summary = {}
    for kind in ("data", "asimov"):
        d_head = sum(r[f"delta_pl_minus_ou_{kind}"] for r in results if r["name"] in head)
        d_all = sum(r[f"delta_pl_minus_ou_{kind}"] for r in results)
        summary[kind] = {
            "delta_headline": d_head, "delta_all_analysed": d_all,
            "b_ou_all_analysed": sum(r["ou"][kind]["b_gw_vs_noise"] for r in results),
            "b_pl_all_analysed": sum(r["pl"][kind]["b_gw_vs_noise"] for r in results),
            "verdict_headline": verdict(d_head),
        }
        print(f"\n[{kind}] Delta (headline {len(head & {r['name'] for r in results})} psr) "
              f"= {d_head:+.2f};  all {len(results)} analysed = {d_all:+.2f}")
        print(f"[{kind}] {summary[kind]['verdict_headline']}")

    out = {
        "log10_a_gw": args.log10_a, "gw_gamma": GW_GAMMA, "n_freqs": args.n_freqs,
        "t_array_s": t_array, "priors": {"ou": OU_PRIOR, "pl": PL_PRIOR},
        "grid_steps": {"amp": args.step_amp, "ou_shape": args.step_ou,
                       "pl_shape": args.step_pl},
        "asimov_intrinsic_rn": args.asimov_intrinsic_rn,
        "headline_pulsars": sorted(head), "thresholds": {"supported": SUPPORTED,
                                                         "refuted": REFUTED},
        "summary": summary, "pulsars": results,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.out}")
    if args.plot:
        plot(results, maps, grids, args.plot)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
