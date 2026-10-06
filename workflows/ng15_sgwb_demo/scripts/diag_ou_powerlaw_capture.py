#!/usr/bin/env python
"""How much of an injected power-law GWB can a common OU process capture? (CPU, analytic)

Why this exists
---------------
MDC2 injects the GWB as a power law in residual PSD, S(f) ∝ f^-gamma with
gamma = 13/3. Argus models the common process as an integrated OU, whose residual PSD
falls as f^-2 above the corner gamma_a / 2 pi. If no OU spectrum reproduces the
injected one over the band where the signal clears the white noise, the OU model can
only ever recover part of the evidence the data hold, and the GW+noise vs noise-only
Bayes factor shrinks accordingly.

This computes it without sampling, as the expected log-likelihood (a KL divergence)
of each model when the data are drawn from the true model. It works in the time
domain on each pulsar's real TOAs, after projecting out its timing model (the
feather's design matrix), so the quadratic spin-down fit's absorption of the lowest
frequencies is included exactly:

    E[ln L_true] - E[ln L_model] = KL( N(0, C_true) || N(0, C_model) )

with C_true = white + power-law GWB and C_model = white + model red process, all
projected by G (G^T M = 0). The capture fraction is

    R = 1 - KL(true || best OU) / KL(true || white only),

1 for a model that recovers all of the detectable signal, 0 for one that recovers
none. "Best OU" is a single (pivot log-PSD, gamma_a) shared by every pulsar, as in
CURN. The per-pulsar best OU (each pulsar its own fit) is reported as an upper bound.

    python scripts/diag_ou_powerlaw_capture.py --data data/mdc2_d1_all \
        --log10-a -15.18045606445813 --out outputs/diag_ou_powerlaw_capture_mdc2_d1.json
"""

import argparse
import glob
import json
import math
import os

import numpy as np

SEC_PER_YEAR = 365.25 * 86400.0
F_YR = 1.0 / SEC_PER_YEAR
PIVOT_F = 1.0 / (5.0 * SEC_PER_YEAR)


def powerlaw_psd(f, log10_a, gamma=13.0 / 3.0):
    """One-sided residual PSD of a power-law GWB (enterprise convention), s^2/Hz."""
    a = 10.0**log10_a
    return a**2 / (12.0 * math.pi**2) * F_YR ** (gamma - 3.0) * f ** (-gamma)


def ou_psd(f, log10_pivot_psd, log10_gamma_a):
    """One-sided residual PSD of the integrated OU, normalised at the pivot.

    ``log10_pivot_psd`` is the TWO-sided density at PIVOT_F, as sampled in ridge mode,
    so the one-sided spectrum is twice it. Shape: 1 / (w^2 (gamma_a^2 + w^2)).
    """
    ga = 10.0**log10_gamma_a
    w, wp = 2 * math.pi * f, 2 * math.pi * PIVOT_F
    shape = (wp**2 * (ga**2 + wp**2)) / (w**2 * (ga**2 + w**2))
    return 2.0 * 10.0**log10_pivot_psd * shape


def covariance_from_psd(toas, psd_fn, t_span):
    """Stationary covariance C(tau) = ∫ S(f) cos(2 pi f tau) df on a fine grid.

    The integral starts at 1/(20 T): power below that is a near-quadratic trend over
    the span and is removed by the timing-model projection anyway.
    """
    f_lo = 1.0 / (20.0 * t_span)
    f_hi = 1.0 / (2.0 * 7.0 * 86400.0)
    df = f_lo / 4.0
    f = np.arange(f_lo, f_hi, df)
    s = psd_fn(f) * df
    # cos(a - b) = cos a cos b + sin a sin b keeps memory at O(n * n_f).
    ph = 2 * math.pi * np.outer(toas - toas[0], f)
    c, sn = np.cos(ph), np.sin(ph)
    return (c * s) @ c.T + (sn * s) @ sn.T


def projector(mmat):
    """Orthonormal basis G of the complement of the timing-model design matrix."""
    u, _, _ = np.linalg.svd(mmat, full_matrices=True)
    return u[:, mmat.shape[1]:]


def kl_gauss(c_true, c_model):
    """KL(N(0, c_true) || N(0, c_model)) for symmetric positive-definite matrices."""
    lm = np.linalg.cholesky(c_model)
    lt = np.linalg.cholesky(c_true)
    x = np.linalg.solve(lm, lt)
    n = c_true.shape[0]
    logdet_m = 2 * np.sum(np.log(np.diag(lm)))
    logdet_t = 2 * np.sum(np.log(np.diag(lt)))
    return 0.5 * (np.sum(x * x) - n + logdet_m - logdet_t)


def load_pulsar(path):
    import pandas as pd

    d = pd.read_feather(path)
    toas = d["toas"].to_numpy(float)
    mcols = [c for c in d.columns if c.startswith("Mmat_")]
    mmat = d[mcols].to_numpy(float)
    # Column-normalise for a well-conditioned SVD.
    mmat = mmat / np.linalg.norm(mmat, axis=0)
    return {
        "name": os.path.splitext(os.path.basename(path))[0],
        "toas": toas,
        "white": d["toaerrs"].to_numpy(float) ** 2,
        "G": projector(mmat),
    }


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", required=True, help="Directory of per-pulsar feathers.")
    p.add_argument("--log10-a", type=float, required=True, help="Injected log10 A.")
    p.add_argument("--gamma", type=float, default=13.0 / 3.0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    psrs = [load_pulsar(f) for f in sorted(glob.glob(os.path.join(args.data, "*.feather")))]
    pivot_grid = np.round(np.arange(-10.0, -4.99, 0.1), 3)
    gamma_grid = np.round(np.arange(-11.0, -5.99, 0.25), 3)

    kl_white = np.zeros(len(psrs))
    kl_ou = np.zeros((len(psrs), pivot_grid.size, gamma_grid.size))
    for i, psr in enumerate(psrs):
        t = psr["toas"]
        span = t.max() - t.min()
        G = psr["G"]
        white = G.T @ np.diag(psr["white"]) @ G
        c_gw = G.T @ covariance_from_psd(
            t, lambda f: powerlaw_psd(f, args.log10_a, args.gamma), span) @ G
        c_true = white + c_gw
        kl_white[i] = kl_gauss(c_true, white)
        # OU covariance is linear in 10**pivot, so build the unit-pivot one per gamma_a.
        for k, lga in enumerate(gamma_grid):
            unit = G.T @ covariance_from_psd(t, lambda f: ou_psd(f, 0.0, lga), span) @ G
            for j, s in enumerate(pivot_grid):
                kl_ou[i, j, k] = kl_gauss(c_true, white + 10.0**s * unit)
        print(f"  {psr['name']}: KL(true||white) = {kl_white[i]:.2f}  "
              f"best own OU KL = {kl_ou[i].min():.2f}", flush=True)

    total_white = kl_white.sum()
    common = kl_ou.sum(axis=0)
    j, k = np.unravel_index(np.argmin(common), common.shape)
    per_psr_best = kl_ou.reshape(len(psrs), -1).min(axis=1).sum()
    res = {
        "log10_a": args.log10_a,
        "gamma": args.gamma,
        "expected_dlnl_true_vs_white": float(total_white),
        "common_ou_best": {
            "log10_pivot_psd_two_sided": float(pivot_grid[j]),
            "log10_gamma_a": float(gamma_grid[k]),
            "kl_to_true": float(common[j, k]),
            "capture_fraction": float(1.0 - common[j, k] / total_white),
            "on_grid_edge": bool(j in (0, pivot_grid.size - 1)
                                 or k in (0, gamma_grid.size - 1)),
        },
        "per_pulsar_ou_best": {
            "kl_to_true": float(per_psr_best),
            "capture_fraction": float(1.0 - per_psr_best / total_white),
        },
        "injected_log10_pivot_psd_two_sided": float(
            math.log10(powerlaw_psd(PIVOT_F, args.log10_a, args.gamma) / 2.0)
        ),
        "pulsars": [
            {"name": psr["name"], "kl_true_vs_white": float(kl_white[i]),
             "kl_true_vs_common_ou": float(kl_ou[i, j, k])}
            for i, psr in enumerate(psrs)
        ],
    }
    print(f"\nExpected ln L gain of the true power law over white only: {total_white:.2f}")
    c = res["common_ou_best"]
    print(f"Best common OU: pivot {c['log10_pivot_psd_two_sided']:.1f}, "
          f"log10 gamma_a {c['log10_gamma_a']:.2f} -> KL {c['kl_to_true']:.2f}, "
          f"capture {c['capture_fraction']:.2f}" + (" (GRID EDGE)" if c["on_grid_edge"] else ""))
    print(f"Per-pulsar best OU: capture {res['per_pulsar_ou_best']['capture_fraction']:.2f}")
    print(f"Injected two-sided pivot log-PSD: {res['injected_log10_pivot_psd_two_sided']:.2f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
