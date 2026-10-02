#!/usr/bin/env python
"""Settle the sidedness of the OU residual PSD used in every OU-vs-power-law comparison.

The repo compares the OU residual PSD

    S_OU(f) = sigma^2 / (w^2 (gamma^2 + w^2)),   w = 2 pi f

(``inject_powerlaw_gwb.ou_psd``; per-pulsar red noise uses sigma^2 = sigma_p^2 / f0^2)
directly against the enterprise power law ``P(f)``, which is ONE-sided
(``int_0^inf P df`` = variance). Analytically, the OU frequency state has stationary
variance sigma^2/(2 gamma) = ``int_-inf^inf sigma^2/(gamma^2+w^2) df``, which would make
``S_OU`` TWO-sided and every such comparison 0.30 dex off.

This script checks that numerically with the repo's own generators
(``inject_red_noise`` for per-pulsar spin noise, ``inject_ou_gwb`` for the GW state). The
OU residual is a random walk at low frequency (non-stationary), so we take the one-sided
Welch PSD of the FIRST-DIFFERENCED residual, whose exact PSD is
``S(f) * 4 sin^2(pi f dt)``. A white-noise series (one-sided PSD ``2 s^2 dt``) is the
estimator control.

Verdict: the median log10 ratio (Welch / model) over the mid-band is ~0 if ``S_OU`` is
one-sided and ~+0.301 if it is two-sided.

CPU only, ~1 min:

    JAX_PLATFORMS=cpu python workflows/ng15_sgwb_demo/scripts/check_psd_sidedness.py
"""

import os
import sys

import numpy as np
from scipy.signal import welch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inject_powerlaw_gwb import (  # noqa: E402
    SEC_PER_DAY,
    inject_ou_gwb,
    inject_red_noise,
    ou_psd,
)

DT = 1.0 * SEC_PER_DAY  # uniform 1-day sampling
NSTEP = 2**17  # ~360 yr of daily samples, so Welch can average many segments
NPERSEG = 2**13
GAMMA = 1.0 / (30.0 * SEC_PER_DAY)  # corner f_c = gamma/2pi ~ 1/(190 d), mid-band
F0 = 300.0  # Hz, arbitrary spin frequency for the red-noise path
SIGMA_P = 1e-15  # arbitrary driving amplitude; ratios are scale-free
LOG10_HA = -14.0


def diff_welch(x):
    """One-sided Welch PSD of the first difference of ``x`` (sampled every ``DT``)."""
    f, p = welch(np.diff(x), fs=1.0 / DT, nperseg=NPERSEG, return_onesided=True)
    return f[1:], p[1:]


def midband(f):
    """Bins away from the lowest (few-segment) and highest (near-Nyquist) frequencies."""
    return (f > 20.0 / (NPERSEG * DT)) & (f < 0.2 / DT)


def report(label, f, p_meas, p_model):
    """Print the median log10(measured / model) over the mid-band."""
    m = midband(f)
    r = np.log10(p_meas[m] / p_model[m])
    print(
        f"{label:<34} median log10(Welch/model) = {np.median(r):+.3f}  "
        f"[16-84%: {np.percentile(r, 16):+.3f}, {np.percentile(r, 84):+.3f}]  "
        f"({m.sum()} bins)"
    )
    return float(np.median(r))


def main():
    """Run the white-noise control and the two OU checks; print the verdict."""
    rng = np.random.default_rng(20261002)
    t = np.arange(NSTEP) * DT
    diff_tf = lambda f: 4.0 * np.sin(np.pi * f * DT) ** 2  # noqa: E731

    # Control: white noise of variance s^2 has one-sided PSD 2 s^2 dt.
    s = 1.0
    f, p = diff_welch(rng.standard_normal(NSTEP) * s)
    ctrl = report("white control (one-sided 2s^2dt)", f, p, 2 * s**2 * DT * diff_tf(f))

    # Per-pulsar OU spin red noise, residual = dphi/f0. Model: ou_psd form with
    # sigma^2 = sigma_p^2/f0^2.
    res = inject_red_noise(t, [GAMMA], [SIGMA_P], [F0], rng)[0]
    f, p = diff_welch(res)
    w = 2 * np.pi * f
    s_rn = (SIGMA_P**2 / F0**2) / (w**2 * (GAMMA**2 + w**2))
    rn = report("per-pulsar OU red noise vs S_OU", f, p, s_rn * diff_tf(f))

    # GW OU state, single pulsar (Gamma = 1). Model: ou_psd exactly.
    res = inject_ou_gwb(t, LOG10_HA, np.log10(GAMMA), np.eye(1), np.eye(1), rng)[0]
    f, p = diff_welch(res)
    gw = report("GW OU vs ou_psd", f, p, ou_psd(f, LOG10_HA, np.log10(GAMMA)) * diff_tf(f))

    print()
    if abs(ctrl) > 0.05:
        print("ESTIMATOR CONTROL FAILED -- do not trust the OU verdict.")
        return
    for name, v in (("red noise", rn), ("GW", gw)):
        side = "TWO-sided" if abs(v - np.log10(2)) < 0.05 else (
            "one-sided" if abs(v) < 0.05 else "INCONCLUSIVE"
        )
        print(f"{name}: S_OU is {side} (offset {v:+.3f} dex; log10 2 = 0.301)")


if __name__ == "__main__":
    main()
