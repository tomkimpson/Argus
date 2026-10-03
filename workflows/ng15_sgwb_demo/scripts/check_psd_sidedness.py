#!/usr/bin/env python
"""Check that the OU residual PSD used in OU-vs-power-law comparisons is ONE-sided.

The enterprise power law ``P(f)`` is one-sided (``int_0^inf P df`` = variance). Until
2026-10-03 the repo's OU PSD (``inject_powerlaw_gwb.ou_psd`` and its copies in
``check_mdc2_truth.py`` / ``compare_ou_recovery.py``) was

    S(f) = sigma^2 / (w^2 (gamma^2 + w^2)),   w = 2 pi f,

which is the TWO-sided density (its integral over +-f is the OU stationary variance
sigma^2/(2 gamma)), so every OU-vs-power-law comparison read the OU 0.30 dex low. This
script found that and now guards the fix (``ou_psd`` carries the factor 2).

It simulates with the repo's own generators (``inject_red_noise`` for per-pulsar spin
noise, mapped onto ``ou_psd`` via ha^2 = 12 sigma_p^2 / (f0^2 gamma_p); ``inject_ou_gwb``
for the GW state) and takes the one-sided Welch PSD of the FIRST-DIFFERENCED residual
(the OU residual is a random walk at low frequency), whose exact PSD is
``S(f) * 4 sin^2(pi f dt)``. A white-noise series (one-sided PSD ``2 s^2 dt``) is the
estimator control. Expect a median log10 ratio (Welch / ou_psd) of ~0; ~+0.301 would
mean ``ou_psd`` is two-sided again.

CPU only, ~15 s:

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

    # Per-pulsar OU spin red noise, residual = dphi/f0: the GW form with
    # sigma_a2 = sigma_p^2/f0^2, i.e. ha^2 = 12 sigma_p^2 / (f0^2 gamma_p).
    res = inject_red_noise(t, [GAMMA], [SIGMA_P], [F0], rng)[0]
    f, p = diff_welch(res)
    log10_ha_rn = 0.5 * np.log10(12.0 * SIGMA_P**2 / (F0**2 * GAMMA))
    rn = report("per-pulsar OU red noise vs ou_psd", f, p,
                ou_psd(f, log10_ha_rn, np.log10(GAMMA)) * diff_tf(f))

    # GW OU state, single pulsar (Gamma = 1). Model: ou_psd exactly.
    res = inject_ou_gwb(t, LOG10_HA, np.log10(GAMMA), np.eye(1), np.eye(1), rng)[0]
    f, p = diff_welch(res)
    gw = report("GW OU vs ou_psd", f, p, ou_psd(f, LOG10_HA, np.log10(GAMMA)) * diff_tf(f))

    print()
    if abs(ctrl) > 0.05:
        print("ESTIMATOR CONTROL FAILED -- do not trust the OU verdict.")
        return
    ok = True
    for name, v in (("red noise", rn), ("GW", gw)):
        side = "TWO-sided" if abs(v - np.log10(2)) < 0.05 else (
            "one-sided" if abs(v) < 0.05 else "INCONCLUSIVE"
        )
        ok &= side == "one-sided"
        print(f"{name}: ou_psd is {side} (offset {v:+.3f} dex; log10 2 = 0.301)")
    print("PASS: ou_psd is one-sided like powerlaw_psd" if ok else "FAIL")


if __name__ == "__main__":
    main()
