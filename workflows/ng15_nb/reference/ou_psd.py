"""Argus's integrated-OU residual PSDs, in the form a Fourier-basis GP needs.

Shared by both environments: the Argus-env tests check these against the Kalman
filter's own F/Q (the ground truth), and the Discovery-env reference driver uses them
as GP priors. The PSDs use only arithmetic operators and ``**``, so they accept numpy
and jax arrays alike (and trace under jit/grad).

Conventions, matching ``argus.model`` and ``argus.jax_kalman_filter``:

- Each process is an integrated OU state (x, v) with dx = v dt, dv = -γ v dt + σ dW.
  The position's TWO-sided PSD in ordinary frequency f, with ω = 2πf, is
  σ² / (ω² (γ² + ω²)). The ONE-sided PSD, which is what an enterprise/Discovery power
  law is, is twice that.
- GW: H = -1 on the position and σ² = (h_a² / 12) γ_a (× the ORF), so the residual PSD
  has σ² = h_a² γ_a / 12. The ridge basis samples ``log10_pivot_psd``, the TWO-sided
  density at the pivot frequency, and derives h_a.
- Spin noise: H = 1/f0 on the phase and σ² = σ_p², so the residual PSD has
  σ² = σ_p² / f0².

A Discovery GP prior is the per-coefficient variance S₁(f)·Δf, with f and Δf already
repeated for the sin/cos pair. Discovery names a GP's parameters after the prior
function's arguments, so the two priors below are written out with explicit names. The
spin prior takes log10 σ_r = log10 σ_p − log10 f0 (residual units): a Discovery common
GP applies one prior function to every pulsar, so the per-pulsar f0 is folded in by the
caller.
"""

import math

import numpy as np

TWO_PI = 2.0 * math.pi
LOG10_12 = math.log10(12.0)

# The pivot frequency the frozen ridge basis uses, as written in every ladder config
# (evidence_procedure_v1.json, gw_parameterization.pivot_freq_hz). It is 1/(5 yr)
# truncated to 5 significant figures, and the truncation matters at 1e-12.
PIVOT_F_HZ = 6.3376e-09


def ou_psd_twosided(f, sigma2, gamma):
    """Two-sided PSD of an integrated OU position: σ² / (ω² (γ² + ω²))."""
    w = TWO_PI * f
    return sigma2 / (w**2 * (gamma**2 + w**2))


def gw_psd_onesided(f, log10_pivot_psd, log10_gamma_a, pivot_f=PIVOT_F_HZ):
    """One-sided GW residual PSD in the ridge basis (Argus ``gw_parameterization = ridge``)."""
    ga = 10.0**log10_gamma_a
    w, wp = TWO_PI * f, TWO_PI * pivot_f
    shape = (wp**2 * (ga**2 + wp**2)) / (w**2 * (ga**2 + w**2))
    return 2.0 * 10.0**log10_pivot_psd * shape


def log10_ha_from_pivot(log10_pivot_psd, log10_gamma_a, pivot_f=PIVOT_F_HZ):
    """The h_a that the ridge basis derives (``parameter_sampling.sample_gw_parameters``).

    Post-processing only (numpy, not traceable).
    """
    ga = 10.0**log10_gamma_a
    wp = TWO_PI * pivot_f
    return 0.5 * (LOG10_12 + log10_pivot_psd + 2.0 * math.log10(wp) + np.log10(ga**2 + wp**2) - log10_gamma_a)


def spin_psd_onesided(f, log10_gamma_p, log10_sigma_p, f0):
    """One-sided spin-noise residual PSD for a pulsar of spin frequency ``f0`` (Hz)."""
    return 2.0 * ou_psd_twosided(f, 10.0 ** (2.0 * log10_sigma_p) / f0**2, 10.0**log10_gamma_p)


def ou_rn_prior(f, df, log10_gamma_p, log10_sigma_r):
    """Discovery prior for per-pulsar spin noise; σ_r = σ_p / f0."""
    return 2.0 * ou_psd_twosided(f, 10.0 ** (2.0 * log10_sigma_r), 10.0**log10_gamma_p) * df


def ou_gw_prior(f, df, log10_pivot_psd, log10_gamma_a):
    """Discovery prior for the common GW process in the ridge basis."""
    return gw_psd_onesided(f, log10_pivot_psd, log10_gamma_a) * df
