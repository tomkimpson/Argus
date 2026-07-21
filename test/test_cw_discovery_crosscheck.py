"""Cross-check Argus's CW Earth-term waveform against NanoGrav Discovery's.

NanoGrav's Discovery (``discovery.deterministic.makedelay_binary``) and Argus
(``gravitational_waves.compute_cw_signal_single_pulsar``) both implement the
continuous-wave timing residual from a circular SMBH binary (Ellis et al. 2012,
2013). They use different but equivalent internal conventions:

  * Discovery applies the polarization angle psi by rotating the plus/cross
    *amplitudes* (r_+, r_x) and uses an antenna pattern ``fpc_fast`` without psi;
    its amplitude prefactor is ``alpha = h0 / (2 pi f0)``.
  * Argus applies psi inside the polarization *tensors* (so psi enters F_+, F_x)
    and carries the Earth-term amplitudes ``h0(1+cos^2 i)/(2 Omega)`` and
    ``-h0 cos i / Omega``.

Both encode the SAME physics. Empirically (verified by this test to ~1e-15) the
two residual time series are related by a *single, parameter-independent* map for
arbitrary source/pulsar geometry, inclination, phase, and frequency:

        residual_argus(psi) = -0.5 * residual_discovery(-psi)

i.e. they agree up to (a) a global convention constant C = -1/2 (Discovery's
amplitude is 2x Argus's, with an overall sign), and (b) a polarization-angle
*sign* flip, ``psi_argus = -psi_discovery`` -- the well-known handedness
convention ambiguity in the definition of psi. Anyone transferring CW parameters
between the two codes must flip the sign of psi (the amplitude/sign convention is
absorbed when fitting h0). This test asserts that exact relationship.

A self-contained re-implementation of Discovery's Earth-term formula is vendored
here (≈30 lines, faithful to ``discovery/src/discovery/deterministic.py``) so the
test has no dependency on the ``discovery`` package.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from argus.gravitational_waves import (
    antenna_pattern_single,
    compute_cw_signal_single_pulsar,
)

TREF = 86400.0 * 51544.5  # MJD J2000 in seconds (Discovery's phase reference)


def _discovery_fpc(pos, gwtheta, gwphi):
    """Antenna patterns, faithful to discovery.deterministic.fpc_fast."""
    x, y, z = pos
    sin_phi, cos_phi = np.sin(gwphi), np.cos(gwphi)
    sin_th, cos_th = np.sin(gwtheta), np.cos(gwtheta)

    m_dot = sin_phi * x - cos_phi * y
    n_dot = -cos_th * cos_phi * x - cos_th * sin_phi * y + sin_th * z
    omhat_dot = -sin_th * cos_phi * x - sin_th * sin_phi * y - cos_th * z

    denom = 1.0 + omhat_dot
    fplus = 0.5 * (m_dot**2 - n_dot**2) / denom
    fcross = (m_dot * n_dot) / denom
    return fplus, fcross


def _discovery_earth_term(toas, ra_psr, dec_psr, log10_h0, log10_f0,
                          ra_gw, sindec_gw, cosinc, psi, phi_earth):
    """Earth-term-only CW residual, faithful to discovery makedelay_binary(pulsarterm=False)."""
    h0 = 10.0**log10_h0
    f0 = 10.0**log10_f0
    dec_gw, inc = np.arcsin(sindec_gw), np.arccos(cosinc)

    # Pulsar position unit vector (standard convention, matches argus.pulsar_direction).
    pos = np.array([np.cos(ra_psr) * np.cos(dec_psr),
                    np.sin(ra_psr) * np.cos(dec_psr),
                    np.sin(dec_psr)])
    fplus, fcross = _discovery_fpc(pos, 0.5 * np.pi - dec_gw, ra_gw)

    phase = phi_earth + 2.0 * np.pi * f0 * (toas - TREF)
    delta_sin, delta_cos = np.sin(phase), np.cos(phase)

    At = -(1.0 + np.cos(inc) ** 2) * delta_sin
    Bt = 2.0 * np.cos(inc) * delta_cos
    alpha = h0 / (2.0 * np.pi * f0)

    rplus = alpha * (-At * np.cos(2 * psi) + Bt * np.sin(2 * psi))
    rcross = alpha * (At * np.sin(2 * psi) + Bt * np.cos(2 * psi))
    return -fplus * rplus - fcross * rcross


def _argus_earth_term(toas, ra_psr, dec_psr, log10_h0, log10_f0,
                      ra_gw, sindec_gw, cosinc, psi, phi_earth):
    """Earth-term-only CW residual from Argus (pulsar_distance=0 => no pulsar term)."""
    h0 = 10.0**log10_h0
    f_gw = 10.0**log10_f0
    dec_gw = np.arcsin(sindec_gw)

    F_plus, F_cross = antenna_pattern_single(ra_psr, dec_psr, ra_gw, dec_gw, psi)
    # Argus references phase from t=0; Discovery references from TREF. Both are
    # pure sinusoids at Omega, so we evaluate Argus at (toas - TREF) and fold the
    # Earth phase into Phi0 to put them on a common phase origin.
    res = compute_cw_signal_single_pulsar(
        jnp.asarray(toas - TREF),
        f_gw, h0, cosinc, phi_earth, F_plus, F_cross,
        pulsar_distance=0.0, geometric_factor=0.0,
    )
    return np.asarray(res)


@pytest.mark.parametrize("seed", range(8))
def test_argus_matches_discovery_earth_term_up_to_global_constant(seed):
    rng = np.random.default_rng(seed)

    # Randomized geometry / source parameters.
    ra_psr = rng.uniform(0, 2 * np.pi)
    dec_psr = np.arcsin(rng.uniform(-1, 1))
    ra_gw = rng.uniform(0, 2 * np.pi)
    sindec_gw = rng.uniform(-1, 1)
    cosinc = rng.uniform(-1, 1)
    psi = rng.uniform(0, np.pi)
    phi_earth = rng.uniform(0, 2 * np.pi)
    log10_f0 = rng.uniform(-8.5, -7.0)
    log10_h0 = rng.uniform(-15.0, -13.0)

    toas = np.sort(rng.uniform(0, 15 * 365.25 * 86400.0, size=200)) + TREF

    arg = _argus_earth_term(toas, ra_psr, dec_psr, log10_h0, log10_f0,
                            ra_gw, sindec_gw, cosinc, psi, phi_earth)
    # Convention map: psi_argus = -psi_discovery.
    disc = _discovery_earth_term(toas, ra_psr, dec_psr, log10_h0, log10_f0,
                                 ra_gw, sindec_gw, cosinc, -psi, phi_earth)

    # Fit a single global constant C: arg ~= C * disc.
    denom = np.dot(disc, disc)
    assert denom > 0
    C = np.dot(arg, disc) / denom
    rel_resid = np.linalg.norm(arg - C * disc) / np.linalg.norm(arg)

    assert rel_resid < 1e-8, (
        f"Argus and Discovery Earth-term waveforms disagree beyond the "
        f"convention map (rel_resid={rel_resid:.2e}, C={C:.4f}, seed={seed})"
    )
    # The convention constant is C = -1/2 (Discovery amplitude is 2x, opposite sign).
    assert abs(C - (-0.5)) < 1e-6, f"Unexpected global constant C={C} (expected -0.5)"


def test_global_constant_is_consistent_across_parameters():
    """C must be the SAME (= -1/2) for different geometries under the psi-sign
    convention map: a fixed convention constant, not a parameter-dependent fudge."""
    rng = np.random.default_rng(123)
    consts = []
    for _ in range(10):
        ra_psr = rng.uniform(0, 2 * np.pi)
        dec_psr = np.arcsin(rng.uniform(-1, 1))
        ra_gw = rng.uniform(0, 2 * np.pi)
        sindec_gw = rng.uniform(-1, 1)
        cosinc = rng.uniform(-1, 1)
        psi = rng.uniform(0, np.pi)
        phi_earth = rng.uniform(0, 2 * np.pi)
        toas = np.sort(rng.uniform(0, 15 * 365.25 * 86400.0, size=150)) + TREF
        arg = _argus_earth_term(toas, ra_psr, dec_psr, -14.0, -8.0,
                                ra_gw, sindec_gw, cosinc, psi, phi_earth)
        disc = _discovery_earth_term(toas, ra_psr, dec_psr, -14.0, -8.0,
                                     ra_gw, sindec_gw, cosinc, -psi, phi_earth)
        consts.append(np.dot(arg, disc) / np.dot(disc, disc))

    consts = np.array(consts)
    assert np.allclose(consts, -0.5, rtol=1e-6), (
        f"Convention constant not consistently -0.5 across geometry: {consts}"
    )
