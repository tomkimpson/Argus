"""The OU PSDs the Discovery reference uses are the PSDs the Kalman filter implies.

Spec task 1.3 (spec/spec.md). The NG15 narrowband reference runs Discovery with
OU-shaped spectra (``workflows/ng15_nb/reference/ou_psd.py``) so that its lnB can be
compared with Argus's. That comparison means nothing unless the two describe the same
process, so the PSDs are checked against the filter's own transition and process-noise
blocks (``argus.model.get_F_block`` / ``get_Q_block``), which are the ground truth.

Two second moments pin the factor of 2 (sidedness) and the σ² convention. For the
integrated OU (dx = v dt, dv = -γ v dt + σ dW), with S₁ the one-sided position PSD:

- velocity variance: ∫₀^∞ ω² S₁(f) df = σ²/(2γ), the filter's stationary Q22;
- position structure function: ∫₀^∞ S₁(f)·2(1 − cos ωτ) df
  = σ²/γ² [τ − (1 − e^{−γτ})/γ]
  = F12(τ)² σ²/(2γ) + σ² q11(τ), with the velocity at its stationary variance.

The middle form is the bridge: quadrature ties the PSD to it, and the filter's F/Q are
compared with it directly. A convention error would show as a factor of 2 or more, so
the quadrature tolerance (1e-5) is loose only relative to rounding, not to what is
being tested.

The module is then checked against the two scripts it replaces, to 1e-12.
"""

import importlib.util
import math
import os
import sys

import jax
import numpy as np
import pytest
from scipy.integrate import quad

from argus.model import get_F_block, get_Q_block

jax.config.update("jax_enable_x64", True)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO_ROOT, "workflows", "ng15_sgwb_demo", "scripts")
DAY = 86400.0


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ou():
    return _load(os.path.join(REPO_ROOT, "workflows", "ng15_nb", "reference", "ou_psd.py"), "ng15_nb_ou_psd")


def _structure_closed_form(sigma2, gamma, tau):
    return sigma2 / gamma**2 * (tau + math.expm1(-gamma * tau) / gamma)


def _structure_from_filter(sigma2, gamma, tau):
    f12 = float(get_F_block(gamma, tau)[0, 1])
    q11 = float(get_Q_block(gamma, tau)[0, 0])
    return f12**2 * sigma2 / (2 * gamma) + sigma2 * q11


def _structure_from_psd(s1, tau, gamma):
    """∫₀^∞ S₁(f)·2(1 − cos 2πfτ) df, in units of the lower cut ``a`` so QUADPACK stays in range.

    Below ``a`` the integrand is smooth (it tends to a constant at f = 0). Above it the
    non-oscillatory part is integrated in log f and the cosine part goes to QAWF,
    QUADPACK's Fourier-integral routine.
    """
    a = min(gamma, 1.0 / tau) / (2 * math.pi) / 10.0
    norm = s1(a)
    h = lambda x: s1(a * x) / norm  # noqa: E731
    wvar = 2 * math.pi * a * tau
    low, _ = quad(lambda x: h(x) * 2 * (1 - math.cos(wvar * x)), 0, 1, epsabs=0, epsrel=1e-12, limit=500)
    flat, _ = quad(lambda u: h(math.exp(u)) * math.exp(u), 0, 40, epsabs=0, epsrel=1e-12, limit=1000)
    osc, _ = quad(h, 1, np.inf, weight="cos", wvar=wvar, limlst=200)
    return a * norm * (low + 2 * flat - 2 * osc)


@pytest.mark.parametrize("gamma,sigma2", [(1e-11, 1.0), (3e-9, 2.5e-3), (1e-7, 7.0)])
def test_velocity_variance_matches_stationary_q(ou, gamma, sigma2):
    w2s1 = lambda f: (2 * math.pi * f) ** 2 * 2.0 * ou.ou_psd_twosided(f, sigma2, gamma)  # noqa: E731
    u0 = math.log(gamma / (2 * math.pi))
    from_psd, _ = quad(lambda u: w2s1(math.exp(u)) * math.exp(u), u0 - 40, u0 + 40, epsabs=0, epsrel=1e-12, limit=1000)
    q22_inf = sigma2 * float(get_Q_block(gamma, 1e3 / gamma)[1, 1])
    assert q22_inf == pytest.approx(sigma2 / (2 * gamma), rel=1e-14)
    assert from_psd == pytest.approx(q22_inf, rel=1e-8)


# (γ [rad/s], σ², τ [days]) with γτ between 0.03 and 100, where the quadrature is well conditioned.
PSD_CASES = [(3e-9, 2.5e-3, 30.0), (3e-9, 2.5e-3, 1000.0), (1e-7, 7.0, 30.0), (1e-7, 7.0, 1000.0), (1e-11, 1.0, 1e5)]


@pytest.mark.parametrize("gamma,sigma2,tau_days", PSD_CASES)
def test_structure_function_psd_matches_closed_form(ou, gamma, sigma2, tau_days):
    tau = tau_days * DAY
    s1 = lambda f: 2.0 * ou.ou_psd_twosided(f, sigma2, gamma)  # noqa: E731
    assert _structure_from_psd(s1, tau, gamma) == pytest.approx(_structure_closed_form(sigma2, gamma, tau), rel=1e-5)


@pytest.mark.parametrize("gamma", [1e-11, 1e-10, 3e-9, 1e-7, 1e-6])
@pytest.mark.parametrize("tau_days", [1.0, 7.0, 30.0, 1000.0])
def test_structure_function_filter_matches_closed_form(gamma, tau_days):
    tau = tau_days * DAY
    got, want = _structure_from_filter(1.3, gamma, tau), _structure_closed_form(1.3, gamma, tau)
    x = gamma * tau
    if x >= 1e-3:
        assert got == pytest.approx(want, rel=1e-10)
    else:
        # get_Q_block's q11 = [dt - 2(1-e^{-x})/γ + (1-e^{-2x})/(2γ)]/γ² cancels to ~x³/3 out of
        # O(x) terms, so it carries a relative rounding error of order eps/x². At the bottom of the
        # MDC2 γ_p box (1e-12 rad/s) and a weekly step that is ~6e-4: small next to the white noise,
        # but not exact. Pinned here so a change in either direction is noticed.
        assert got == pytest.approx(want, rel=max(1e-10, 10 * np.finfo(float).eps / x**2))


def test_gw_psd_matches_ridge_reference(ou):
    ref = _load(os.path.join(SCRIPTS, "diag_ou_powerlaw_capture.py"), "diag_ou_powerlaw_capture")
    f = np.logspace(-10, -6.5, 200)
    for pivot, lga in [(-6.1, -8.5), (-13.0, -11.0), (-3.0, -6.0)]:
        got = ou.gw_psd_onesided(f, pivot, lga, pivot_f=ref.PIVOT_F)
        np.testing.assert_allclose(got, ref.ou_psd(f, pivot, lga), rtol=1e-12, atol=0)


def test_gw_psd_is_the_filters_sigma_a2(ou):
    """The ridge pivot PSD and the h_a the sampler derives give the same σ_a² = h_a² γ_a / 12."""
    from argus.jax_kalman_filter import _compute_sigma_matrix

    f = np.logspace(-10, -6.5, 50)
    for pivot, lga in [(-6.1, -8.5), (-10.0, -10.5)]:
        log10_ha = ou.log10_ha_from_pivot(pivot, lga)
        sigma_a2 = float(_compute_sigma_matrix(10.0 ** (2 * log10_ha), 10.0**lga, 1.0))
        np.testing.assert_allclose(
            ou.gw_psd_onesided(f, pivot, lga), 2.0 * ou.ou_psd_twosided(f, sigma_a2, 10.0**lga), rtol=1e-12
        )
        df = np.full_like(f, 1e-9)
        np.testing.assert_allclose(ou.ou_gw_prior(f, df, pivot, lga), ou.gw_psd_onesided(f, pivot, lga) * df, rtol=1e-12)


def test_spin_psd_matches_budget_reference(ou):
    sys.path.insert(0, SCRIPTS)
    try:
        ref = _load(os.path.join(SCRIPTS, "ng15_red_noise_budget.py"), "ng15_red_noise_budget")
    finally:
        sys.path.remove(SCRIPTS)
    f = np.logspace(-10, -6.5, 200)
    for lgp, lsp, f0 in [(-8.0, -12.0, 339.3), (-11.0, -15.0, 29.0), (-6.0, -6.0, 641.9)]:
        log10_sigma_r2 = 2 * lsp - 2 * math.log10(f0)
        np.testing.assert_allclose(
            ou.spin_psd_onesided(f, lgp, lsp, f0), ref.ou_psd_onesided(f, lgp, log10_sigma_r2), rtol=1e-12
        )
        df = np.full_like(f, 1e-9)
        np.testing.assert_allclose(
            ou.ou_rn_prior(f, df, lgp, lsp - math.log10(f0)), ou.spin_psd_onesided(f, lgp, lsp, f0) * df, rtol=1e-12
        )
