"""The MDC2 diffuse golden is the α → ∞ limit of the informative filter.

The informative filter's timing prior precision scales as P_eps⁻¹ = P_base⁻¹ / α, so

    lnL_inf(α) − ½ logdet P_base⁻¹ + ½ M_sum ln α  →  lnL_diffuse   as α → ∞,

with an O(1/α) error. This is an independent check of the diffuse branch on the real MDC2
case, where Λ = A is badly conditioned (equilibrated condition number ~1e12). The small
synthetic checks in test_jax_kalman_filter.py::TestDiffuseFilter missed a jitter of
1e-9·trace(Λ)/n that biased this value by −34.7 nat; this test would have caught it.

Measured on 2026-10-17: the error is −0.18, −2.2e-3 and +3.3e-4 nat at α = 1e6, 1e8 and
1e10. Both sides carry ~1e-3 nat of compile-dependent rounding noise from Λ's
conditioning, so the bound at α = 1e10 is 5e-3 nat. The ordering against α = 1e6 is the
robust convergence check.
"""

import os
import pickle

import jax.numpy as jnp
import numpy as np
import pytest

from argus import bayesian_inference
from argus import io_manager
from argus import jax_kalman_filter as jk
from argus.utils import get_efac_equad_injections, get_psr_noise_injections


@pytest.fixture(scope="module")
def mdc2():
    """The MDC2 golden dataset and its recorded parameter vector."""

    class MockConfig:
        def get(self, section, key, fallback=None):
            return fallback

    io_manager.setup_single_logger(MockConfig(), enable_file_logging=False)

    here = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(here, "data/processed_pulsar_data.pkl")
    noise_path = os.path.join(here, "data/noise_parameters.json")
    spin_path = os.path.join(here, "data/spin_injections.pkl")
    for path in (data_path, noise_path, spin_path):
        if not os.path.exists(path):
            pytest.skip(f"Test data not found: {path}")

    with open(data_path, "rb") as f:
        pulsar_data = pickle.load(f)
    efac, equad = get_efac_equad_injections(noise_path, excluded_psrs=["J1640+2224"])
    sigma_p, gamma_p = get_psr_noise_injections(spin_path, excluded_psrs=["J1640+2224"])

    gamma_a = 1e-9
    params = bayesian_inference.Parameters(
        γa=gamma_a,
        ha=1e-15,
        log10_gamma_a=jnp.log10(gamma_a),
        γp=gamma_p,
        σp=sigma_p,
        EFAC=efac,
        EQUAD=equad,
    )
    return pulsar_data, params


def test_diffuse_golden_is_the_large_alpha_limit(mdc2):
    data, params = mdc2
    kf = jk.JaxKalmanFilter(
        data=data, use_gw=True, use_marginal=True, timing_prior="diffuse"
    )
    ll_diffuse = float(kf.get_likelihood(params))
    assert abs(ll_diffuse - 59454.72) < 1e-2

    kf_base = jk.JaxKalmanFilter(
        data=data, use_marginal=True, prior_scale=1.0, timing_prior="informative"
    )
    _, logdet_base = np.linalg.slogdet(np.asarray(kf_base.P_eps_inv))
    M_sum = int(kf.M_sum)

    def error(alpha):
        kf_a = jk.JaxKalmanFilter(
            data=data, use_marginal=True, prior_scale=alpha, timing_prior="informative"
        )
        ll_a = float(kf_a.get_likelihood(params))
        return ll_a - 0.5 * logdet_base + 0.5 * M_sum * np.log(alpha) - ll_diffuse

    err = {a: abs(error(a)) for a in (1e6, 1e10)}
    assert err[1e10] < err[1e6]
    assert err[1e10] < 5e-3, err
