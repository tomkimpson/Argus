"""Tests for the CW F-statistic / B-statistic module (argus.cw_fstatistic).

The central correctness test ties the Kalman-*whitened* inner products used by the
F-statistic to Argus's full CW likelihood: for any Earth-term signal ``s``,

    logL(signal=s) - logL(signal=0)  ==  <d|s> - 1/2 <s|s>

where ``<u|v> = sum_k nu_k(u) nu_k(v) / S_k`` comes from the whitener. If this holds
to numerical precision across a multi-pulsar setup, the F-statistic
``2F_e = X^T M^-1 X`` (a maximisation of exactly this quadratic over the linear
amplitudes) is correct by construction.
"""

import logging

import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
import pytest

jax.config.update("jax_enable_x64", True)

# Initialise the argus logger (CWKalmanFilter requires a handler present).
_logger = logging.getLogger("argus")
if not _logger.handlers:
    _logger.addHandler(logging.NullHandler())
    _logger.setLevel(logging.WARNING)

from argus.cw_kalman_filter import CWKalmanFilter
from argus.bayesian_inference import CWParameters
from argus.gravitational_waves import (
    antenna_pattern_single,
    compute_cw_signal_single_pulsar,
)
from argus.cw_fstatistic import (
    whiten_single_pulsar,
    _initialize_single_pulsar,
    _peps_block,
    pulsar_template_products,
    assemble_XM,
    f_statistic,
    b_statistic,
)


# --------------------------------------------------------------------------- #
# Synthetic multi-pulsar data (mirrors test_cw_kalman_filter.simple_cw_data).
# --------------------------------------------------------------------------- #
def make_data(seed=0, Npsr=3):
    rng = np.random.default_rng(seed)
    nobs = [60, 55, 50][:Npsr]
    toas_list, res_list, err_list = [], [], []
    for n in range(Npsr):
        t = np.sort(rng.uniform(0, 10 * 365.25 * 86400.0, nobs[n]))
        toas_list.append(t)
        res_list.append(rng.normal(0, 1e-7, nobs[n]))
        err_list.append(np.full(nobs[n], 1e-7))
    metadata = pd.DataFrame({
        "name": [f"J{i:04d}+0000" for i in range(Npsr)],
        "dim_M": [4, 5, 4][:Npsr],
        "RA": [0.5, 1.5, 3.0][:Npsr],
        "DEC": [0.3, -0.2, 0.8][:Npsr],
        "F0": [200.0, 300.0, 150.0][:Npsr],
        "distance_kpc": [1.0, 1.0, 1.0][:Npsr],
    })
    # Realistic design matrix + P_eps, exactly as LoadWidebandPulsarData does:
    # scale columns to unit norm, P_eps = inv(M_scaled^T N^-1 M_scaled). Using an
    # arbitrary (large) P_eps instead makes the timing-model uncertainty swamp the
    # white noise, rendering any signal undetectable.
    design_matrices, P_eps = [], []
    for n in range(Npsr):
        M_n = int(metadata["dim_M"].iloc[n])
        M = rng.standard_normal((nobs[n], M_n))
        M_scaled = M / np.sqrt(np.sum(M ** 2, axis=0))
        Ninv = np.diag(1.0 / np.asarray(err_list[n]) ** 2)
        P_eps.append(np.linalg.inv(M_scaled.T @ Ninv @ M_scaled))
        design_matrices.append(M_scaled)
    data = {
        "processed_residuals": {
            "toas": toas_list, "residuals": res_list, "errors": err_list,
            "n_obs": np.array(nobs),
        },
        "metadata": metadata,
        "design_matrices": design_matrices,
        "parameter_covariances": P_eps,
        "hd_correlation": None,
    }
    return data


NOISE = dict(
    gamma_p=lambda N: jnp.full(N, 1e-8),
    sigma_p=lambda N: jnp.full(N, 1e-14),
    efac=lambda N: jnp.ones(N),
    equad=lambda N: jnp.full(N, 1e-9),
)


def _pulsar_inner(kf, n, u, v, gamma_p_n, sigma_p_sq_n, efac_n, equad_n):
    """<u|v> for pulsar n via the whitener (u, v are length-max_nobs vectors)."""
    x0, P0 = _initialize_single_pulsar(kf.state_dim, sigma_p_sq_n, gamma_p_n, _peps_block(kf, n))
    R = (efac_n * kf.jax_errors[n]) ** 2 + equad_n ** 2
    nu_u, S = whiten_single_pulsar(u, x0, P0, kf.jax_H[n], R, kf.jax_dt[n],
                                   gamma_p_n, kf.state_dim, sigma_p_sq_n)
    nu_v, _ = whiten_single_pulsar(v, x0, P0, kf.jax_H[n], R, kf.jax_dt[n],
                                   gamma_p_n, kf.state_dim, sigma_p_sq_n)
    m = kf.jax_mask[n]
    return float(jnp.sum(m * nu_u * nu_v / S))


def _signal_vectors(kf, params):
    """Per-pulsar Earth-term signal vectors (padded to max_nobs), as the KF forms them."""
    N = kf.Npsr
    sigs = np.zeros((N, kf.max_nobs))
    for n in range(N):
        Fp, Fc = antenna_pattern_single(kf.pulsar_ra[n], kf.pulsar_dec[n],
                                        params.alpha_gw, params.delta_gw, params.psi)
        s = compute_cw_signal_single_pulsar(
            kf.jax_toas[n], params.f_gw, params.h0, params.cos_iota, params.Phi0,
            Fp, Fc, pulsar_distance=0.0, geometric_factor=0.0)
        sigs[n] = np.asarray(s) * np.asarray(kf.jax_mask[n])
    return jnp.array(sigs)


def _params(N, h0, f_gw=2e-8, alpha=2.0, delta=0.3, cos_iota=0.3, psi=0.7, Phi0=1.0):
    return CWParameters(
        alpha_gw=alpha, delta_gw=delta, f_gw=f_gw, h0=h0, cos_iota=cos_iota,
        psi=psi, Phi0=Phi0, chi=jnp.zeros(N),
        gamma_p=NOISE["gamma_p"](N), sigma_p=NOISE["sigma_p"](N),
        EFAC=NOISE["efac"](N), EQUAD=NOISE["equad"](N),
    )


# --------------------------------------------------------------------------- #
# Core test: whitener inner products == Kalman likelihood ratio.
# --------------------------------------------------------------------------- #
def test_whitener_matches_kalman_likelihood_ratio():
    data = make_data(seed=1)
    kf = CWKalmanFilter(data, include_pulsar_term=False, phase_parameterization=True)
    N = kf.Npsr

    params = _params(N, h0=3e-14)
    params0 = _params(N, h0=0.0)

    # Actual likelihood ratio from the full Kalman filter.
    dLL_actual = float(kf.get_likelihood(params)) - float(kf.get_likelihood(params0))

    # Predicted from whitened inner products: sum_a [<d|s> - 1/2 <s|s>].
    s = _signal_vectors(kf, params)
    gp = NOISE["gamma_p"](N); sps = NOISE["sigma_p"](N) ** 2
    ef = NOISE["efac"](N); eq = NOISE["equad"](N)
    dLL_pred = 0.0
    for n in range(N):
        d = kf.jax_residuals[n]
        dds = _pulsar_inner(kf, n, d, s[n], gp[n], sps[n], ef[n], eq[n])
        sss = _pulsar_inner(kf, n, s[n], s[n], gp[n], sps[n], ef[n], eq[n])
        dLL_pred += dds - 0.5 * sss

    assert np.isfinite(dLL_actual)
    assert abs(dLL_pred - dLL_actual) <= 1e-6 * (abs(dLL_actual) + 1.0), (
        f"whitener ratio {dLL_pred:.6e} != Kalman ratio {dLL_actual:.6e}"
    )


def test_inner_product_is_bilinear():
    data = make_data(seed=2)
    kf = CWKalmanFilter(data, include_pulsar_term=False)
    N = kf.Npsr
    gp = NOISE["gamma_p"](N); sps = NOISE["sigma_p"](N) ** 2
    ef = NOISE["efac"](N); eq = NOISE["equad"](N)
    rng = np.random.default_rng(3)
    u = jnp.array(rng.standard_normal(kf.max_nobs) * 1e-7) * kf.jax_mask[0]
    v = jnp.array(rng.standard_normal(kf.max_nobs) * 1e-7) * kf.jax_mask[0]
    uu = _pulsar_inner(kf, 0, u, u, gp[0], sps[0], ef[0], eq[0])
    vv = _pulsar_inner(kf, 0, v, v, gp[0], sps[0], ef[0], eq[0])
    uv = _pulsar_inner(kf, 0, u, v, gp[0], sps[0], ef[0], eq[0])
    uvuv = _pulsar_inner(kf, 0, u + v, u + v, gp[0], sps[0], ef[0], eq[0])
    assert abs(uvuv - (uu + 2 * uv + vv)) <= 1e-6 * (abs(uvuv) + 1.0)


# --------------------------------------------------------------------------- #
# F-statistic detects an injected signal and stays small under the null.
# --------------------------------------------------------------------------- #
def _twoF_at(kf, f_gw, alpha, delta, noise):
    N = kf.Npsr
    f_grid = jnp.array([f_gw])
    prods = [pulsar_template_products(kf, n, noise["gamma_p"](N)[n],
                                      noise["sigma_p"](N)[n] ** 2,
                                      noise["efac"](N)[n], noise["equad"](N)[n], f_grid)
             for n in range(N)]
    keys = ["Ids", "Idc", "Iss", "Icc", "Ics"]
    products = {k: jnp.array([prods[n][k][0] for n in range(N)]) for k in keys}
    Fp, Fx = [], []
    for n in range(N):
        fp, fx = antenna_pattern_single(kf.pulsar_ra[n], kf.pulsar_dec[n], alpha, delta, 0.0)
        Fp.append(fp); Fx.append(fx)
    X, M = assemble_XM(products, jnp.array(Fp), jnp.array(Fx))
    twoF, a_hat = f_statistic(X, M)
    return float(twoF), X, M


def test_fstatistic_detects_injection_and_null_is_small():
    N = 3
    data = make_data(seed=5, Npsr=N)
    # Inject a strong Earth-term CW into the residuals.
    inj = _params(N, h0=3e-13, f_gw=2e-8, alpha=2.0, delta=0.3)
    kf_noise = CWKalmanFilter(data, include_pulsar_term=False)
    s = _signal_vectors(kf_noise, inj)
    for n in range(N):
        nobs = int(data["processed_residuals"]["n_obs"][n])
        data["processed_residuals"]["residuals"][n] = (
            np.asarray(data["processed_residuals"]["residuals"][n])
            + np.asarray(s[n])[:nobs]
        )
    kf = CWKalmanFilter(data, include_pulsar_term=False)

    twoF_sig, _, _ = _twoF_at(kf, inj.f_gw, inj.alpha_gw, inj.delta_gw, NOISE)

    # Null: no-injection data.
    data0 = make_data(seed=6, Npsr=N)
    kf0 = CWKalmanFilter(data0, include_pulsar_term=False)
    twoF_null, _, _ = _twoF_at(kf0, inj.f_gw, inj.alpha_gw, inj.delta_gw, NOISE)

    assert twoF_sig > 50.0, f"expected large 2F_e for strong injection, got {twoF_sig}"
    assert twoF_null < 30.0, f"null 2F_e unexpectedly large: {twoF_null}"
    assert twoF_sig > 5.0 * twoF_null


def test_bstatistic_finite_and_favours_signal():
    N = 3
    data = make_data(seed=7, Npsr=N)
    inj = _params(N, h0=3e-13, f_gw=2e-8)
    kf_noise = CWKalmanFilter(data, include_pulsar_term=False)
    s = _signal_vectors(kf_noise, inj)
    for n in range(N):
        nobs = int(data["processed_residuals"]["n_obs"][n])
        data["processed_residuals"]["residuals"][n] = (
            np.asarray(data["processed_residuals"]["residuals"][n]) + np.asarray(s[n])[:nobs]
        )
    kf = CWKalmanFilter(data, include_pulsar_term=False)
    _, X, M = _twoF_at(kf, inj.f_gw, inj.alpha_gw, inj.delta_gw, NOISE)
    lnB = float(b_statistic(X, M, sigma_a=1e-6))
    assert np.isfinite(lnB)
    assert lnB > 0.0, f"expected lnB>0 for a strong injection, got {lnB}"
