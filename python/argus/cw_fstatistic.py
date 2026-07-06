"""Coherent Earth-term F-statistic (F_e) and Bayesian B-statistic for CW detection.

This is an *additive* module: it does not modify any existing (validated) Argus code.
It reuses the per-pulsar scalar Kalman-filter building blocks from
:mod:`argus.cw_kalman_filter` to build a fast, sampling-free continuous-wave (CW)
*detection* statistic — the field-standard object for "is there a CW, and how
significant", as opposed to full parameter estimation (which needs MCMC that
struggles with the multimodal CW posterior).

Method
------
The Earth-term CW timing residual is **linear in four global amplitude
coefficients** ``a = (a1,a2,a3,a4)`` (functions of ``h0, Phi0, psi, cos_iota``),
with four basis functions that depend only on the *intrinsic* parameters
(frequency ``f_gw``, sky location) and the per-pulsar antenna patterns::

    res_a(t) = a1 F+0_a sin(Omega t) + a2 F+0_a cos(Omega t)
             + a3 Fx0_a sin(Omega t) + a4 Fx0_a cos(Omega t)

where ``Omega = 2 pi f_gw`` and ``F+0_a, Fx0_a`` are the ``psi=0`` antenna patterns
of pulsar ``a`` (psi is folded into the amplitudes). Define the **Kalman-whitened
inner product** ``<u|v> = u^T C^-1 v``, where ``C`` is Argus's full per-pulsar
noise + timing-marginalisation covariance. With ``X_i = sum_a <d_a|A_i^a>`` and
``M_ij = sum_a <A_i^a|A_j^a>``:

* Frequentist:  ``2 F_e = X^T M^-1 X``  (under H0, ``2F_e ~ chi^2_4``); ``a_hat = M^-1 X``.
* Bayesian:     ``ln B = 1/2 X^T (M + Sigma^-1)^-1 X - 1/2 ln det(I + Sigma M)``
  for a Gaussian amplitude prior ``Sigma = sigma_a^2 I``. This integrates over the
  amplitude prior, so it avoids the ``h0=0`` nesting problem of Savage-Dickey.

Whitening
---------
Argus's scalar filter already produces, per observation, an innovation ``nu_k`` and
variance ``S_k`` (:func:`argus.cw_kalman_filter._cw_scalar_update`). The innovation
map is **linear** in the input and the ``S_k``/gains are **data-independent**, so
``<u|v> = sum_k nu_k(u) nu_k(v) / S_k`` (masked over valid observations). We whiten
the data once per pulsar; the two time-templates ``sin/cos(Omega t)`` once per
pulsar per frequency; and the sky only enters afterward through the scalar antenna
patterns — a large speed-up over whitening all 4 basis functions per (f, sky).
"""

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax, vmap

from argus.cw_kalman_filter import (
    _cw_scalar_update,
    _build_F_single,
    _build_Q_single,
    _initialize_single_pulsar,
)
from argus.gravitational_waves import antenna_pattern_single


# --------------------------------------------------------------------------- #
# Kalman whitener: (nu, S) innovation sequences for an arbitrary input vector.
# --------------------------------------------------------------------------- #
@partial(jax.jit, static_argnums=(7,))
def whiten_single_pulsar(z, x0, P0, h_vectors, R_scalars, dt_array,
                         gamma_p, state_dim, sigma_p_sq):
    """Innovation sequence ``(nu, S)`` for input ``z`` under the noise-only filter.

    Mirrors :func:`argus.cw_kalman_filter._run_single_pulsar_filter` (obs-0 update,
    then ``lax.scan``) but returns the per-observation innovation ``nu`` and its
    variance ``S`` instead of the summed log-likelihood. ``S`` is independent of
    ``z``; ``nu`` is linear in ``z`` — this is what makes the inner-product
    decomposition exact.

    Parameters mirror the filter. ``z`` has shape ``(max_nobs,)``. Returns two
    arrays of shape ``(max_nobs,)``.
    """
    Q_eps = jnp.zeros((state_dim - 2, state_dim - 2))

    x, P, nu0, S0 = _cw_scalar_update(x0, P0, h_vectors[0], R_scalars[0], z[0])

    def step(carry, inp):
        x, P = carry
        zk, h, R, dt = inp
        F = _build_F_single(gamma_p, state_dim, dt)
        Q = _build_Q_single(gamma_p, sigma_p_sq, state_dim, dt, Q_eps)
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        x_new, P_new, nu, S = _cw_scalar_update(x_pred, P_pred, h, R, zk)
        return (x_new, P_new), (nu, S)

    inputs = (z[1:], h_vectors[1:], R_scalars[1:], dt_array)
    _, (nu_arr, S_arr) = lax.scan(step, (x, P), inputs)

    nu = jnp.concatenate([jnp.atleast_1d(nu0), nu_arr])
    S = jnp.concatenate([jnp.atleast_1d(S0), S_arr])
    return nu, S


def _inner(nu_u, nu_v, S, mask):
    """Masked whitened inner product ``<u|v> = sum_k mask_k nu_u nu_v / S``."""
    return jnp.sum(mask * nu_u * nu_v / S)


# --------------------------------------------------------------------------- #
# Per-pulsar template Gram matrices, as a function of frequency.
# --------------------------------------------------------------------------- #
def pulsar_template_products(kf, n, gamma_p_n, sigma_p_sq_n, efac_n, equad_n, f_grid):
    """Whitened inner products of the data and the sin/cos templates for one pulsar.

    Returns a dict of arrays over the frequency grid:
      Ids, Idc : <data|sin>, <data|cos>            shape (Nf,)
      Iss, Icc, Ics : <sin|sin>, <cos|cos>, <sin|cos>   shape (Nf,)
    plus Idd = <data|data> (scalar, frequency-independent) for reference.

    Sky location enters later via the scalar antenna patterns, so these
    frequency-only products are all that is needed per pulsar.
    """
    state_dim = kf.state_dim
    x0, P0 = _initialize_single_pulsar(state_dim, sigma_p_sq_n, gamma_p_n, _peps_block(kf, n))

    h_vectors = kf.jax_H[n]
    toas = kf.jax_toas[n]
    errors = kf.jax_errors[n]
    mask = kf.jax_mask[n]
    dt = kf.jax_dt[n]
    R = (efac_n * errors) ** 2 + equad_n ** 2

    # Data whitening (frequency-independent) -> nu_d, S (S reused for templates).
    nu_d, S = whiten_single_pulsar(kf.jax_residuals[n], x0, P0, h_vectors, R, dt,
                                   gamma_p_n, state_dim, sigma_p_sq_n)
    Idd = _inner(nu_d, nu_d, S, mask)

    def per_freq(f):
        omega = 2.0 * jnp.pi * f
        s = jnp.sin(omega * toas) * mask  # zero padded entries (mask applied)
        c = jnp.cos(omega * toas) * mask
        nu_s, _ = whiten_single_pulsar(s, x0, P0, h_vectors, R, dt,
                                       gamma_p_n, state_dim, sigma_p_sq_n)
        nu_c, _ = whiten_single_pulsar(c, x0, P0, h_vectors, R, dt,
                                       gamma_p_n, state_dim, sigma_p_sq_n)
        return jnp.array([
            _inner(nu_d, nu_s, S, mask),   # Ids
            _inner(nu_d, nu_c, S, mask),   # Idc
            _inner(nu_s, nu_s, S, mask),   # Iss
            _inner(nu_c, nu_c, S, mask),   # Icc
            _inner(nu_s, nu_c, S, mask),   # Ics
        ])

    prods = vmap(per_freq)(f_grid)  # (Nf, 5)
    return {
        "Ids": prods[:, 0], "Idc": prods[:, 1],
        "Iss": prods[:, 2], "Icc": prods[:, 3], "Ics": prods[:, 4],
        "Idd": Idd,
    }


def _peps_block(kf, n):
    """The (max_M x max_M) padded timing covariance the filter uses for pulsar n."""
    # kf.jax_P_eps is already padded to (Npsr, max_M, max_M) with large diagonal on
    # padded dims (see _build_per_pulsar_P_eps_padded). Use it as-is.
    return kf.jax_P_eps[n]


# --------------------------------------------------------------------------- #
# Assemble X, M at a sky location from per-pulsar template products + antenna.
# --------------------------------------------------------------------------- #
def assemble_XM(products, Fp, Fx):
    """Build the 4-vector X and 4x4 matrix M at one (frequency, sky) point.

    products : per-pulsar dict arrays already selected at a single frequency, with
               keys Ids, Idc, Iss, Icc, Ics each shape (Npsr,).
    Fp, Fx   : psi=0 antenna patterns per pulsar at this sky location, shape (Npsr,).

    Basis order A = [F+ sin, F+ cos, Fx sin, Fx cos].
    """
    Ids, Idc = products["Ids"], products["Idc"]
    Iss, Icc, Ics = products["Iss"], products["Icc"], products["Ics"]

    # X_i = sum_a antenna_i * (data|template_i)
    X = jnp.array([
        jnp.sum(Fp * Ids),
        jnp.sum(Fp * Idc),
        jnp.sum(Fx * Ids),
        jnp.sum(Fx * Idc),
    ])

    pp = Fp * Fp
    xx = Fx * Fx
    px = Fp * Fx
    # M blocks (symmetric). Rows/cols: [F+s, F+c, Fxs, Fxc]
    M = jnp.array([
        [jnp.sum(pp * Iss), jnp.sum(pp * Ics), jnp.sum(px * Iss), jnp.sum(px * Ics)],
        [jnp.sum(pp * Ics), jnp.sum(pp * Icc), jnp.sum(px * Ics), jnp.sum(px * Icc)],
        [jnp.sum(px * Iss), jnp.sum(px * Ics), jnp.sum(xx * Iss), jnp.sum(xx * Ics)],
        [jnp.sum(px * Ics), jnp.sum(px * Icc), jnp.sum(xx * Ics), jnp.sum(xx * Icc)],
    ])
    return X, M


def f_statistic(X, M, ridge=1e-30):
    """Coherent F-statistic: returns (twoF, a_hat). ``2F = X^T M^-1 X``."""
    M_reg = M + ridge * jnp.eye(4)
    a_hat = jnp.linalg.solve(M_reg, X)
    twoF = X @ a_hat
    return twoF, a_hat


def b_statistic(X, M, sigma_a, ridge=1e-30):
    """Bayesian B-statistic: ln(evidence_signal/evidence_null) with Gaussian
    amplitude prior of width ``sigma_a`` (isotropic, cov = sigma_a^2 I)."""
    Sigma_inv = jnp.eye(4) / sigma_a ** 2
    A = M + Sigma_inv + ridge * jnp.eye(4)
    quad = 0.5 * X @ jnp.linalg.solve(A, X)
    # ln det(I + Sigma M) = ln det(I + sigma_a^2 M)
    sign, logdet = jnp.linalg.slogdet(jnp.eye(4) + sigma_a ** 2 * M)
    return quad - 0.5 * logdet


# --------------------------------------------------------------------------- #
# Full frequency x sky scan.
# --------------------------------------------------------------------------- #
def scan_grid(kf, gamma_p, sigma_p, efac, equad, f_grid, ra_grid, sindec_grid,
              sigma_a=1e-6):
    """Evaluate 2F_e and lnB over a frequency x sky grid.

    Parameters
    ----------
    kf : CWKalmanFilter
        Initialised filter (provides padded data/H/P_eps and pulsar sky positions).
    gamma_p, sigma_p, efac, equad : array (Npsr,)
        Fixed per-pulsar noise parameters.
    f_grid : array (Nf,)
        GW frequencies (Hz).
    ra_grid, sindec_grid : array (Nra,), (Ndec,)
        Sky grid axes (right ascension in rad, sin(declination)).
    sigma_a : float
        Gaussian amplitude-prior width for the B-statistic.

    Returns
    -------
    dict with:
        twoF : (Nf, Nra, Ndec)
        lnB  : (Nf, Nra, Ndec)
        f_grid, ra_grid, dec_grid
    """
    Npsr = kf.Npsr
    sigma_p_sq = jnp.asarray(sigma_p) ** 2
    gamma_p = jnp.asarray(gamma_p)
    efac = jnp.asarray(efac)
    equad = jnp.asarray(equad)
    f_grid = jnp.asarray(f_grid)

    # 1. Per-pulsar template products over the frequency grid (the expensive step).
    per_pulsar = [
        pulsar_template_products(kf, n, gamma_p[n], sigma_p_sq[n], efac[n], equad[n], f_grid)
        for n in range(Npsr)
    ]
    # Stack into (Npsr, Nf) arrays per key.
    keys = ["Ids", "Idc", "Iss", "Icc", "Ics"]
    stacked = {k: jnp.stack([per_pulsar[n][k] for n in range(Npsr)], axis=0) for k in keys}

    # 2. Antenna patterns (psi=0) per pulsar over the sky grid.
    pra = kf.pulsar_ra
    pdec = kf.pulsar_dec
    dec_grid = jnp.arcsin(jnp.asarray(sindec_grid))

    def antenna_at(alpha, delta):
        Fp, Fx = vmap(lambda ra, dec: antenna_pattern_single(ra, dec, alpha, delta, 0.0))(pra, pdec)
        return Fp, Fx  # each (Npsr,)

    # 3. Evaluate statistics over (f, ra, dec).
    def at_point(fi, alpha, delta):
        products = {k: stacked[k][:, fi] for k in keys}  # each (Npsr,)
        Fp, Fx = antenna_at(alpha, delta)
        X, M = assemble_XM(products, Fp, Fx)
        twoF, _ = f_statistic(X, M)
        lnB = b_statistic(X, M, sigma_a)
        return twoF, lnB

    Nf = f_grid.shape[0]
    ra_grid = jnp.asarray(ra_grid)
    # vmap over sky for each frequency (Python loop over Nf keeps memory modest).
    def sky_map(fi):
        f_over_ra = vmap(lambda a: vmap(lambda d: at_point(fi, a, d))(dec_grid))(ra_grid)
        return f_over_ra  # tuple of (Nra, Ndec)

    twoF_all = np.zeros((Nf, ra_grid.shape[0], dec_grid.shape[0]))
    lnB_all = np.zeros_like(twoF_all)
    for fi in range(Nf):
        twoF_ra_dec, lnB_ra_dec = sky_map(fi)
        twoF_all[fi] = np.asarray(twoF_ra_dec)
        lnB_all[fi] = np.asarray(lnB_ra_dec)

    return {
        "twoF": twoF_all,
        "lnB": lnB_all,
        "f_grid": np.asarray(f_grid),
        "ra_grid": np.asarray(ra_grid),
        "dec_grid": np.asarray(dec_grid),
    }
