"""Laplace marginalisation of per-pulsar noise parameters for CW inference.

This module analytically marginalises the per-pulsar red noise parameters
(gamma_p, sigma_p) from the CW likelihood using a Laplace approximation.
Given CW source parameters and noise hyperparameters, each pulsar's 2D
noise posterior is independently optimised and integrated via:

    log p(data_n | CW, hyperparams) ≈ f(phi*_n) + log(2pi) - 0.5 * log(det(-H_n))

where phi*_n is the MAP of (log10_gamma_p_n, log10_ratio_n) and H_n is the
2x2 Hessian at that point. The per-pulsar independence allows vmapping
all 31 optimisations in parallel.
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

from argus.model import get_F_block, get_Q_block
from argus.gravitational_waves import (
    compute_antenna_patterns,
    compute_cw_signal_single_pulsar,
)
from argus.cw_kalman_filter import (
    _initialize_single_pulsar,
    _run_single_pulsar_filter,
)


def _per_pulsar_log_posterior(
    phi_n,
    z_tilde_n,
    h_vectors_n,
    R_scalars_n,
    dt_array_n,
    mask_n,
    P_eps_n,
    state_dim,
    gamma_p_mean,
    gamma_p_std_scaled,
    ratio_mean,
    ratio_std_scaled,
):
    """Log-posterior for a single pulsar's noise parameters.

    Parameters
    ----------
    phi_n : jax.Array
        Shape (2,): [log10_gamma_p_n, log10_ratio_n].
    z_tilde_n : jax.Array
        CW-subtracted residuals, shape (max_nobs,).
    h_vectors_n : jax.Array
        Observation vectors, shape (max_nobs, state_dim).
    R_scalars_n : jax.Array
        Measurement noise variances, shape (max_nobs,).
    dt_array_n : jax.Array
        Time differences, shape (max_nobs - 1,).
    mask_n : jax.Array
        Observation mask, shape (max_nobs,).
    P_eps_n : jax.Array
        Timing model covariance, shape (max_M, max_M).
    state_dim : int
        State vector dimension (2 + max_M).
    gamma_p_mean : float
        Hierarchical mean of log10_gamma_p.
    gamma_p_std_scaled : float
        Hierarchical std of log10_gamma_p (already scaled by 1/sqrt(Npsr)).
    ratio_mean : float
        Hierarchical mean of log10_ratio.
    ratio_std_scaled : float
        Hierarchical std of log10_ratio (already scaled by 1/sqrt(Npsr)).

    Returns
    -------
    float
        log_likelihood + log_prior for this pulsar's noise parameters.
    """
    log10_gamma_p_n = phi_n[0]
    log10_ratio_n = phi_n[1]
    log10_sigma_p_n = log10_gamma_p_n + log10_ratio_n

    gamma_p_n = 10.0**log10_gamma_p_n
    sigma_p_n = 10.0**log10_sigma_p_n
    sigma_p_sq_n = sigma_p_n**2

    # Kalman filter likelihood for this pulsar
    x0, P0 = _initialize_single_pulsar(state_dim, sigma_p_sq_n, gamma_p_n, P_eps_n)
    ll_n = _run_single_pulsar_filter(
        x0, P0, z_tilde_n, h_vectors_n, R_scalars_n, dt_array_n,
        mask_n, gamma_p_n, sigma_p_sq_n, state_dim,
    )

    # Hierarchical Normal log-prior (matching the gradient-balanced parameterisation)
    log_prior_gamma = (
        -0.5 * ((log10_gamma_p_n - gamma_p_mean) / gamma_p_std_scaled) ** 2
        - jnp.log(gamma_p_std_scaled)
    )
    log_prior_ratio = (
        -0.5 * ((log10_ratio_n - ratio_mean) / ratio_std_scaled) ** 2
        - jnp.log(ratio_std_scaled)
    )

    return ll_n + log_prior_gamma + log_prior_ratio


def _newton_step_2d(phi, grad, hess):
    """Single Newton step for a 2D problem using analytic inverse.

    Parameters
    ----------
    phi : jax.Array
        Current point, shape (2,).
    grad : jax.Array
        Gradient at phi, shape (2,).
    hess : jax.Array
        Hessian at phi, shape (2, 2).

    Returns
    -------
    jax.Array
        Updated point, shape (2,).
    """
    det = hess[0, 0] * hess[1, 1] - hess[0, 1] * hess[1, 0]
    # Regularise to avoid singular Hessian
    det = jnp.where(jnp.abs(det) < 1e-30, -1e-30, det)
    inv_hess = jnp.array([
        [hess[1, 1], -hess[0, 1]],
        [-hess[1, 0], hess[0, 0]],
    ]) / det
    return phi - inv_hess @ grad


def _optimize_and_laplace_single_pulsar(
    z_tilde_n,
    h_vectors_n,
    R_scalars_n,
    dt_array_n,
    mask_n,
    P_eps_n,
    state_dim,
    gamma_p_mean,
    gamma_p_std_scaled,
    ratio_mean,
    ratio_std_scaled,
    n_newton_steps,
):
    """Find MAP and compute Laplace correction for one pulsar's noise params.

    Parameters
    ----------
    z_tilde_n : jax.Array
        CW-subtracted residuals for pulsar n.
    h_vectors_n : jax.Array
        Observation vectors for pulsar n.
    R_scalars_n : jax.Array
        Measurement noise for pulsar n.
    dt_array_n : jax.Array
        Time differences for pulsar n.
    mask_n : jax.Array
        Observation mask for pulsar n.
    P_eps_n : jax.Array
        Timing model covariance for pulsar n.
    state_dim : int
        State vector dimension.
    gamma_p_mean : float
        Hierarchical mean of log10_gamma_p.
    gamma_p_std_scaled : float
        Hierarchical std of log10_gamma_p (scaled by 1/sqrt(Npsr)).
    ratio_mean : float
        Hierarchical mean of log10_ratio.
    ratio_std_scaled : float
        Hierarchical std of log10_ratio (scaled by 1/sqrt(Npsr)).
    n_newton_steps : int
        Number of Newton iterations.

    Returns
    -------
    float
        Laplace-approximated marginal log-likelihood for this pulsar.
    """
    # Objective function (closure over per-pulsar data)
    def objective(phi):
        return _per_pulsar_log_posterior(
            phi, z_tilde_n, h_vectors_n, R_scalars_n, dt_array_n,
            mask_n, P_eps_n, state_dim,
            gamma_p_mean, gamma_p_std_scaled, ratio_mean, ratio_std_scaled,
        )

    grad_fn = jax.grad(objective)
    hess_fn = jax.hessian(objective)

    # Start from the prior mode
    phi_init = jnp.array([gamma_p_mean, ratio_mean])

    # Newton optimisation via fori_loop with damping for stability
    def body(_, phi):
        g = grad_fn(phi)
        H = hess_fn(phi)
        phi_new = _newton_step_2d(phi, g, H)
        # Clamp step size to avoid divergence (max 2.0 in each coordinate)
        delta = phi_new - phi
        delta = jnp.clip(delta, -2.0, 2.0)
        phi_new = phi + delta
        # If Newton produced NaN, fall back to current point
        phi_new = jnp.where(jnp.all(jnp.isfinite(phi_new)), phi_new, phi)
        return phi_new

    phi_star = lax.fori_loop(0, n_newton_steps, body, phi_init)

    # If optimisation diverged, fall back to prior mode
    phi_star = jnp.where(
        jnp.all(jnp.isfinite(phi_star)), phi_star, phi_init
    )

    # Evaluate objective and Hessian at MAP
    f_star = objective(phi_star)
    H_star = hess_fn(phi_star)

    # Laplace correction: f(phi*) + log(2pi) - 0.5 * log(det(-H))
    neg_H = -H_star
    det_neg_H = neg_H[0, 0] * neg_H[1, 1] - neg_H[0, 1] * neg_H[1, 0]
    # Clamp determinant to avoid log(0) or log(negative)
    det_neg_H = jnp.maximum(det_neg_H, 1e-30)
    log_det = jnp.log(det_neg_H)

    # Guard against non-finite result
    result = f_star + jnp.log(2.0 * jnp.pi) - 0.5 * log_det
    return jnp.where(jnp.isfinite(result), result, f_star)


@partial(jax.jit, static_argnames=(
    "Npsr", "state_dim", "include_pulsar_term",
    "phase_parameterization", "n_newton_steps",
))
def laplace_marginalised_likelihood(
    alpha_gw, delta_gw, f_gw, h0, cos_iota, psi, Phi0,
    chi, gamma_p_mean, gamma_p_std_scaled, ratio_mean, ratio_std_scaled,
    EFAC, EQUAD,
    toas, residuals, errors, mask, dt, H, P_eps,
    pulsar_ra, pulsar_dec, pulsar_distances,
    Npsr, state_dim,
    include_pulsar_term, phase_parameterization,
    n_newton_steps,
):
    """Compute the Laplace-marginalised CW log-likelihood over all pulsars.

    Mirrors the structure of _cw_likelihood (steps 1-4: antenna patterns,
    CW signal, subtraction, R_scalars) but replaces the direct KF evaluation
    with per-pulsar Laplace-marginalised optimisation.

    Parameters
    ----------
    alpha_gw, delta_gw, f_gw, h0, cos_iota, psi, Phi0 : float
        CW source parameters.
    chi : jax.Array
        Per-pulsar phase parameters, shape (Npsr,).
    gamma_p_mean : float
        Hierarchical mean of log10_gamma_p.
    gamma_p_std_scaled : float
        Hierarchical std of log10_gamma_p (scaled by 1/sqrt(Npsr)).
    ratio_mean : float
        Hierarchical mean of log10_ratio.
    ratio_std_scaled : float
        Hierarchical std of log10_ratio (scaled by 1/sqrt(Npsr)).
    EFAC : jax.Array
        Per-pulsar error scale factors, shape (Npsr,).
    EQUAD : jax.Array
        Per-pulsar quadrature noise, shape (Npsr,).
    toas, residuals, errors, mask, dt, H, P_eps : jax.Array
        Per-pulsar data arrays (padded).
    pulsar_ra, pulsar_dec, pulsar_distances : jax.Array
        Pulsar sky positions and distances.
    Npsr : int
        Number of pulsars.
    state_dim : int
        State vector dimension.
    include_pulsar_term : bool
        Whether to include pulsar term.
    phase_parameterization : bool
        Whether to use phase parameterisation for pulsar term.
    n_newton_steps : int
        Number of Newton iterations for inner optimisation.

    Returns
    -------
    float
        Total Laplace-marginalised log-likelihood.
    """
    # 1. Compute antenna patterns
    F_plus, F_cross = compute_antenna_patterns(
        pulsar_ra, pulsar_dec, alpha_gw, delta_gw, psi
    )

    # 2. Compute CW signal (same three-way branch as _cw_likelihood)
    if include_pulsar_term and phase_parameterization:
        from argus.gravitational_waves import compute_cw_signal_single_pulsar_phase
        cw_signal = jax.vmap(
            lambda t, fp, fc, ch: compute_cw_signal_single_pulsar_phase(
                t, f_gw, h0, cos_iota, Phi0, fp, fc, ch
            )
        )(toas, F_plus, F_cross, chi)
    elif include_pulsar_term:
        from argus.gravitational_waves import gw_propagation_direction, pulsar_direction
        n_hat = gw_propagation_direction(alpha_gw, delta_gw)
        geometric_factors = jax.vmap(
            lambda ra, dec: 1.0 + jnp.dot(n_hat, pulsar_direction(ra, dec))
        )(pulsar_ra, pulsar_dec)
        cw_signal = jax.vmap(
            lambda t, fp, fc, d, g: compute_cw_signal_single_pulsar(
                t, f_gw, h0, cos_iota, Phi0, fp, fc, d, g
            )
        )(toas, F_plus, F_cross, pulsar_distances, geometric_factors)
    else:
        cw_signal = jax.vmap(
            lambda t, fp, fc, d, g: compute_cw_signal_single_pulsar(
                t, f_gw, h0, cos_iota, Phi0, fp, fc, d, g
            )
        )(toas, F_plus, F_cross, pulsar_distances, jnp.zeros(Npsr))

    # 3. Subtract CW signal
    z_tilde = residuals - cw_signal

    # 4. Compute per-pulsar measurement noise
    R_scalars = jnp.square(EFAC[:, None] * errors) + jnp.square(EQUAD[:, None])

    # 5. Laplace-marginalise per-pulsar noise via vmap
    laplace_ll = jax.vmap(
        lambda z, h_v, R, dt_a, m, pe: _optimize_and_laplace_single_pulsar(
            z, h_v, R, dt_a, m, pe,
            state_dim,
            gamma_p_mean, gamma_p_std_scaled, ratio_mean, ratio_std_scaled,
            n_newton_steps,
        )
    )(z_tilde, H, R_scalars, dt, mask, P_eps)

    return jnp.sum(laplace_ll)
