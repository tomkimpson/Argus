"""Per-pulsar scalar Kalman filter for continuous wave (CW) signal analysis.

This module implements the CW-mode Kalman filter where the deterministic CW signal
is subtracted from observations, yielding N independent per-pulsar scalar filters.
Each pulsar's filter has state dimension 2 + M^(n) (spin noise + timing model),
compared to the joint 4N + M_sum state in GWB mode.

The per-pulsar decomposition enables:
- Scalar innovations (no matrix inversion)
- Trivial parallelization via JAX vmap
- Non-simultaneous observations (no epoch alignment required)
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

from argus.model import get_F_block, get_Q_block
from argus.gravitational_waves import (
    compute_antenna_patterns,
    compute_cw_signal_single_pulsar,
    compute_cw_signal_single_pulsar_phase,
    gw_propagation_direction,
    pulsar_direction,
)

# kpc -> seconds (light travel time): kpc in metres / c in m/s.
KPC_TO_SECONDS = 3.0857e19 / 2.9979e8


def get_logger():
    """Get the centralized logger instance."""
    from argus.io_manager import get_argus_logger

    return get_argus_logger()


def _build_per_pulsar_H_vectors(Npsr, max_nobs, max_M, f0, design_matrices, M_dims):
    """Build observation vectors h for all pulsars, padded to common dimensions.

    For pulsar n at observation k, h^(n)_k = (1/f0^(n), 0, M^(n)_{k,1}, ..., M^(n)_{k,M^(n)})

    Parameters
    ----------
    Npsr : int
        Number of pulsars.
    max_nobs : int
        Maximum number of observations across pulsars.
    max_M : int
        Maximum timing model dimension across pulsars.
    f0 : np.ndarray
        Pulsar spin frequencies, shape (Npsr,).
    design_matrices : list of np.ndarray
        Scaled design matrices, one per pulsar.
    M_dims : np.ndarray
        Timing model dimensions per pulsar, shape (Npsr,).

    Returns
    -------
    np.ndarray
        H vectors of shape (Npsr, max_nobs, state_dim) where state_dim = 2 + max_M.
    """
    state_dim = 2 + max_M
    H = np.zeros((Npsr, max_nobs, state_dim))

    for n in range(Npsr):
        nobs_n = design_matrices[n].shape[0]
        # Spin phase coefficient: 1/f0
        H[n, :nobs_n, 0] = 1.0 / f0[n]
        # Spin frequency coefficient: 0 (already zero)
        # Timing model coefficients
        M_n = M_dims[n]
        H[n, :nobs_n, 2 : 2 + M_n] = design_matrices[n]

    return H


def _build_per_pulsar_P_eps_padded(P_eps_matrices, max_M):
    """Build padded P_eps matrices for all pulsars.

    Parameters
    ----------
    P_eps_matrices : list of np.ndarray
        Parameter covariance matrices, one per pulsar.
    max_M : int
        Maximum timing model dimension.

    Returns
    -------
    np.ndarray
        Padded P_eps matrices of shape (Npsr, max_M, max_M).
    """
    Npsr = len(P_eps_matrices)
    P_eps_padded = np.zeros((Npsr, max_M, max_M))

    for n in range(Npsr):
        M_n = P_eps_matrices[n].shape[0]
        P_eps_padded[n, :M_n, :M_n] = P_eps_matrices[n]
        # Unused entries get large diagonal (uninformative prior)
        if M_n < max_M:
            for j in range(M_n, max_M):
                P_eps_padded[n, j, j] = 1e10

    return P_eps_padded


@jax.jit
def _cw_scalar_update(x_pred, P_pred, h, R_scalar, z_tilde):
    """Scalar Kalman filter update step.

    Parameters
    ----------
    x_pred : jax.Array
        Predicted state vector, shape (state_dim, 1).
    P_pred : jax.Array
        Predicted covariance matrix, shape (state_dim, state_dim).
    h : jax.Array
        Observation vector, shape (state_dim,).
    R_scalar : float
        Scalar measurement noise variance.
    z_tilde : float
        CW-subtracted observation (scalar).

    Returns
    -------
    tuple
        (x_updated, P_updated, innovation, innovation_variance)
    """
    # Scalar innovation
    nu = z_tilde - h @ x_pred[:, 0]  # scalar

    # Scalar innovation variance
    S = h @ P_pred @ h + R_scalar  # scalar

    # Kalman gain (column vector)
    K = (P_pred @ h) / S  # shape (state_dim,)

    # State update
    x_new = x_pred[:, 0] + K * nu
    x_new = x_new[:, None]  # restore column vector shape

    # Covariance update (Joseph form for numerical stability)
    I_KhT = jnp.eye(len(h)) - jnp.outer(K, h)
    P_new = I_KhT @ P_pred @ I_KhT.T + R_scalar * jnp.outer(K, K)

    return x_new, P_new, nu, S


@jax.jit
def _cw_scalar_log_likelihood(nu, S):
    """Compute log-likelihood contribution from a single scalar innovation.

    Parameters
    ----------
    nu : float
        Scalar innovation.
    S : float
        Scalar innovation variance.

    Returns
    -------
    float
        Log-likelihood contribution.
    """
    return -0.5 * (nu**2 / S + jnp.log(2.0 * jnp.pi * S))


@partial(jax.jit, static_argnums=(1,))
def _build_F_single(gamma_p, state_dim, dt):
    """Build per-pulsar transition matrix.

    Parameters
    ----------
    gamma_p : float
        Pulsar OU damping rate.
    state_dim : int
        State vector dimension (2 + max_M).
    dt : float
        Time step.

    Returns
    -------
    jax.Array
        Transition matrix of shape (state_dim, state_dim).
    """
    F_block = get_F_block(gamma_p, dt)
    F = jnp.eye(state_dim)
    F = F.at[:2, :2].set(F_block)
    return F


@partial(jax.jit, static_argnums=(2,))
def _build_Q_single(gamma_p, sigma_p_sq, state_dim, dt, Q_eps):
    """Build per-pulsar process noise matrix.

    Parameters
    ----------
    gamma_p : float
        Pulsar OU damping rate.
    sigma_p_sq : float
        Pulsar OU driving variance (sigma_p^2).
    state_dim : int
        State vector dimension (2 + max_M).
    dt : float
        Time step.
    Q_eps : jax.Array
        Timing model process noise (typically zeros), shape (max_M, max_M).

    Returns
    -------
    jax.Array
        Process noise matrix of shape (state_dim, state_dim).
    """
    Q_block = get_Q_block(gamma_p, dt) * sigma_p_sq
    Q = jnp.zeros((state_dim, state_dim))
    Q = Q.at[:2, :2].set(Q_block)
    Q = Q.at[2:, 2:].set(Q_eps)
    return Q


def _initialize_single_pulsar(state_dim, sigma_p_sq, gamma_p, P_eps):
    """Initialize state and covariance for a single pulsar.

    Parameters
    ----------
    state_dim : int
        State vector dimension (2 + max_M).
    sigma_p_sq : float
        Pulsar OU driving variance.
    gamma_p : float
        Pulsar OU damping rate.
    P_eps : jax.Array
        Timing model covariance, shape (max_M, max_M).

    Returns
    -------
    tuple
        (x0, P0) initial state and covariance.
    """
    x0 = jnp.zeros((state_dim, 1))

    # P0 = blockdiag([[eps, 0], [0, sigma_p^2/(2*gamma_p)]], P_eps)
    P0 = jnp.zeros((state_dim, state_dim))
    P0 = P0.at[0, 0].set(1e-40)  # small phase variance
    P0 = P0.at[1, 1].set(sigma_p_sq / (2.0 * gamma_p))  # OU stationary variance
    P0 = P0.at[2:, 2:].set(P_eps)

    return x0, P0


def _run_single_pulsar_filter(
    x0,
    P0,
    z_tilde,
    h_vectors,
    R_scalars,
    dt_array,
    mask,
    gamma_p,
    sigma_p_sq,
    state_dim,
):
    """Run the Kalman filter for a single pulsar using lax.scan.

    Parameters
    ----------
    x0 : jax.Array
        Initial state, shape (state_dim, 1).
    P0 : jax.Array
        Initial covariance, shape (state_dim, state_dim).
    z_tilde : jax.Array
        CW-subtracted observations, shape (max_nobs,).
    h_vectors : jax.Array
        Observation vectors, shape (max_nobs, state_dim).
    R_scalars : jax.Array
        Measurement noise variances, shape (max_nobs,).
    dt_array : jax.Array
        Time differences, shape (max_nobs - 1,).
    mask : jax.Array
        Boolean mask for valid observations, shape (max_nobs,).
    gamma_p : float
        Pulsar OU damping rate.
    sigma_p_sq : float
        Pulsar OU driving variance.
    state_dim : int
        State vector dimension.

    Returns
    -------
    float
        Total log-likelihood for this pulsar.
    """
    Q_eps = jnp.zeros((state_dim - 2, state_dim - 2))

    # First observation: update only (no predict)
    x, P, nu, S = _cw_scalar_update(x0, P0, h_vectors[0], R_scalars[0], z_tilde[0])
    ll0 = mask[0] * _cw_scalar_log_likelihood(nu, S)

    def step(carry, inputs):
        x, P = carry
        z, h, R, dt, m = inputs

        # Predict
        F = _build_F_single(gamma_p, state_dim, dt)
        Q = _build_Q_single(gamma_p, sigma_p_sq, state_dim, dt, Q_eps)
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # Update
        x_new, P_new, nu, S_val = _cw_scalar_update(x_pred, P_pred, h, R, z)

        # Masked log-likelihood (zero for padded entries)
        ll = m * _cw_scalar_log_likelihood(nu, S_val)

        return (x_new, P_new), ll

    inputs = (
        z_tilde[1:],
        h_vectors[1:],
        R_scalars[1:],
        dt_array,
        mask[1:],
    )

    _, ll_arr = lax.scan(step, (x, P), inputs)

    return ll0 + jnp.sum(ll_arr)


class CWKalmanFilter:
    """Per-pulsar scalar Kalman filter for continuous wave signal analysis.

    In CW mode, the deterministic CW signal is subtracted from observations,
    and each pulsar is processed independently with a scalar Kalman filter.
    State vector per pulsar: [delta_phi, delta_f, delta_epsilon_1, ..., delta_epsilon_M].

    Parameters
    ----------
    data : dict
        Data dictionary from LoadWidebandPulsarData.get_processed_residuals(mode='cw'),
        containing per-pulsar observations, metadata, design matrices, and covariances.
    """

    def __init__(
        self,
        data: dict,
        include_pulsar_term: bool = False,
        phase_parameterization: bool = True,
    ):
        """Initialize the CW Kalman filter.

        Parameters
        ----------
        data : dict
            Data dictionary from LoadWidebandPulsarData.get_processed_residuals(mode='cw').
        include_pulsar_term : bool
            If True, include the pulsar term in the CW signal model.
            Default False (Earth-term only).
        phase_parameterization : bool
            If True (default), use per-pulsar phase parameters chi instead of
            physical distances for the pulsar term (arXiv 2410.10087).
            Only relevant when include_pulsar_term=True.
        """
        get_logger().info("Initializing CWKalmanFilter...")

        observations = data["processed_residuals"]
        df_psr = data["metadata"]
        pulsar_design_matrices = data["design_matrices"]
        P_eps_matrices = data["parameter_covariances"]

        self.Npsr = len(df_psr)
        self.include_pulsar_term = include_pulsar_term
        self.phase_parameterization = phase_parameterization
        get_logger().info(f"Number of pulsars: {self.Npsr}")
        get_logger().info(f"Include pulsar term: {self.include_pulsar_term}")
        get_logger().info(f"Phase parameterization: {self.phase_parameterization}")

        # Per-pulsar metadata
        self.M = df_psr["dim_M"].values.astype(int)
        self.max_M = int(np.max(self.M))
        self.state_dim = 2 + self.max_M
        self.f0 = df_psr["F0"].values

        # Pulsar sky positions (for antenna pattern computation)
        self.pulsar_ra = jnp.array(df_psr["RA"].values.astype(float))
        self.pulsar_dec = jnp.array(df_psr["DEC"].values.astype(float))

        # Pulsar distances in seconds (d/c) for distance-based pulsar term
        if include_pulsar_term and not phase_parameterization:
            distances_kpc = df_psr["distance_kpc"].values.astype(float)
            self.pulsar_distances = jnp.array(distances_kpc * KPC_TO_SECONDS)
            get_logger().info(
                f"Pulsar distances (kpc): min={distances_kpc.min():.2f}, max={distances_kpc.max():.2f}"
            )
        else:
            # Not needed for phase parameterization (chi replaces distance)
            # or Earth-term only mode
            self.pulsar_distances = jnp.zeros(self.Npsr)

        # Per-pulsar observation data
        toas_list = observations["toas"]
        residuals_list = observations["residuals"]
        errors_list = observations["errors"]
        n_obs = observations["n_obs"]

        self.max_nobs = int(np.max(n_obs))
        get_logger().info(f"Max observations per pulsar: {self.max_nobs}")
        get_logger().info(f"State dimension per pulsar: {self.state_dim}")

        # Pad per-pulsar arrays to common dimensions
        self._pad_observations(toas_list, residuals_list, errors_list, n_obs)

        # Build per-pulsar H vectors (observation model)
        H_np = _build_per_pulsar_H_vectors(
            self.Npsr,
            self.max_nobs,
            self.max_M,
            self.f0,
            pulsar_design_matrices,
            self.M,
        )
        self.jax_H = jnp.array(H_np)

        # Build padded P_eps matrices
        P_eps_np = _build_per_pulsar_P_eps_padded(P_eps_matrices, self.max_M)
        self.jax_P_eps = jnp.array(P_eps_np)

        get_logger().info("CWKalmanFilter initialization complete.")

    def _pad_observations(self, toas_list, residuals_list, errors_list, n_obs):
        """Pad per-pulsar observation arrays to common dimensions.

        Creates JAX arrays of shape (Npsr, max_nobs) with boolean mask.
        """
        toas_padded = np.zeros((self.Npsr, self.max_nobs))
        residuals_padded = np.zeros((self.Npsr, self.max_nobs))
        errors_padded = np.ones(
            (self.Npsr, self.max_nobs)
        )  # ones to avoid division by zero
        mask = np.zeros((self.Npsr, self.max_nobs), dtype=bool)
        dt_padded = np.ones((self.Npsr, self.max_nobs - 1))  # ones for safe dt

        for n in range(self.Npsr):
            nobs_n = n_obs[n]
            toas_padded[n, :nobs_n] = toas_list[n]
            residuals_padded[n, :nobs_n] = residuals_list[n]
            errors_padded[n, :nobs_n] = errors_list[n]
            mask[n, :nobs_n] = True
            if nobs_n > 1:
                dt_padded[n, : nobs_n - 1] = np.diff(toas_list[n])

        self.jax_toas = jnp.array(toas_padded)
        self.jax_residuals = jnp.array(residuals_padded)
        self.jax_errors = jnp.array(errors_padded)
        self.jax_mask = jnp.array(mask, dtype=jnp.float64)  # float for multiplication
        self.jax_dt = jnp.array(dt_padded)

    def get_likelihood(self, theta):
        """Compute the total log-likelihood for CW model parameters.

        Parameters
        ----------
        theta : CWParameters
            Parameter struct containing CW signal params and noise params.

        Returns
        -------
        float
            Total log-likelihood summed over all pulsars and observations.
        """
        return _cw_likelihood(
            theta=theta,
            toas=self.jax_toas,
            residuals=self.jax_residuals,
            errors=self.jax_errors,
            mask=self.jax_mask,
            dt=self.jax_dt,
            H=self.jax_H,
            P_eps=self.jax_P_eps,
            pulsar_ra=self.pulsar_ra,
            pulsar_dec=self.pulsar_dec,
            pulsar_distances=self.pulsar_distances,
            Npsr=self.Npsr,
            state_dim=self.state_dim,
            include_pulsar_term=self.include_pulsar_term,
            phase_parameterization=self.phase_parameterization,
        )


@partial(
    jax.jit,
    static_argnames=(
        "Npsr",
        "state_dim",
        "include_pulsar_term",
        "phase_parameterization",
    ),
)
def _cw_likelihood(
    theta,
    toas,
    residuals,
    errors,
    mask,
    dt,
    H,
    P_eps,
    pulsar_ra,
    pulsar_dec,
    pulsar_distances,
    Npsr,
    state_dim,
    include_pulsar_term,
    phase_parameterization,
):
    """Compute CW log-likelihood using vmapped per-pulsar scalar Kalman filters.

    Parameters
    ----------
    theta : CWParameters
        CW signal and noise parameters.
    toas : jax.Array
        Padded observation times, shape (Npsr, max_nobs).
    residuals : jax.Array
        Padded residuals, shape (Npsr, max_nobs).
    errors : jax.Array
        Padded TOA errors, shape (Npsr, max_nobs).
    mask : jax.Array
        Observation mask, shape (Npsr, max_nobs).
    dt : jax.Array
        Padded time differences, shape (Npsr, max_nobs-1).
    H : jax.Array
        Observation vectors, shape (Npsr, max_nobs, state_dim).
    P_eps : jax.Array
        Timing model covariances, shape (Npsr, max_M, max_M).
    pulsar_ra : jax.Array
        Pulsar right ascensions, shape (Npsr,).
    pulsar_dec : jax.Array
        Pulsar declinations, shape (Npsr,).
    pulsar_distances : jax.Array
        Pulsar distances in seconds, shape (Npsr,). Only used when
        include_pulsar_term=True and phase_parameterization=False.
    Npsr : int
        Number of pulsars.
    state_dim : int
        State vector dimension (2 + max_M).
    include_pulsar_term : bool
        Whether to include the pulsar term.
    phase_parameterization : bool
        If True, use theta.chi for pulsar term phase instead of distances.

    Returns
    -------
    float
        Total log-likelihood.
    """
    # 1. Compute antenna patterns for all pulsars
    F_plus, F_cross = compute_antenna_patterns(
        pulsar_ra, pulsar_dec, theta.alpha_gw, theta.delta_gw, theta.psi
    )

    # 2. Compute CW signal — three-way branch (resolved at JIT compile time)
    if include_pulsar_term and phase_parameterization:
        # Phase reparameterization: use per-pulsar chi from theta
        cw_signal = jax.vmap(
            lambda t, fp, fc, ch: compute_cw_signal_single_pulsar_phase(
                t, theta.f_gw, theta.h0, theta.cos_iota, theta.Phi0, fp, fc, ch
            )
        )(toas, F_plus, F_cross, theta.chi)
    elif include_pulsar_term:
        # Distance-based pulsar term (original implementation)
        n_hat = gw_propagation_direction(theta.alpha_gw, theta.delta_gw)
        geometric_factors = jax.vmap(
            lambda ra, dec: 1.0 + jnp.dot(n_hat, pulsar_direction(ra, dec))
        )(pulsar_ra, pulsar_dec)
        cw_signal = jax.vmap(
            lambda t, fp, fc, d, g: compute_cw_signal_single_pulsar(
                t, theta.f_gw, theta.h0, theta.cos_iota, theta.Phi0, fp, fc, d, g
            )
        )(toas, F_plus, F_cross, pulsar_distances, geometric_factors)
    else:
        # Earth-term only (no pulsar term)
        cw_signal = jax.vmap(
            lambda t, fp, fc, d, g: compute_cw_signal_single_pulsar(
                t, theta.f_gw, theta.h0, theta.cos_iota, theta.Phi0, fp, fc, d, g
            )
        )(toas, F_plus, F_cross, pulsar_distances, jnp.zeros(Npsr))

    # 3. Subtract CW signal from observations
    z_tilde = residuals - cw_signal

    # 4. Compute per-pulsar scalar R
    R_scalars = jnp.square(theta.EFAC[:, None] * errors) + jnp.square(
        theta.EQUAD[:, None]
    )  # shape (Npsr, max_nobs)

    # 5. Initialize per-pulsar states
    sigma_p_sq = theta.sigma_p**2

    def run_one_pulsar(x0, P0, z, h, R, dt_arr, m, gp, sp2):
        return _run_single_pulsar_filter(x0, P0, z, h, R, dt_arr, m, gp, sp2, state_dim)

    # Build initial states for all pulsars
    x0_all = jnp.zeros((Npsr, state_dim, 1))
    P0_all = jax.vmap(
        lambda sp2, gp, pe: _initialize_single_pulsar(state_dim, sp2, gp, pe)[1]
    )(sigma_p_sq, theta.gamma_p, P_eps)

    # 6. Run vmapped per-pulsar filters
    ll_per_pulsar = jax.vmap(run_one_pulsar)(
        x0_all,
        P0_all,
        z_tilde,
        H,
        R_scalars,
        dt,
        mask,
        theta.gamma_p,
        sigma_p_sq,
    )

    # 7. Sum log-likelihoods over pulsars
    return jnp.sum(ll_per_pulsar)
